"""
Hardware capability probe.

The premise: **support flags lie, and measurement doesn't.**

A Tesla P40 will happily accept a ``bfloat16`` tensor and run a matmul on it
without raising anything.  It has no bf16 hardware whatsoever -- torch quietly
converts to fp32 and back, and the result is roughly three times *slower* than
just asking for fp32 in the first place.  Nothing in the API tells you this.
``torch.cuda.is_bf16_supported()`` says True.

So this module does two things and reports them side by side:

  * what the architecture **claims**, derived from compute capability
  * what the card **does**, measured in achieved TFLOP/s

When those disagree, the measurement wins, and the disagreement is the finding.

The same applies to Triton.  Conventional wisdom says Triton needs SM 7.0+.
On a P40 (SM 6.1) it compiles and runs correctly -- verified.  What actually
hurts is the autotune space: config sets designed around 164 KB of shared
memory and tensor cores, benchmarked one at a time on a card that has 48 KB
and neither.  So this probe measures cold-compile cost and reports the
autotune tax explicitly, instead of asserting a version floor that is wrong.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

__all__ = [
    "ArchFacts",
    "DeviceInfo",
    "DtypeReport",
    "TritonReport",
    "ProbeResult",
    "arch_facts",
    "device_info",
    "measure_dtype",
    "probe_triton",
    "probe",
    "main",
]

# Square matmul edge used for throughput measurement.  Big enough that the
# kernel dominates launch overhead on a 2016 card, small enough to fit in the
# smallest board this is meant to run on.
_MATMUL_N = 2048

# Typical number of configs in an upstream Triton autotune space (fla's
# chunk kernels sit around here).  Used only to turn a measured cold-compile
# cost into a legible "this is what warmup will cost you" number.
_TYPICAL_AUTOTUNE_CONFIGS = 40

# How much faster a reduced-precision dtype must measure before it is worth
# the numerical trouble.  A few percent is noise dressed as an optimisation;
# see ProbeResult.recommended_dtype for the P40 case that set this.
_WORTH_THE_NUMERICS = 1.15


@dataclass(frozen=True)
class ArchFacts:
    """What the compute capability *claims*, before any measurement."""

    compute_capability: tuple[int, int]
    tensor_cores: bool           # SM 7.0+ (Volta)
    bf16_native: bool            # SM 8.0+ (Ampere)
    fp16_fast: bool              # SM 7.0+; Pascal GP102 runs fp16 at 1/64 rate
    tf32_available: bool         # SM 8.0+
    generation: str

    @property
    def sm(self) -> float:
        major, minor = self.compute_capability
        return major + minor / 10.0


_GENERATIONS: tuple[tuple[int, str], ...] = (
    (10, "Blackwell"),
    (9, "Hopper"),
    (8, "Ampere/Ada"),
    (7, "Volta/Turing"),
    (6, "Pascal"),
    (5, "Maxwell"),
    (3, "Kepler"),
)


def arch_facts(capability: tuple[int, int]) -> ArchFacts:
    """Derive architectural claims from a compute capability.

    Pure function, no device access -- so it is testable on a CPU-only box.
    """
    major, minor = capability
    generation = next((name for floor, name in _GENERATIONS if major >= floor), "pre-Kepler")
    return ArchFacts(
        compute_capability=capability,
        tensor_cores=major >= 7,
        bf16_native=major >= 8,
        fp16_fast=major >= 7,
        tf32_available=major >= 8,
        generation=generation,
    )


@dataclass(frozen=True)
class DeviceInfo:
    index: int
    name: str
    uuid: str
    total_memory_mb: int
    shared_memory_per_block_kb: int
    multiprocessors: int
    arch: ArchFacts

    @property
    def shared_memory_verdict(self) -> str:
        """Shared memory is the ceiling that decides which block sizes fit.

        Ampere offers 164 KB opt-in, Hopper 228 KB.  Pascal has 48 KB and no
        opt-in path, which is why large-BK/BV autotune configs are dead weight.
        """
        kb = self.shared_memory_per_block_kb
        if kb >= 200:
            return "large (Hopper-class); nearly all upstream block sizes fit"
        if kb >= 160:
            return "ample (Ampere-class); most upstream block sizes fit"
        if kb >= 96:
            return "moderate; some large block configs will not fit"
        return "small; many upstream autotune configs cannot fit and are wasted work"


def device_info(index: int = 0) -> DeviceInfo:
    """Collect static properties for one CUDA device."""
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device available")
    props = torch.cuda.get_device_properties(index)
    capability = torch.cuda.get_device_capability(index)
    # uuid is not exposed on every torch build; fall back rather than fail.
    raw_uuid = getattr(props, "uuid", None)
    uuid = str(raw_uuid) if raw_uuid is not None else "unavailable"
    shared = getattr(props, "shared_memory_per_block_optin", None) or props.shared_memory_per_block
    return DeviceInfo(
        index=index,
        name=props.name,
        uuid=uuid,
        total_memory_mb=props.total_memory // (1024 * 1024),
        shared_memory_per_block_kb=shared // 1024,
        multiprocessors=props.multi_processor_count,
        arch=arch_facts(capability),
    )


@dataclass(frozen=True)
class DtypeReport:
    """Measured, not claimed."""

    dtype: str
    runs: bool
    tflops: float | None = None
    error: str | None = None
    relative_to_fp32: float | None = None

    @property
    def emulated(self) -> bool:
        """Slower than fp32 means there is no hardware for it behind the API."""
        return self.relative_to_fp32 is not None and self.relative_to_fp32 < 0.95


def measure_dtype(
    dtype: torch.dtype,
    index: int = 0,
    n: int = _MATMUL_N,
    warmup: int = 3,
    reps: int = 10,
) -> DtypeReport:
    """Measure achieved TFLOP/s for a square matmul in ``dtype``.

    Returns a report with ``runs=False`` rather than raising, because "this
    dtype errors on this card" is a legitimate finding the caller wants in the
    table next to the ones that worked.
    """
    device = torch.device(f"cuda:{index}")
    name = str(dtype).removeprefix("torch.")
    try:
        a = torch.randn(n, n, device=device, dtype=dtype)
        b = torch.randn(n, n, device=device, dtype=dtype)
        for _ in range(warmup):
            a @ b
        torch.cuda.synchronize(index)
        start = time.perf_counter()
        for _ in range(reps):
            a @ b
        torch.cuda.synchronize(index)
        elapsed = (time.perf_counter() - start) / reps
        flops = 2.0 * n**3
        return DtypeReport(dtype=name, runs=True, tflops=flops / elapsed / 1e12)
    except Exception as exc:  # noqa: BLE001 -- a failure here is data, not a crash
        return DtypeReport(dtype=name, runs=False, error=f"{type(exc).__name__}: {exc}"[:200])
    finally:
        torch.cuda.empty_cache()


@dataclass(frozen=True)
class TritonReport:
    available: bool
    version: str | None = None
    elementwise_ok: bool = False
    dot_ok: bool = False
    max_dot_block: int | None = None
    cold_compile_s: float | None = None
    error: str | None = None

    @property
    def autotune_tax_s(self) -> float | None:
        """Rough cost of one upstream autotune sweep, per kernel per shape/dtype key."""
        if self.cold_compile_s is None:
            return None
        return self.cold_compile_s * _TYPICAL_AUTOTUNE_CONFIGS


def probe_triton(index: int = 0, blocks: tuple[int, ...] = (16, 32, 64, 128)) -> TritonReport:
    """Find out whether Triton actually works here, and at what block sizes.

    Does not consult the compute capability.  Compiles and runs real kernels,
    because the version floors in circulation are wrong for Pascal.
    """
    try:
        import triton
        import triton.language as tl
    except Exception as exc:  # noqa: BLE001
        return TritonReport(available=False, error=f"{type(exc).__name__}: {exc}"[:200])

    device = torch.device(f"cuda:{index}")

    @triton.jit
    def _add(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask), mask=mask)

    @triton.jit
    def _dot(a_ptr, b_ptr, c_ptr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        idx = offs[:, None] * BLOCK + offs[None, :]
        tl.store(c_ptr + idx, tl.dot(tl.load(a_ptr + idx), tl.load(b_ptr + idx)))

    elementwise_ok = False
    cold_compile_s: float | None = None
    try:
        n = 1024
        x = torch.rand(n, device=device)
        y = torch.rand(n, device=device)
        out = torch.empty_like(x)
        start = time.perf_counter()
        _add[(1,)](x, y, out, n, BLOCK=1024)
        torch.cuda.synchronize(index)
        cold_compile_s = time.perf_counter() - start
        elementwise_ok = bool(torch.allclose(out, x + y))
    except Exception as exc:  # noqa: BLE001
        return TritonReport(
            available=True,
            version=getattr(triton, "__version__", None),
            error=f"elementwise: {type(exc).__name__}: {exc}"[:200],
        )

    max_dot_block: int | None = None
    for block in blocks:
        try:
            a = torch.randn(block, block, device=device)
            b = torch.randn(block, block, device=device)
            c = torch.empty_like(a)
            _dot[(1,)](a, b, c, BLOCK=block)
            torch.cuda.synchronize(index)
            if torch.allclose(c, a @ b, atol=1e-3, rtol=1e-3):
                max_dot_block = block
        except Exception:  # noqa: BLE001 -- expected once shared memory runs out
            break

    return TritonReport(
        available=True,
        version=getattr(triton, "__version__", None),
        elementwise_ok=elementwise_ok,
        dot_ok=max_dot_block is not None,
        max_dot_block=max_dot_block,
        cold_compile_s=cold_compile_s,
    )


@dataclass
class ProbeResult:
    device: DeviceInfo
    dtypes: list[DtypeReport]
    triton: TritonReport
    torch_version: str
    python_version: str
    notes: list[str] = field(default_factory=list)

    @property
    def recommended_dtype(self) -> str:
        """The fastest dtype worth its numerical cost -- not simply the fastest.

        Ranking by raw matmul throughput gives the wrong answer on pre-Volta
        hardware.  A P40 measures fp16 ~8% *faster* than fp32, because it has
        no fp16 compute at all: cuBLAS widens to fp32 internally and the only
        gain is halved memory traffic.  That 8% does not survive the loss
        scaling that fp16 training then requires, and it buys none of bf16's
        exponent range.  So a reduced-precision dtype has to clear two bars:
        the hardware must actually have a path for it, and the measured win
        must be large enough to be worth the numerics.
        """
        ran = [d for d in self.dtypes if d.runs and d.tflops is not None]
        if not ran:
            return "unknown"
        if not any(d.dtype == "float32" for d in ran):
            return max(ran, key=lambda d: d.tflops or 0.0).dtype

        arch = self.device.arch
        viable = [
            d
            for d in ran
            if d.dtype != "float32"
            and (d.relative_to_fp32 or 0.0) >= _WORTH_THE_NUMERICS
            and not (d.dtype == "float16" and not arch.fp16_fast)
            and not (d.dtype == "bfloat16" and not arch.bf16_native)
        ]
        if not viable:
            return "float32"
        return max(viable, key=lambda d: d.tflops or 0.0).dtype

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": asdict(self.device),
            "dtypes": [asdict(d) for d in self.dtypes],
            "triton": asdict(self.triton),
            "torch_version": self.torch_version,
            "python_version": self.python_version,
            "recommended_dtype": self.recommended_dtype,
            "notes": self.notes,
        }


def _build_notes(device: DeviceInfo, dtypes: list[DtypeReport], triton: TritonReport) -> list[str]:
    """Turn measurements into the sentences a person would actually act on."""
    notes: list[str] = []
    arch = device.arch

    for report in dtypes:
        if not report.runs or report.dtype == "float32":
            continue
        relative = report.relative_to_fp32 or 0.0
        if report.emulated:
            notes.append(
                f"{report.dtype} runs but is {1 / (relative or 1):.1f}x SLOWER than "
                f"float32 -- there is no {report.dtype} hardware here, only emulation. Do not use it."
            )
        elif (report.dtype == "float16" and not arch.fp16_fast) or (
            report.dtype == "bfloat16" and not arch.bf16_native
        ):
            notes.append(
                f"{report.dtype} measures {relative:.2f}x fp32, but this card has no {report.dtype} "
                "compute path -- cuBLAS widens to fp32 internally, so the gain is memory traffic "
                "only. Not recommended: it does not survive the loss scaling reduced precision "
                "then requires."
            )

    if not arch.tensor_cores:
        notes.append(
            "No tensor cores (SM < 7.0). fp32 GEMM via cuBLAS is this card's strongest path; "
            "kernels hand-tuned for tensor cores will tend to lose to plain torch autograd."
        )
    if not arch.bf16_native:
        notes.append(
            "bfloat16 is not native (SM < 8.0). torch.cuda.is_bf16_supported() may still "
            "return True -- it reports API acceptance, not hardware."
        )

    if triton.available and triton.dot_ok:
        notes.append(
            f"Triton {triton.version} WORKS here (tl.dot up to block {triton.max_dot_block}), "
            "despite the widely repeated SM 7.0 floor."
        )
        tax = triton.autotune_tax_s
        if tax is not None and not arch.tensor_cores:
            notes.append(
                f"Autotune tax: ~{tax:.0f}s per kernel per (shape, dtype) key, estimated from a "
                f"{triton.cold_compile_s:.2f}s cold compile x {_TYPICAL_AUTOTUNE_CONFIGS} configs. "
                "Pin configs or persist TRITON_CACHE_DIR; most upstream configs cannot fit here anyway."
            )
    elif triton.available:
        notes.append(f"Triton imports but tl.dot failed: {triton.error or 'unknown'}")
    else:
        notes.append(f"Triton unavailable: {triton.error or 'not installed'} -- fp32 torch paths only.")

    notes.append(f"Shared memory {device.shared_memory_per_block_kb} KB/block: {device.shared_memory_verdict}.")
    return notes


def probe(index: int = 0, skip_triton: bool = False) -> ProbeResult:
    """Run the full probe against one CUDA device."""
    device = torch.device(f"cuda:{index}")
    torch.cuda.set_device(device)
    info = device_info(index)

    dtypes = [measure_dtype(dt, index=index) for dt in (torch.float32, torch.float16, torch.bfloat16)]
    baseline = next((d.tflops for d in dtypes if d.dtype == "float32" and d.tflops), None)
    if baseline:
        dtypes = [
            DtypeReport(
                dtype=d.dtype,
                runs=d.runs,
                tflops=d.tflops,
                error=d.error,
                relative_to_fp32=(d.tflops / baseline) if d.tflops else None,
            )
            for d in dtypes
        ]

    triton_report = (
        TritonReport(available=False, error="skipped by request")
        if skip_triton
        else probe_triton(index=index)
    )

    return ProbeResult(
        device=info,
        dtypes=dtypes,
        triton=triton_report,
        torch_version=torch.__version__,
        python_version=platform.python_version(),
        notes=_build_notes(info, dtypes, triton_report),
    )


def format_report(result: ProbeResult) -> str:
    """Human-readable rendering.  Shape first, numbers second."""
    device, arch = result.device, result.device.arch
    lines: list[str] = []
    add = lines.append

    add("=" * 74)
    add(f"  {device.name}  (cuda:{device.index})")
    add("=" * 74)
    add(f"  architecture       {arch.generation}, SM {arch.sm:.1f}")
    add(f"  memory             {device.total_memory_mb} MB across {device.multiprocessors} SMs")
    add(f"  shared mem/block   {device.shared_memory_per_block_kb} KB")
    add(f"  torch / python     {result.torch_version} / {result.python_version}")
    add("")
    add("  CLAIMED (from compute capability)")
    for label, value in (
        ("tensor cores", arch.tensor_cores),
        ("bf16 native", arch.bf16_native),
        ("fp16 fast path", arch.fp16_fast),
        ("tf32", arch.tf32_available),
    ):
        add(f"    {label:<18} {'yes' if value else 'no'}")
    add("")
    add("  MEASURED (achieved TFLOP/s, 2048^3 matmul)")
    add(f"    {'dtype':<12} {'TFLOP/s':>9}  {'vs fp32':>8}   status")
    for report in result.dtypes:
        if not report.runs:
            add(f"    {report.dtype:<12} {'-':>9}  {'-':>8}   FAILED: {report.error}")
            continue
        rel = f"{report.relative_to_fp32:.2f}x" if report.relative_to_fp32 else "-"
        status = "EMULATED (slower than fp32)" if report.emulated and report.dtype != "float32" else "ok"
        add(f"    {report.dtype:<12} {report.tflops:>9.2f}  {rel:>8}   {status}")
    add("")
    add("  TRITON")
    triton = result.triton
    if triton.available:
        add(f"    version            {triton.version}")
        add(f"    elementwise        {'ok' if triton.elementwise_ok else 'FAILED'}")
        add(f"    tl.dot             {'ok up to block ' + str(triton.max_dot_block) if triton.dot_ok else 'FAILED'}")
        if triton.cold_compile_s is not None:
            add(f"    cold compile       {triton.cold_compile_s:.2f}s")
            add(f"    est. autotune tax  ~{triton.autotune_tax_s:.0f}s per kernel per (shape, dtype)")
    else:
        add(f"    unavailable        {triton.error}")
    add("")
    add(f"  RECOMMENDED DTYPE    {result.recommended_dtype}")
    add("")
    add("  NOTES")
    for note in result.notes:
        add(f"    * {note}")
    add("=" * 74)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lumen-probe",
        description="Measure what a GPU actually does, rather than what its support flags claim.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device index to probe")
    parser.add_argument("--all", action="store_true", help="Probe every visible CUDA device")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a formatted report")
    parser.add_argument("--skip-triton", action="store_true", help="Skip Triton probing (it compiles kernels)")
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print("no CUDA device available", file=sys.stderr)
        return 1

    indices = range(torch.cuda.device_count()) if args.all else [args.device]
    results = [probe(index=i, skip_triton=args.skip_triton) for i in indices]

    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
    else:
        print("\n".join(format_report(r) for r in results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
