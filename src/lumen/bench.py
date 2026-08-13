"""
Layer benchmarking, with the backward pass treated as a first-class citizen.

Most published kernel benchmarks report the forward pass.  Training does not
care about the forward pass in isolation, and on pre-Ampere hardware the two
can point in opposite directions.

Measured on a Tesla P40 (SM 6.1), a crossed-group gated delta net in plain
fp32 torch against fla's Triton Gated DeltaNet at matched parameter count:

    T      forward         backward        end-to-end
    512    triton 1.21x    torch 1.98x     torch 1.58x
    1024   triton 1.35x    torch 2.01x     torch 1.60x
    2048   triton 1.46x    torch 2.06x     torch 1.62x
    4096   triton 1.51x    torch 2.07x     torch 1.62x

Triton wins every forward and loses every backward, and the backward is the
larger term, so the hand-rolled fp32 implementation wins overall by a stable
~1.6x.  A forward-only benchmark would have reported the exact opposite
conclusion, with a straight face.

So: ``fwd_ms``, ``bwd_ms`` and ``fwd_bwd_ms`` are always reported separately.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

__all__ = ["BenchResult", "benchmark_layer", "compare", "format_table"]


def _unwrap(output: Any) -> torch.Tensor:
    """Layers differ on whether they return a tensor or a tuple; take the tensor."""
    while isinstance(output, (tuple, list)):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"layer produced {type(output).__name__}, expected a Tensor")
    return output


@dataclass(frozen=True)
class BenchResult:
    label: str
    dtype: str
    seq_len: int
    batch: int
    params: int
    fwd_ms: float
    fwd_bwd_ms: float
    error: str | None = None

    @property
    def bwd_ms(self) -> float:
        return self.fwd_bwd_ms - self.fwd_ms

    @property
    def tokens_per_s(self) -> float:
        if self.fwd_bwd_ms <= 0:
            return 0.0
        return self.batch * self.seq_len / (self.fwd_bwd_ms / 1000.0)


def _sync(device: torch.device) -> None:
    """Block until `device` has finished the work already queued on it.

    On CUDA this is what makes the timing mean anything: a launch returns
    immediately, so timing without it measures the dispatch and not the kernel.
    On CPU the work is complete when the call returns and there is nothing to
    wait for, so the honest implementation is to do nothing.

    Branching on `device.type` rather than on `torch.cuda.is_available()`: the
    question is what this measurement is running on, not what the machine has.
    A CPU cell on a GPU box must take the CPU path.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time(fn: Callable[[], Any], warmup: int, reps: int, device: torch.device) -> float:
    """Median wall-clock ms per call.

    Median rather than mean: the first timed call after warmup often still
    carries JIT or allocator noise, and one outlier should not move the number.
    """
    for _ in range(warmup):
        fn()
    _sync(device)
    samples: list[float] = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        _sync(device)
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def benchmark_layer(
    layer: nn.Module,
    *,
    label: str,
    batch: int = 4,
    seq_len: int = 1024,
    d_model: int = 512,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cuda:0",
    warmup: int = 5,
    reps: int = 15,
) -> BenchResult:
    """Time one layer's forward and forward+backward on ``[batch, seq_len, d_model]``.

    Failures are captured into the result rather than raised, so one broken
    implementation does not abort a comparison table.
    """
    device = torch.device(device)
    dtype_name = str(dtype).removeprefix("torch.")
    params = sum(p.numel() for p in layer.parameters())
    try:
        layer = layer.to(device=device, dtype=dtype)
        x = torch.randn(batch, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)

        def forward() -> torch.Tensor:
            return _unwrap(layer(x))

        def forward_backward() -> None:
            # .float() before the loss so fp16/bf16 runs do not silently
            # underflow the scalar and produce a meaningless gradient.
            forward().float().pow(2).mean().backward()

        fwd_ms = _time(forward, warmup, reps, device)
        fwd_bwd_ms = _time(forward_backward, warmup, reps, device)
        return BenchResult(
            label=label, dtype=dtype_name, seq_len=seq_len, batch=batch,
            params=params, fwd_ms=fwd_ms, fwd_bwd_ms=fwd_bwd_ms,
        )
    except Exception as exc:  # noqa: BLE001 -- a failure is a table row, not a crash
        return BenchResult(
            label=label, dtype=dtype_name, seq_len=seq_len, batch=batch,
            params=params, fwd_ms=float("nan"), fwd_bwd_ms=float("nan"),
            error=f"{type(exc).__name__}: {exc}"[:160],
        )
    finally:
        # Same condition as `_sync`, for a different reason.  This one does not
        # raise on CPU -- `empty_cache()` returns early when CUDA was never
        # initialised -- so it is correct by accident on a CPU-only box and
        # wrong everywhere else: on a GPU box, timing a CPU layer would flush
        # the CUDA allocator once per cell for no reason.  A measurement should
        # not have side effects on a device it is not measuring.
        if device.type == "cuda":
            torch.cuda.empty_cache()


def compare(
    builders: Mapping[str, Callable[[], nn.Module]],
    *,
    seq_lens: Sequence[int] = (512, 1024, 2048),
    dtypes: Sequence[torch.dtype] = (torch.float32,),
    batch: int = 4,
    d_model: int = 512,
    device: torch.device | str = "cuda:0",
    warmup: int = 5,
    reps: int = 15,
) -> list[BenchResult]:
    """Benchmark several layers across sequence lengths and dtypes.

    ``builders`` maps a label to a zero-argument factory rather than to a
    module, so every (seq_len, dtype) cell gets a freshly constructed layer.
    Reusing one instance across cells lets allocator state and any internal
    caching leak between measurements.
    """
    results: list[BenchResult] = []
    for seq_len in seq_lens:
        for dtype in dtypes:
            for label, build in builders.items():
                results.append(
                    benchmark_layer(
                        build(), label=label, batch=batch, seq_len=seq_len,
                        d_model=d_model, dtype=dtype, device=device,
                        warmup=warmup, reps=reps,
                    )
                )
    return results


def format_table(results: Sequence[BenchResult]) -> str:
    """Render results, grouped by sequence length, with a ratio against the fastest."""
    if not results:
        return "(no results)"

    header = ("T", "impl", "dtype", "params", "fwd ms", "bwd ms", "fwd+bwd ms", "tok/s", "vs best")
    rows: list[tuple[str, ...]] = []
    for seq_len in sorted({r.seq_len for r in results}):
        group = [r for r in results if r.seq_len == seq_len]
        ok = [r for r in group if r.error is None]
        best = min((r.fwd_bwd_ms for r in ok), default=None)
        for r in group:
            if r.error is not None:
                rows.append((str(seq_len), r.label, r.dtype, f"{r.params/1e6:.2f}M",
                             "-", "-", "FAILED", "-", r.error[:30]))
                continue
            ratio = f"{r.fwd_bwd_ms / best:.2f}x" if best else "-"
            rows.append((
                str(seq_len), r.label, r.dtype, f"{r.params/1e6:.2f}M",
                f"{r.fwd_ms:.1f}", f"{r.bwd_ms:.1f}", f"{r.fwd_bwd_ms:.1f}",
                f"{r.tokens_per_s/1000:.1f}k", ratio,
            ))

    widths = [max(len(str(row[i])) for row in (header, *rows)) for i in range(len(header))]
    line = "-+-".join("-" * w for w in widths)
    out = [" | ".join(h.ljust(widths[i]) for i, h in enumerate(header)), line]
    out.extend(" | ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)) for row in rows)
    return "\n".join(out)
