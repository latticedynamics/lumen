"""`backend="triton"` is a faster path for Sparse Delta Memory, not a different model.

The claim this file holds: Lumen's Triton kernels and the reference compute
**the same function**, forward and backward, to fp32 round-off -- one algorithm
reached by two routes, which is what earns a tight tolerance.  The yardstick is
the fp64 sequential oracle, and the question asked of the Triton path is the
one ``test_gdn_backend.py`` asks of fla: not "do the two agree" but "is it as
close to the recurrence as the path already trusted".

The kernels are new code with new failure modes, and the tests are aimed at
them:

* **collisions** -- slots written several times in one chunk, read and written
  at one position, read twice at one position.  The terms kernel walks each
  slot's segment of writes, and the walk stores only a slot's first writer; a
  fixture that never collides would pass both without exercising either;
* **edges of the tiling** -- a chunk of 1, a ragged last chunk, one write per
  position, a value width that is not a multiple of any block;
* **the table across calls** -- a state handed back in must continue exactly
  where it left off, which is the arena and the walk agreeing about the table;
* **where the kernels do not run** -- a CPU tensor, fp64, a ``torch.func``
  transform -- which must be the reference exactly, not approximately.

The config-surface and fallback tests need no GPU.  The numerical ones need
CUDA and a working triton, and carry the project's markers.
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn.functional as F

from lumen.sdm import SparseDeltaMemory, SparseDeltaMemoryConfig
from lumen.sdm import reference as ref
from lumen.sdm import triton_kernels as tk

# fp32 round-off on these shapes, as a relative norm.  The two routes sum the
# same products in different orders; demanding more would specify a tolerance
# the arithmetic cannot meet.
FP32_ROUNDOFF = 5e-6

requires_cuda = pytest.mark.gpu
requires_triton = pytest.mark.triton
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
needs_triton = pytest.mark.skipif(not tk.HAS_TRITON, reason="triton is not installed")

DIFFERENTIABLE = ("memory", "write_val", "log_decay", "v", "beta", "read_val")


def draw(
    *,
    batch: int = 2,
    heads: int = 2,
    seq_len: int = 48,
    n_slots: int = 24,
    n_writes: int = 4,
    n_reads: int = 3,
    d_v: int = 5,
    shared_memory: bool = True,
    crowd: int = 0,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Kernel inputs obeying the layer's invariants, in fp64 on the host.

    Distinct writes per position, softmax weights, `β ∈ (0, 2)`, log-decays
    from near a full wipe to almost none.  **The table is small on purpose**, as
    in ``test_sdm.py``: 24 slots against 4 writes per position collide in every
    chunk.  ``crowd`` > 0 confines writes AND reads to that many slots, which
    also puts the same slot in one position's reads twice.  ``shared_memory``
    gives one `(H, N, d_v)` table broadcast over the batch, as a learned
    initial table is.
    """
    g = torch.Generator().manual_seed(seed)
    lead = (batch, heads)
    pool = crowd if crowd else n_slots
    write_idx = torch.rand(*lead, seq_len, pool, generator=g).argsort(-1)[..., :n_writes]
    if crowd:
        read_idx = torch.randint(0, pool, (*lead, seq_len, n_reads), generator=g)
    else:
        read_idx = torch.rand(*lead, seq_len, n_slots, generator=g).argsort(-1)[..., :n_reads]
    randn = lambda *shape: torch.randn(*shape, generator=g, dtype=torch.float64)  # noqa: E731
    log_alpha = -F.softplus(2.0 * randn(*lead, seq_len))
    memory_lead = (heads,) if shared_memory else lead
    return {
        "memory": randn(*memory_lead, n_slots, d_v),
        "write_idx": write_idx,
        "write_val": torch.softmax(randn(*lead, seq_len, n_writes), -1),
        "log_decay": log_alpha.unsqueeze(-1).expand(*lead, seq_len, n_writes).contiguous(),
        "v": randn(*lead, seq_len, d_v),
        "beta": 2.0 * torch.sigmoid(randn(*lead, seq_len)),
        "read_idx": read_idx,
        "read_val": torch.softmax(randn(*lead, seq_len, n_reads), -1),
    }


def place(inputs: dict[str, torch.Tensor], device: str, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Leaves on ``device`` in ``dtype``, the differentiable ones wanting gradients."""
    out = {}
    for name, x in inputs.items():
        x = x.to(device)
        if name in DIFFERENTIABLE:
            x = x.to(dtype).requires_grad_(True)
        out[name] = x
    return out


ARGS = ("memory", "write_idx", "write_val", "log_decay", "v", "beta", "read_idx", "read_val")


def run(
    kernel, inputs: dict[str, torch.Tensor], chunk: int, seed: int = 1, **kwargs: object
) -> dict[str, torch.Tensor]:
    """Outputs, final table and every input gradient under one fixed upstream signal."""
    out, final = kernel(*(inputs[a] for a in ARGS), chunk_size=chunk, **kwargs)
    g = torch.Generator().manual_seed(seed)
    signal_out = torch.randn(out.shape, generator=g, dtype=torch.float64).to(out)
    signal_final = torch.randn(final.shape, generator=g, dtype=torch.float64).to(final)
    leaves = [inputs[name] for name in DIFFERENTIABLE]
    grads = torch.autograd.grad((out, final), leaves, (signal_out, signal_final))
    result = {"out": out.detach(), "final": final.detach()}
    result.update({f"d_{name}": grad for name, grad in zip(DIFFERENTIABLE, grads)})
    return result


def oracle(inputs: dict[str, torch.Tensor], chunk: int) -> dict[str, torch.Tensor]:
    """The sequential recurrence, in fp64.  It does not broadcast a shared table."""

    def sequential(memory, *rest, chunk_size):  # noqa: ANN001, ANN202
        return ref.sequential_sparse_delta(
            memory.expand(*rest[0].shape[:-2], *memory.shape[-2:]), *rest
        )

    return run(sequential, place(inputs, "cuda", torch.float64), chunk)


def distance(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


# ── the config surface: no device needed ──────────────────────────────────


def _config(**overrides: object) -> SparseDeltaMemoryConfig:
    base: dict[str, object] = dict(d_model=32, n_heads=2, n_slots=16**2, initial_memory="zero")
    base.update(overrides)
    return SparseDeltaMemoryConfig(**base)  # type: ignore[arg-type]


def test_reference_is_the_default() -> None:
    """Opt-in, never auto-detected -- even with triton importable and a GPU present."""
    assert _config().backend == "reference"


def test_an_unknown_backend_is_refused() -> None:
    with pytest.raises(ValueError, match="backend must be one of"):
        _config(backend="fla")


def test_triton_without_triton_is_refused_at_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refused where the layer is built, not several thousand steps into a run."""
    monkeypatch.setattr(tk, "HAS_TRITON", False)
    with pytest.raises(RuntimeError, match="triton did not import"):
        SparseDeltaMemory(_config(backend="triton"))


def test_the_backend_is_in_the_repr() -> None:
    """Which code produced a checkpoint has to be visible without the config."""
    assert "backend=reference" in repr(SparseDeltaMemory(_config()))


# ── where the kernels do not run: the reference, exactly ─────────────────


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["fp32", "fp64"])
def test_a_cpu_tensor_runs_the_reference_exactly(dtype: torch.dtype) -> None:
    """Not "close to": the same code, so the same bits, outputs and gradients."""
    inputs = draw()
    got = run(tk.chunk_sparse_delta, place(inputs, "cpu", dtype), 8)
    want = run(ref.chunk_sparse_delta, place(inputs, "cpu", dtype), 8)
    for name in got:
        assert torch.equal(got[name], want[name]), name


@requires_cuda
@needs_cuda
def test_fp64_on_cuda_runs_the_reference_exactly() -> None:
    """The kernels are compiled for fp32; fp64 is the reference's, which is the
    correct answer rather than a limitation -- it is what the oracle runs in.

    Under deterministic algorithms, because "exactly" is otherwise not a claim
    the reference can make about itself on CUDA: its gather backward is a
    scatter-add, and two runs of the same code differ in the last bits.
    ``warn_only`` for the reason ``test_sdm.py`` gives -- cuBLAS wants a
    workspace setting no test can arrange, and the equality is the check.
    """
    inputs = draw()
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            got = run(tk.chunk_sparse_delta, place(inputs, "cuda", torch.float64), 8)
            want = run(ref.chunk_sparse_delta, place(inputs, "cuda", torch.float64), 8)
    finally:
        torch.use_deterministic_algorithms(previous)
    for name in got:
        assert torch.equal(got[name], want[name]), name


def test_vmap_runs_the_reference() -> None:
    """A ``torch.func`` transform cannot see into the kernels; the reference's
    functional holder runs instead, as it does for the reference itself."""
    inputs = place(draw(batch=3, heads=1, shared_memory=False), "cpu", torch.float64)
    args = [inputs[a].detach() for a in ARGS]

    def call(kernel):  # noqa: ANN001, ANN202
        return torch.func.vmap(
            lambda *xs: kernel(*xs, chunk_size=8, check_writes=False)
        )(*args)

    for got, want in zip(call(tk.chunk_sparse_delta), call(ref.chunk_sparse_delta)):
        assert torch.equal(got, want)


# ── the numerics: what licenses the switch ────────────────────────────────

CASES = {
    # name: (draw overrides, chunk)
    "default": (dict(), 8),
    "ragged-last-chunk": (dict(seq_len=45), 8),
    "chunk-of-one": (dict(seq_len=12), 1),
    "chunk-not-a-power-of-two": (dict(seq_len=50), 6),
    "one-write-one-read": (dict(n_writes=1, n_reads=1), 8),
    "crowded": (dict(crowd=6, n_reads=4, seq_len=40), 8),
    "wide-values": (dict(d_v=40, n_slots=64, n_writes=8, n_reads=8, seq_len=64), 16),
    "per-stream-table": (dict(shared_memory=False), 8),
    "realistic": (dict(batch=1, n_slots=4096, n_writes=64, n_reads=64, d_v=64, seq_len=96), 32),
}


@requires_cuda
@requires_triton
@needs_cuda
@needs_triton
@pytest.mark.parametrize("case", list(CASES), ids=list(CASES))
def test_as_close_to_the_oracle_as_the_reference_is(case: str) -> None:
    """Both routes scored against one fp64 oracle, outputs and every gradient.

    A bare threshold would still pass if the kernels were ten times further out
    and the threshold happened to be loose; the comparison with the reference's
    own distance is the claim.
    """
    overrides, chunk = CASES[case]
    inputs = draw(**overrides)
    want = oracle(inputs, chunk)
    got = run(tk.chunk_sparse_delta, place(inputs, "cuda", torch.float32), chunk)
    trusted = run(ref.chunk_sparse_delta, place(inputs, "cuda", torch.float32), chunk)
    for name in want:
        error = distance(got[name], want[name])
        baseline = distance(trusted[name], want[name])
        assert error < FP32_ROUNDOFF, f"{name}: {error:.2e} from the oracle"
        assert error < max(10 * baseline, 1e-7), f"{name}: {error:.2e} vs reference {baseline:.2e}"


@requires_cuda
@requires_triton
@needs_cuda
@needs_triton
def test_the_fixture_collides() -> None:
    """The equivalence tests cannot pass by never meeting a collision: the
    default draw writes some slot twice in a chunk and reads a slot written
    earlier in the same chunk."""
    inputs = draw()
    w = inputs["write_idx"][..., :8, :].reshape(-1, 8 * 4)
    assert any(len(set(row.tolist())) < row.numel() for row in w)
    r = inputs["read_idx"][..., :8, :]
    written = inputs["write_idx"][..., :8, :]
    assert bool((r.unsqueeze(-1).unsqueeze(-1) == written.unsqueeze(-3).unsqueeze(-3)).any())


@requires_cuda
@requires_triton
@needs_cuda
@needs_triton
def test_without_gradients_the_forward_is_the_same_forward() -> None:
    """The no-grad walk keeps nothing for a backward and gathers again where the
    differentiable one reads its copy -- the same values, so the same bits."""
    inputs = place(draw(n_slots=4096, n_writes=16, n_reads=16, d_v=48, seq_len=80), "cuda", torch.float32)
    args = [inputs[a] for a in ARGS]
    with_grad = tk.chunk_sparse_delta(*args, chunk_size=16)
    with torch.no_grad():
        without = tk.chunk_sparse_delta(*args, chunk_size=16)
    for a, b in zip(with_grad, without):
        assert torch.equal(a.detach(), b)


@requires_cuda
@requires_triton
@needs_cuda
@needs_triton
def test_an_incoming_table_is_carried() -> None:
    """A stream resumed from the table it handed back matches one pass over it."""
    inputs = place(draw(seq_len=64, shared_memory=False), "cuda", torch.float32)
    args = {name: x.detach() for name, x in inputs.items()}
    whole, whole_final = tk.chunk_sparse_delta(*(args[a] for a in ARGS), chunk_size=8)
    split = 27  # mid-chunk on purpose: the second call starts a fresh chunking
    head = {n: (x if n == "memory" else x[:, :, :split]) for n, x in args.items()}
    first, carried = tk.chunk_sparse_delta(*(head[a] for a in ARGS), chunk_size=8)
    tail = {n: (carried if n == "memory" else x[:, :, split:]) for n, x in args.items()}
    second, final = tk.chunk_sparse_delta(*(tail[a] for a in ARGS), chunk_size=8)
    assert distance(torch.cat([first, second], dim=2), whole) < FP32_ROUNDOFF
    assert distance(final, whole_final) < FP32_ROUNDOFF


@requires_cuda
@requires_triton
@needs_cuda
@needs_triton
def test_the_input_table_is_never_written() -> None:
    """The walk writes its own copy: the caller's table -- a state, perhaps a
    broadcast view of a learned one -- is untouched."""
    inputs = place(draw(), "cuda", torch.float32)
    before = inputs["memory"].detach().clone()
    run(tk.chunk_sparse_delta, inputs, 8)
    assert torch.equal(inputs["memory"].detach(), before)


@requires_cuda
@requires_triton
@needs_cuda
@needs_triton
def test_recompute_pairwise_changes_nothing() -> None:
    """The Triton terms keep no pairwise arrays, so the dial has nothing to do."""
    inputs = draw()
    plain = run(tk.chunk_sparse_delta, place(inputs, "cuda", torch.float32), 8)
    dialled = run(tk.chunk_sparse_delta, place(inputs, "cuda", torch.float32), 8, recompute_pairwise=True)
    for name in plain:
        assert distance(dialled[name], plain[name]) < FP32_ROUNDOFF, name


@requires_cuda
@requires_triton
@needs_cuda
@needs_triton
def test_the_layer_agrees_with_itself_across_backends() -> None:
    """End to end, one set of weights, both backends -- the consumer's question.

    The kernels in isolation are tested above; this is the layer wiring them:
    the dispatch, the learned table's broadcast, the dial passed through.
    """
    config = dict(
        d_model=64, n_heads=2, n_slots=16**2, initial_memory="learned",
        n_writes=8, n_reads=8, chunk_size=16,
    )
    torch.manual_seed(0)
    reference = SparseDeltaMemory(SparseDeltaMemoryConfig(**config)).cuda()
    accelerated = SparseDeltaMemory(SparseDeltaMemoryConfig(**config, backend="triton")).cuda()
    accelerated.load_state_dict(reference.state_dict())
    with torch.no_grad():  # a learned table that is not zero, so it is exercised
        reference.initial_memory.normal_()
        accelerated.initial_memory.copy_(reference.initial_memory)
    x = torch.randn(2, 100, 64, device="cuda")
    y_ref, y_tri = reference(x), accelerated(x)
    assert distance(y_tri, y_ref) < FP32_ROUNDOFF
    y_ref.square().sum().backward()
    y_tri.square().sum().backward()
    for (name, p_ref), p_tri in zip(reference.named_parameters(), accelerated.parameters()):
        assert distance(p_tri.grad, p_ref.grad) < FP32_ROUNDOFF, name
