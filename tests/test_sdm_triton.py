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
    decay: str = "write_set",
    write_slots: tuple[int, int] | None = None,
    log_alpha: torch.Tensor | None = None,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Kernel inputs obeying the layer's invariants, in fp64 on the host.

    Distinct writes per position, softmax weights, `β ∈ (0, 2)`, log-decays
    from near a full wipe to almost none.  **The table is small on purpose**, as
    in ``test_sdm.py``: 24 slots against 4 writes per position collide in every
    chunk.  ``crowd`` > 0 confines writes AND reads to that many slots, which
    also puts the same slot in one position's reads twice.  ``shared_memory``
    gives one `(H, N, d_v)` table broadcast over the batch, as a learned
    initial table is.  ``decay="key"`` weights each write's log-decay by its
    write weight; ``write_slots=(lo, hi)`` confines writes to that range.
    """
    g = torch.Generator().manual_seed(seed)
    lead = (batch, heads)
    pool = crowd if crowd else n_slots
    write_idx = torch.rand(*lead, seq_len, pool, generator=g).argsort(-1)[..., :n_writes]
    if write_slots is not None:
        lo, hi = write_slots
        write_idx = lo + torch.rand(*lead, seq_len, hi - lo, generator=g).argsort(-1)[..., :n_writes]
    if crowd:
        read_idx = torch.randint(0, pool, (*lead, seq_len, n_reads), generator=g)
    else:
        read_idx = torch.rand(*lead, seq_len, n_slots, generator=g).argsort(-1)[..., :n_reads]
    randn = lambda *shape: torch.randn(*shape, generator=g, dtype=torch.float64)  # noqa: E731
    if log_alpha is None:
        log_alpha = -F.softplus(2.0 * randn(*lead, seq_len))
    write_val = torch.softmax(randn(*lead, seq_len, n_writes), -1)
    if decay == "key":
        log_decay = log_alpha.unsqueeze(-1) * write_val
    else:
        log_decay = log_alpha.unsqueeze(-1).expand(*lead, seq_len, n_writes)
    memory_lead = (heads,) if shared_memory else lead
    return {
        "memory": randn(*memory_lead, n_slots, d_v),
        "write_idx": write_idx,
        "write_val": write_val,
        "log_decay": log_decay.contiguous(),
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


@requires_cuda
@needs_cuda
def test_a_chunk_past_the_limit_runs_the_reference_exactly() -> None:
    """A chunk too long for the kernels' tiles is the reference's, bit for bit
    in the forward (the backward's own scatter-adds reorder on CUDA)."""
    inputs = draw(seq_len=150)
    chunk = tk.MAX_CHUNK * 2
    args = [x.detach() for x in place(inputs, "cuda", torch.float32).values()]
    got = tk.chunk_sparse_delta(*args, chunk_size=chunk)
    want = ref.chunk_sparse_delta(*args, chunk_size=chunk)
    for a, b in zip(got, want):
        assert torch.equal(a, b)


def test_vmap_on_the_host_runs_the_reference() -> None:
    """Under a transform on a CPU tensor the reference's functional holder
    runs, exactly as it does for the reference itself."""
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
    "shorter-than-a-chunk": (dict(seq_len=5), 8),
    "chunk-not-a-power-of-two": (dict(seq_len=50), 6),
    "one-write-one-read": (dict(n_writes=1, n_reads=1), 8),
    "crowded": (dict(crowd=6, n_reads=4, seq_len=40), 8),
    "wide-values": (dict(d_v=40, n_slots=64, n_writes=8, n_reads=8, seq_len=64), 16),
    "per-stream-table": (dict(shared_memory=False), 8),
    "key-weighted-decay": (dict(decay="key"), 8),
    "realistic": (dict(batch=1, n_slots=4096, n_writes=64, n_reads=64, d_v=64, seq_len=96), 32),
    # Past 32 positions the walk must take tl.dot; past 64 entries a chunk
    # gathers in more than one block.
    "long-chunk": (dict(n_slots=1024, n_writes=8, n_reads=8, d_v=24, seq_len=200), 64),
    "many-entries": (dict(batch=1, heads=1, n_slots=4096, n_writes=128, n_reads=96, d_v=16, seq_len=40), 16),
    "narrow-values": (dict(d_v=3), 8),
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


@requires_cuda
@requires_triton
@needs_cuda
@needs_triton
def test_every_slot_selected_is_gated_deltanet() -> None:
    """`N = d_k`, `W = R = N`, dense unit keys: Gated DeltaNet, exactly the
    reduction ``test_sdm.py`` holds the reference to.  For these kernels it is
    also the worst case of the segment walk: every slot is written at every
    position, so every segment holds the whole chunk."""
    from lumen.gdn.reference import sequential_gated_delta

    batch, seq_len, d_k, d_v = 2, 40, 8, 5
    g = torch.Generator().manual_seed(3)
    randn = lambda *shape: torch.randn(*shape, generator=g, dtype=torch.float64)  # noqa: E731
    q = F.normalize(randn(batch, 1, 1, seq_len, d_k), dim=-1)
    k = F.normalize(randn(batch, 1, 1, seq_len, d_k), dim=-1)
    v = randn(batch, 1, 1, seq_len, d_v)
    beta = 2.0 * torch.sigmoid(randn(batch, 1, 1, seq_len))
    log_alpha = -F.softplus(randn(batch, 1, 1, seq_len))
    y_gdn, m_gdn = sequential_gated_delta(q, k, v, beta, log_alpha)

    every_slot = torch.arange(d_k).expand(batch, seq_len, d_k)
    sdm = [
        torch.zeros(batch, d_k, d_v, dtype=torch.float64),
        every_slot,
        k[:, 0, 0],
        log_alpha[:, 0, 0].unsqueeze(-1).expand(batch, seq_len, d_k).contiguous(),
        v[:, 0, 0],
        beta[:, 0, 0],
        every_slot,
        q[:, 0, 0],
    ]
    sdm = [x.cuda().float() if x.is_floating_point() else x.cuda() for x in sdm]
    y, m = tk.chunk_sparse_delta(*sdm, chunk_size=8)
    assert distance(y, y_gdn[:, 0, 0].cuda()) < FP32_ROUNDOFF
    assert distance(m, m_gdn[:, 0, 0].cuda()) < FP32_ROUNDOFF


@requires_cuda
@requires_triton
@needs_cuda
@needs_triton
def test_unwritten_slots_are_frozen_bit_for_bit() -> None:
    """A slot no position writes comes out as it went in, exactly -- including
    the slots a ragged last chunk's padding names, which it writes back as
    `exp(0)·row + 0·δ`.  The walk stores only first writers, so nothing else
    touches them."""
    n_slots = 24
    inputs = place(draw(seq_len=45, write_slots=(n_slots // 2, n_slots)), "cuda", torch.float32)
    _, final = tk.chunk_sparse_delta(*(inputs[a].detach() for a in ARGS), chunk_size=8)
    entering = inputs["memory"].detach().expand_as(final)
    untouched = slice(0, n_slots // 2)
    assert torch.equal(final[..., untouched, :], entering[..., untouched, :])
    assert not torch.equal(final[..., n_slots // 2 :, :], entering[..., n_slots // 2 :, :])


@requires_cuda
@requires_triton
@needs_cuda
@needs_triton
def test_no_overflow_at_extreme_decay() -> None:
    """Every write a full wipe: finite, and each slot holds exactly its last
    write.  Every exponent the kernels form is `<= 0` -- the carry coefficient's
    `G_end − G` included -- so the worst that happens is underflow to zero,
    which is the right answer for a wiped slot."""
    batch, heads, seq_len = 2, 2, 40
    log_alpha = torch.full((batch, heads, seq_len), -1e4, dtype=torch.float64)
    inputs = draw(log_alpha=log_alpha, seq_len=seq_len, shared_memory=False)
    got = run(tk.chunk_sparse_delta, place(inputs, "cuda", torch.float32), 16)
    assert all(torch.isfinite(x).all() for x in got.values())
    want = oracle(inputs, 16)
    assert distance(got["out"], want["out"]) < FP32_ROUNDOFF
    assert distance(got["final"], want["final"]) < FP32_ROUNDOFF
    expected = inputs["memory"].clone()
    fresh = inputs["write_val"].unsqueeze(-1) * (inputs["beta"].unsqueeze(-1) * inputs["v"]).unsqueeze(-2)
    for t in range(seq_len):  # in order, so the last writer wins
        index = inputs["write_idx"][:, :, t].unsqueeze(-1).expand_as(fresh[:, :, t])
        expected = expected.scatter(-2, index, fresh[:, :, t])
    assert distance(got["final"], expected.cuda()) < FP32_ROUNDOFF


@requires_cuda
@requires_triton
@needs_cuda
@needs_triton
def test_the_backward_is_deterministic_without_asking() -> None:
    """No atomics anywhere: every sum has one owner and a fixed order, so two
    runs agree bit for bit -- outputs and every gradient -- with deterministic
    algorithms off.  The reference cannot claim this on CUDA."""
    assert not torch.are_deterministic_algorithms_enabled()
    inputs = draw(crowd=6, n_reads=4, seq_len=64)
    first = run(tk.chunk_sparse_delta, place(inputs, "cuda", torch.float32), 8)
    second = run(tk.chunk_sparse_delta, place(inputs, "cuda", torch.float32), 8)
    for name in first:
        assert torch.equal(first[name], second[name]), name


@requires_cuda
@requires_triton
@needs_cuda
@needs_triton
def test_vmap_without_gradients_runs_the_kernels() -> None:
    """Many parameter sets in one call: the mapped axis becomes more streams.

    A consumer that evaluates parameter sets under ``vmap`` with no gradient
    gets the kernels, not the reference -- checked by the result's agreement
    with each set run alone, and by the kernels being called at all.  Distinct
    parameter sets and a learned table, so a rule that broadcast one set, or
    lost the table's own mapped axis, could not pass.
    """
    config = SparseDeltaMemoryConfig(
        d_model=32, n_heads=2, n_slots=16**2, initial_memory="learned",
        n_writes=8, n_reads=8, chunk_size=16, backend="triton",
    )
    models = []
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        model = SparseDeltaMemory(config).cuda().eval()
        with torch.no_grad():
            model.initial_memory.normal_()
        models.append(model)
    params, buffers = torch.func.stack_module_state(models)
    base = SparseDeltaMemory(config).cuda().eval()
    x = torch.randn(2, 50, 32, device="cuda")

    def call(p: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.func.functional_call(base, (p, b), (x,))

    calls = []
    original = tk._drive

    def counted(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        calls.append(1)
        return original(*args, **kwargs)

    tk._drive = counted
    try:
        with torch.no_grad():
            batched = torch.func.vmap(call)(params, buffers)
    finally:
        tk._drive = original
    assert calls, "the kernels never ran under vmap"
    with torch.no_grad():
        for model, y in zip(models, batched):
            assert distance(y, model(x)) < FP32_ROUNDOFF


@requires_cuda
@requires_triton
@needs_cuda
@needs_triton
def test_vmap_with_gradients_runs_the_reference() -> None:
    """With gradients under a transform the kernels step aside: the
    reference's functional holder is what every transform can differentiate."""
    inputs = place(draw(batch=3, heads=1, shared_memory=False), "cuda", torch.float32)
    args = [inputs[a].detach() for a in ARGS]

    def loss(memory: torch.Tensor, *rest: torch.Tensor) -> torch.Tensor:
        out, final = tk.chunk_sparse_delta(memory, *rest, chunk_size=8, check_writes=False)
        return out.square().sum() + final.square().sum()

    def loss_ref(memory: torch.Tensor, *rest: torch.Tensor) -> torch.Tensor:
        out, final = ref.chunk_sparse_delta(memory, *rest, chunk_size=8, check_writes=False)
        return out.square().sum() + final.square().sum()

    got = torch.func.grad(loss)(*args)
    want = torch.func.grad(loss_ref)(*args)
    assert distance(got, want) < FP32_ROUNDOFF


@requires_cuda
@requires_triton
@needs_cuda
@needs_triton
def test_a_table_that_wants_no_gradient() -> None:
    """A zero initial table is a fresh tensor that wants no gradient while the
    projections do: the token chain starts without one, and every other input
    still gets its gradient, equal to the reference's."""
    inputs = draw()
    def leaves(device_inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        device_inputs["memory"] = torch.zeros_like(device_inputs["memory"]).detach()
        return device_inputs

    got_inputs = leaves(place(inputs, "cuda", torch.float32))
    want_inputs = leaves(place(inputs, "cuda", torch.float32))
    differentiable = [name for name in DIFFERENTIABLE if name != "memory"]
    results = []
    for kernel, args in ((tk.chunk_sparse_delta, got_inputs), (ref.chunk_sparse_delta, want_inputs)):
        out, final = kernel(*(args[a] for a in ARGS), chunk_size=8)
        grads = torch.autograd.grad(out.square().sum() + final.square().sum(), [args[n] for n in differentiable])
        results.append((out.detach(), final.detach(), *grads))
    for name, a, b in zip(["out", "final", *differentiable], *results):
        assert distance(a, b) < FP32_ROUNDOFF, name


@requires_cuda
@requires_triton
@needs_cuda
@needs_triton
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")
def test_a_batch_on_another_device_than_the_current_one() -> None:
    """Launches go to the current device and streams are per device: a batch
    on another card must be launched, and synchronised, on its own card."""
    inputs = draw(n_slots=4096, n_writes=16, n_reads=16, d_v=32, seq_len=200)
    previous = torch.cuda.current_device()
    torch.cuda.set_device(0)
    try:
        got = run(tk.chunk_sparse_delta, place(inputs, "cuda:1", torch.float32), 8)
        want = run(ref.chunk_sparse_delta, place(inputs, "cuda:1", torch.float32), 8)
    finally:
        torch.cuda.set_device(previous)
    for name in got:
        assert got[name].device == torch.device("cuda:1")
        assert distance(got[name], want[name]) < FP32_ROUNDOFF, name


@requires_cuda
@requires_triton
@needs_cuda
@needs_triton
def test_grouping_is_inert_bit_for_bit() -> None:
    """How many chunks share a launch changes how the work is batched and
    nothing it computes: each chunk's terms are their own program, and the
    walk takes the chunks in the same order either way -- so the same bits,
    outputs and every gradient, at one chunk per group, three, and all."""
    inputs = draw(seq_len=96)
    results = [
        run(tk.chunk_sparse_delta, place(inputs, "cuda", torch.float32), 8, group=group)
        for group in (1, 3, 12)
    ]
    for other in results[1:]:
        for name in results[0]:
            assert torch.equal(other[name], results[0][name]), name
