"""Sparse Delta Memory — the kernel acceptance gates of the design record, as tests.

Organised the way the record's §7 is: the chunkwise path against its oracle,
the reduction to Gated DeltaNet, the properties claimed "by construction", and
the preconditions the kernels refuse to guess about.

Tolerance convention, the same as Gated DeltaNet's (``test_gdn.py``): the
chunkwise and sequential paths are one algorithm reached by two routes, so in
fp64 they agree to round-off and the gate is tight.  fp32 gets a bound
consistent with fp32 round-off, never the fp64 one.
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn.functional as F

from lumen.gdn.reference import chunk_gated_delta, sequential_gated_delta
from lumen.sdm.reference import (
    _partners,
    _partners_dense,
    chunk_sparse_delta,
    read_sparse_delta,
    recurrent_sparse_delta,
    sequential_sparse_delta,
)

EXACT = 1e-9  # fp64: structural agreement
FP32 = 1e-5  # fp32: round-off on these shapes, deliberately not tighter

KERNEL_ARGS = (
    "memory",
    "write_idx",
    "write_val",
    "log_decay",
    "v",
    "beta",
    "read_idx",
    "read_val",
)
DIFFERENTIABLE = ("memory", "write_val", "log_decay", "v", "beta", "read_val")

# The two ways the chunkwise kernel can hold its table between chunks.  Every
# gate the chunkwise path answers to, it answers to under both: the holders are
# separate code, and "the other one passes" is not evidence for either.
HOLDERS = ("in_place", "functional")


def chunked(*kernel_args, holder: str, **kwargs):
    return chunk_sparse_delta(*kernel_args, in_place=holder == "in_place", **kwargs)


def make_inputs(
    *,
    batch: int = 2,
    heads: int = 2,
    seq_len: int = 48,
    n_slots: int = 24,
    n_writes: int = 4,
    n_reads: int = 3,
    d_v: int = 5,
    dtype: torch.dtype = torch.float64,
    decay: str = "write_set",
    write_slots: tuple[int, int] | None = None,
    log_alpha: torch.Tensor | None = None,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Kernel-shaped inputs obeying the layer's own invariants.

    Write indices are distinct within each position, as product keys make
    them; weights are softmaxes, so `k ≥ 0` and `Σk = 1`.  **The table is
    small on purpose**: 24 slots against 4 writes per position means every
    chunk longer than a few positions writes some slots several times and reads
    slots written earlier in the same chunk -- the cases the partner layout
    exists for.  ``test_the_fixture_exercises_shared_slots`` holds that, so the
    equivalence tests cannot pass by never meeting a collision.

    ``write_slots=(lo, hi)`` confines writes to that range of slots, for the
    freeze tests.  ``decay="key"`` weights each write's log-decay by its write
    weight, the arrangement the kernel is agnostic to.
    """
    generator = torch.Generator().manual_seed(seed)
    lead = (batch, heads)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, dtype=dtype)

    lo, hi = write_slots if write_slots is not None else (0, n_slots)
    write_idx = lo + torch.rand(*lead, seq_len, hi - lo, generator=generator).argsort(-1)[
        ..., :n_writes
    ]
    read_idx = torch.rand(*lead, seq_len, n_slots, generator=generator).argsort(-1)[
        ..., :n_reads
    ]
    write_val = torch.softmax(randn(*lead, seq_len, n_writes), -1)
    read_val = torch.softmax(randn(*lead, seq_len, n_reads), -1)

    # Unclamped, and wide: `-softplus(2·randn)` puts some writes near a full
    # wipe and some near no decay at all, which is the spread the paper's
    # trained gate reaches.
    if log_alpha is None:
        log_alpha = -F.softplus(2.0 * randn(*lead, seq_len))
    if decay == "write_set":
        log_decay = log_alpha.unsqueeze(-1).expand(*lead, seq_len, n_writes)
    elif decay == "key":
        log_decay = log_alpha.unsqueeze(-1) * write_val
    else:
        raise ValueError(decay)

    return {
        "memory": randn(*lead, n_slots, d_v),
        "write_idx": write_idx,
        "write_val": write_val,
        "log_decay": log_decay.contiguous(),
        "v": randn(*lead, seq_len, d_v),
        "beta": 2.0 * torch.sigmoid(randn(*lead, seq_len)),
        "read_idx": read_idx,
        "read_val": read_val,
    }


def args(inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    return tuple(inputs[name] for name in KERNEL_ARGS)


def max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


# ── the fixture itself ────────────────────────────────────────────────────


def test_the_fixture_exercises_shared_slots():
    inputs = make_inputs()
    chunk = 8
    write_idx = inputs["write_idx"]
    blocks = write_idx.reshape(*write_idx.shape[:2], -1, chunk * write_idx.shape[-1])
    distinct = torch.tensor(
        [len(torch.unique(block)) for block in blocks.reshape(-1, blocks.shape[-1])]
    )
    # Every chunk writes fewer distinct slots than it has write entries: some
    # slot is written by several positions in the same chunk, everywhere.
    assert bool((distinct < blocks.shape[-1]).all())


# ── partners ──────────────────────────────────────────────────────────────


def chunk_indices(
    layout: str, *, rows: int = 3, chunk: int = 8, n_writes: int = 4, n_reads: int = 3,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One chunk's `(P, C, W)` write and `(P, C, R)` read indices, flat as the
    kernel sees them -- each row's slots offset by `p·N`.

    ``collide``   a table one slot wider than a write set: nearly every slot is
                  written at nearly every position.
    ``same``      every position writes the same `W` slots, in its own order --
                  every entry has a partner at every position.
    ``disjoint``  no slot is written twice in the chunk.
    ``miss``      reads confined to slots no position writes.
    ``fixture``   the kernel fixture's proportions.
    """
    generator = torch.Generator().manual_seed(seed)

    def distinct(n_slots: int, k: int, lo: int = 0) -> torch.Tensor:
        return lo + torch.rand(rows, chunk, n_slots, generator=generator).argsort(-1)[..., :k]

    if layout == "collide":
        n_slots = n_writes + 1
        w_idx, r_idx = distinct(n_slots, n_writes), distinct(n_slots, n_reads)
    elif layout == "same":
        n_slots = n_writes + n_reads
        w_idx, r_idx = distinct(n_writes, n_writes), distinct(n_slots, n_reads)
    elif layout == "disjoint":
        n_slots = chunk * n_writes
        w_idx = (torch.arange(chunk).view(1, chunk, 1) * n_writes
                 + torch.arange(n_writes)).expand(rows, chunk, n_writes)
        r_idx = distinct(n_slots, n_reads)
    elif layout == "miss":
        n_slots = 2 * n_writes + n_reads
        w_idx, r_idx = distinct(n_writes + 1, n_writes), distinct(n_reads, n_reads, lo=n_slots - n_reads)
    elif layout == "fixture":
        n_slots = 24
        w_idx, r_idx = distinct(n_slots, n_writes), distinct(n_slots, n_reads)
    else:
        raise ValueError(layout)
    offset = (torch.arange(rows) * n_slots).view(rows, 1, 1)
    return w_idx + offset, r_idx + offset


@pytest.mark.parametrize("chunk", [1, 2, 5, 8, 64])
@pytest.mark.parametrize("layout", ["collide", "same", "disjoint", "miss", "fixture"])
def test_sorted_partners_are_the_dense_compare_exactly(layout, chunk):
    """One sort and a binary search per cell, against the compare it replaced.

    Exactly -- ``has`` and ``partner`` alike, including the value left where
    there is no partner -- so everything downstream is the same arithmetic on
    the same numbers, not merely close.
    """
    w_idx, r_idx = chunk_indices(layout, chunk=chunk)
    for got, spec in zip(_partners(w_idx, r_idx), _partners_dense(w_idx, r_idx)):
        assert got.dtype == spec.dtype
        assert torch.equal(got, spec)


def test_the_partner_layouts_mean_what_they_say():
    """The adversarial layouts reach the extremes they are named for."""
    w_has, _, r_has, _ = _partners(*chunk_indices("same"))
    assert bool(w_has.all())
    w_has, _, _, _ = _partners(*chunk_indices("disjoint"))
    eye = torch.eye(8, dtype=torch.bool).view(1, 8, 1, 8)
    assert torch.equal(w_has, eye.expand_as(w_has))  # only itself
    _, _, r_has, _ = _partners(*chunk_indices("miss"))
    assert not bool(r_has.any())


# ── chunkwise against the oracle ──────────────────────────────────────────


@pytest.mark.parametrize("holder", HOLDERS)
@pytest.mark.parametrize("decay", ["write_set", "key"])
@pytest.mark.parametrize("chunk", [1, 2, 5, 8, 16, 64])
def test_chunkwise_matches_sequential_fp64(decay, chunk, holder):
    inputs = make_inputs(decay=decay)
    y_ref, m_ref = sequential_sparse_delta(*args(inputs))
    y, m = chunked(*args(inputs), chunk_size=chunk, holder=holder)
    assert max_diff(y, y_ref) < EXACT
    assert max_diff(m, m_ref) < EXACT


@pytest.mark.parametrize("holder", HOLDERS)
def test_chunkwise_matches_sequential_fp32(holder):
    inputs = make_inputs(dtype=torch.float32)
    y_ref, m_ref = sequential_sparse_delta(*args(inputs))
    y, m = chunked(*args(inputs), chunk_size=8, holder=holder)
    assert max_diff(y, y_ref) < FP32
    assert max_diff(m, m_ref) < FP32


@pytest.mark.parametrize("holder", HOLDERS)
@pytest.mark.parametrize("chunk", [2, 3, 8, 16, 64])
def test_chunk_size_is_inert(chunk, holder):
    inputs = make_inputs()
    y_one, m_one = chunked(*args(inputs), chunk_size=1, holder=holder)
    y, m = chunked(*args(inputs), chunk_size=chunk, holder=holder)
    assert max_diff(y, y_one) < EXACT
    assert max_diff(m, m_one) < EXACT


@pytest.mark.parametrize("holder", HOLDERS)
@pytest.mark.parametrize("group", [1, 2, 4, 5])
def test_grouping_is_inert(group, holder):
    """How many chunks have their table-free terms computed together changes
    nothing: one at a time, groups with a short last one, or -- the default at
    this size -- all six at once."""
    inputs = make_inputs()
    y_all, m_all = chunked(*args(inputs), chunk_size=8, holder=holder)
    y, m = chunked(*args(inputs), chunk_size=8, group=group, holder=holder)
    assert max_diff(y, y_all) < EXACT
    assert max_diff(m, m_all) < EXACT


@pytest.mark.parametrize("holder", HOLDERS)
@pytest.mark.parametrize("seq_len", [1, 2, 7, 8, 9, 23])
def test_any_sequence_length_works_and_is_exact(seq_len, holder):
    inputs = make_inputs(seq_len=seq_len)
    y_ref, m_ref = sequential_sparse_delta(*args(inputs))
    y, m = chunked(*args(inputs), chunk_size=8, holder=holder)
    assert y.shape == y_ref.shape
    assert max_diff(y, y_ref) < EXACT
    assert max_diff(m, m_ref) < EXACT


@pytest.mark.parametrize("holder", HOLDERS)
@pytest.mark.parametrize("split", [5, 16, 29])
def test_split_and_resume_equals_one_pass(split, holder):
    inputs = make_inputs()
    y_one, m_one = chunked(*args(inputs), chunk_size=8, holder=holder)

    def part(lo: int, hi: int | None) -> list[torch.Tensor]:
        return [
            inputs[name] if name == "memory" else inputs[name][:, :, lo:hi]
            for name in KERNEL_ARGS
        ]

    first = part(0, split)
    y_a, carried = chunked(*first, chunk_size=8, holder=holder)
    second = part(split, None)
    second[0] = carried
    y_b, m = chunked(*second, chunk_size=8, holder=holder)

    assert max_diff(torch.cat([y_a, y_b], dim=2), y_one) < EXACT
    assert max_diff(m, m_one) < EXACT


def test_step_is_the_oracle_one_position_at_a_time():
    inputs = make_inputs(seq_len=6)
    y_ref, m_ref = sequential_sparse_delta(*args(inputs))
    memory = inputs["memory"]
    for t in range(6):
        y, memory = recurrent_sparse_delta(
            memory, *(inputs[name][:, :, t] for name in KERNEL_ARGS[1:])
        )
        assert torch.equal(y, y_ref[:, :, t])
    assert torch.equal(memory, m_ref)


def test_read_does_not_write():
    inputs = make_inputs(seq_len=1)
    memory = inputs["memory"].clone()
    read_sparse_delta(memory, inputs["read_idx"][:, :, 0], inputs["read_val"][:, :, 0])
    assert torch.equal(memory, inputs["memory"])


# ── the reduction to Gated DeltaNet ───────────────────────────────────────


@pytest.mark.parametrize("path", ["sequential", *HOLDERS])
def test_every_slot_selected_is_gated_deltanet(path):
    """`N = d_k`, `W = R = N`, dense unit keys: Lumen's own GDN oracle.

    The paper claims this reduction.  Here it is a gate: the decay is uniform
    over a write set that is every slot, so `Λ_t = α_t I`, and the recurrence
    is Gated DeltaNet exactly -- against ``sequential_gated_delta``, not
    against a re-derivation of it.
    """
    batch, seq_len, d_k, d_v = 2, 40, 8, 5
    generator = torch.Generator().manual_seed(3)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, dtype=torch.float64)

    q = F.normalize(randn(batch, 1, 1, seq_len, d_k), dim=-1)
    k = F.normalize(randn(batch, 1, 1, seq_len, d_k), dim=-1)
    v = randn(batch, 1, 1, seq_len, d_v)
    beta = 2.0 * torch.sigmoid(randn(batch, 1, 1, seq_len))
    log_alpha = -F.softplus(randn(batch, 1, 1, seq_len))
    y_gdn, m_gdn = sequential_gated_delta(q, k, v, beta, log_alpha)

    every_slot = torch.arange(d_k).expand(batch, seq_len, d_k)
    sdm = (
        torch.zeros(batch, d_k, d_v, dtype=torch.float64),
        every_slot,
        k[:, 0, 0],
        log_alpha[:, 0, 0].unsqueeze(-1).expand(batch, seq_len, d_k),
        v[:, 0, 0],
        beta[:, 0, 0],
        every_slot,
        q[:, 0, 0],
    )
    if path == "sequential":
        y, m = sequential_sparse_delta(*sdm)
    else:
        y, m = chunked(*sdm, chunk_size=8, holder=path)

    assert max_diff(y, y_gdn[:, 0, 0]) < EXACT
    assert max_diff(m, m_gdn[:, 0, 0]) < EXACT
    # And the chunkwise GDN agrees with both, closing the triangle.
    y_chunk, _ = chunk_gated_delta(q, k, v, beta, log_alpha, 8)
    assert max_diff(y, y_chunk[:, 0, 0]) < EXACT


# ── properties claimed by construction ────────────────────────────────────


@pytest.mark.parametrize("path", ["sequential", *HOLDERS])
def test_unwritten_slots_are_frozen_bit_for_bit(path):
    """A slot no position writes is the incoming slot, exactly — decay included.

    Writes are confined to the upper half of the table and the lower half is
    checked with ``torch.equal``.  The sequence length leaves a ragged last
    chunk, whose padding writes slots `0..W-1` with weight 0 -- in the half
    being checked -- so this also holds the claim that padding writes back
    exactly the row it found.
    """
    n_slots = 24
    inputs = make_inputs(seq_len=45, write_slots=(n_slots // 2, n_slots))
    if path == "sequential":
        _, m = sequential_sparse_delta(*args(inputs))
    else:
        _, m = chunked(*args(inputs), chunk_size=8, holder=path)
    untouched = slice(0, n_slots // 2)
    assert torch.equal(m[..., untouched, :], inputs["memory"][..., untouched, :])
    # ...and the written half really was written, so the check means something.
    assert not torch.equal(m[..., n_slots // 2 :, :], inputs["memory"][..., n_slots // 2 :, :])


@pytest.mark.parametrize("holder", HOLDERS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_no_overflow_at_extreme_decay(dtype, holder):
    """Every write a full wipe: finite, and each slot holds exactly its last write.

    `log α = −1e4` underflows `α` to exactly 0 in both formats.  A factored
    `e^{±G}` form would be `inf` by the second write to a slot; the pairwise
    form never builds a factor above 1, and the only thing that can happen is
    underflow to zero -- the right answer for a wiped slot.

    With every write a wipe, the delta rule reads nothing (`retrieved = 0`), so
    a slot's final row is `k · β · v` of the last position that wrote it.
    """
    batch, heads, seq_len = 2, 2, 40
    log_alpha = torch.full((batch, heads, seq_len), -1e4, dtype=dtype)
    inputs = make_inputs(dtype=dtype, log_alpha=log_alpha, seq_len=seq_len)
    y_ref, m_ref = sequential_sparse_delta(*args(inputs))
    y, m = chunked(*args(inputs), chunk_size=16, holder=holder)
    assert torch.isfinite(y).all() and torch.isfinite(m).all()

    bound = EXACT if dtype == torch.float64 else FP32
    assert max_diff(y, y_ref) < bound
    assert max_diff(m, m_ref) < bound

    expected = inputs["memory"].clone()
    write_idx = inputs["write_idx"]
    fresh = inputs["write_val"].unsqueeze(-1) * (
        inputs["beta"].unsqueeze(-1) * inputs["v"]
    ).unsqueeze(-2)
    for t in range(seq_len):  # in order, so the last writer wins
        index = write_idx[:, :, t].unsqueeze(-1).expand_as(fresh[:, :, t])
        expected = expected.scatter(-2, index, fresh[:, :, t])
    assert max_diff(m_ref, expected) < bound
    assert max_diff(m, expected) < bound


@pytest.mark.parametrize("holder", HOLDERS)
def test_a_shared_initial_table_serves_the_batch(holder):
    """`memory` with fewer leading axes broadcasts, and its gradient is the sum.

    A learned initial table is `(H, N, d_v)` and serves every stream in the
    batch.  It must behave as if it had been copied per stream -- outputs and
    final tables identical -- and its gradient must be the sum of the per-stream
    gradients, since every stream reads it.
    """
    inputs = make_inputs()
    shared = inputs["memory"][0].clone().requires_grad_(True)
    copied = shared.detach().expand_as(inputs["memory"]).clone().requires_grad_(True)

    rest = args(inputs)[1:]
    y_shared, m_shared = chunked(shared, *rest, chunk_size=8, holder=holder)
    y_copied, m_copied = chunked(copied, *rest, chunk_size=8, holder=holder)
    assert torch.equal(y_shared, y_copied)
    assert torch.equal(m_shared, m_copied)

    (y_shared.sum() + m_shared.sum()).backward()
    (y_copied.sum() + m_copied.sum()).backward()
    assert max_diff(shared.grad, copied.grad.sum(0)) < EXACT


# ── gradients ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("holder", HOLDERS)
@pytest.mark.parametrize("decay", ["write_set", "key"])
def test_gradients_match_the_oracle(decay, holder):
    """Every differentiable input, through outputs AND the final table."""
    inputs = make_inputs(decay=decay, seq_len=29)
    generator = torch.Generator().manual_seed(11)
    probe_y = torch.randn(2, 2, 29, 5, generator=generator, dtype=torch.float64)
    probe_m = torch.randn(2, 2, 24, 5, generator=generator, dtype=torch.float64)

    def grads(fn) -> dict[str, torch.Tensor]:
        leaves = {
            name: inputs[name].detach().clone().requires_grad_(name in DIFFERENTIABLE)
            for name in KERNEL_ARGS
        }
        y, m = fn(*(leaves[name] for name in KERNEL_ARGS))
        ((y * probe_y).sum() + (m * probe_m).sum()).backward()
        return {name: leaves[name].grad for name in DIFFERENTIABLE}

    reference = grads(sequential_sparse_delta)
    chunkwise = grads(lambda *a: chunked(*a, chunk_size=8, holder=holder))
    for name in DIFFERENTIABLE:
        assert max_diff(chunkwise[name], reference[name]) < EXACT, name


@pytest.mark.parametrize("holder", HOLDERS)
def test_gradcheck_chunkwise(holder):
    inputs = make_inputs(
        batch=1, heads=2, seq_len=7, n_slots=9, n_writes=2, n_reads=2, d_v=3
    )
    names = DIFFERENTIABLE

    def fn(*leaves: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        full = dict(inputs)
        full.update(zip(names, leaves))
        return chunked(*args(full), chunk_size=4, holder=holder)

    leaves = tuple(inputs[name].detach().clone().requires_grad_(True) for name in names)
    assert torch.autograd.gradcheck(fn, leaves)


# ── the two holders ───────────────────────────────────────────────────────
#
# The arena writes the table in place and carries its gradient by hand; the
# functional holder is plain autograd.  What the arena claims, held here: the
# same forward bit for bit, the same gradient up to summation order, a backward
# that can be run again, and nothing retained that the functional one drops.


def _saved_storages(fn) -> list[tuple[int, int]]:
    """`(numel, nbytes)` of every storage autograd saves while ``fn`` runs, once each."""
    seen: dict[int, tuple[int, int]] = {}

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        storage = tensor.untyped_storage()
        seen.setdefault(
            storage.data_ptr(), (storage.nbytes() // tensor.element_size(), storage.nbytes())
        )
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        fn()
    return list(seen.values())


def _leaves(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: inputs[name].detach().clone().requires_grad_(name in DIFFERENTIABLE)
        for name in KERNEL_ARGS
    }


@pytest.mark.parametrize("recording", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_the_holders_agree_bit_for_bit_in_the_forward(dtype, recording):
    """Same arithmetic, different bookkeeping: the forward cannot tell them apart.

    With autograd recording and without, from a shared initial table that the
    arena copies in and the functional holder concatenates.
    """
    inputs = make_inputs(dtype=dtype, seq_len=45)
    inputs["memory"] = inputs["memory"][0]  # (H, N, d_v), broadcast over the batch
    results = {}
    for holder in HOLDERS:
        leaves = _leaves(inputs)
        with torch.set_grad_enabled(recording):
            results[holder] = chunked(
                *(leaves[name] for name in KERNEL_ARGS), chunk_size=8, holder=holder
            )
    for a, b in zip(results["in_place"], results["functional"]):
        assert torch.equal(a, b)


def test_the_holders_gradients_differ_only_in_summation_order():
    """A row's gradient is summed from the same terms by both; only the order moves."""
    inputs = make_inputs(seq_len=45)
    generator = torch.Generator().manual_seed(12)
    probe_y = torch.randn(2, 2, 45, 5, generator=generator, dtype=torch.float64)
    probe_m = torch.randn(2, 2, 24, 5, generator=generator, dtype=torch.float64)
    grads = {}
    for holder in HOLDERS:
        leaves = _leaves(inputs)
        y, m = chunked(*(leaves[name] for name in KERNEL_ARGS), chunk_size=8, holder=holder)
        ((y * probe_y).sum() + (m * probe_m).sum()).backward()
        grads[holder] = {name: leaves[name].grad for name in DIFFERENTIABLE}
    for name in DIFFERENTIABLE:
        assert max_diff(grads["in_place"][name], grads["functional"][name]) < 1e-12, name


def test_the_arena_can_be_backpropagated_twice():
    """``retain_graph``: each backward starts its own table gradient.

    The gradient buffer is handed down the token chain and mutated as it goes,
    so a second pass that inherited the first one's buffer would be silently
    wrong rather than refused.
    """
    leaves = _leaves(make_inputs(seq_len=29))
    y, m = chunked(*(leaves[name] for name in KERNEL_ARGS), chunk_size=8, holder="in_place")
    loss = y.square().sum() + m.square().sum()
    wrt = [leaves[name] for name in DIFFERENTIABLE]
    first = torch.autograd.grad(loss, wrt, retain_graph=True)
    second = torch.autograd.grad(loss, wrt)
    for name, a, b in zip(DIFFERENTIABLE, first, second):
        assert torch.equal(a, b), name


def test_double_backward_is_the_functional_holders_and_refused_by_the_arena():
    """Second order through the table: offered by one holder, refused -- not
    wrong -- by the other."""
    inputs = make_inputs(
        batch=1, heads=1, seq_len=6, n_slots=6, n_writes=2, n_reads=2, d_v=2
    )
    names = ("memory", "v")

    def fn(holder: str):
        def run(*leaves: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            full = dict(inputs)
            full.update(zip(names, leaves))
            return chunked(*args(full), chunk_size=4, holder=holder)

        return run

    leaves = tuple(inputs[name].detach().clone().requires_grad_(True) for name in names)
    assert torch.autograd.gradgradcheck(fn("functional"), leaves)

    y, m = fn("in_place")(*leaves)
    (grad,) = torch.autograd.grad(y.square().sum() + m.square().sum(), leaves[0], create_graph=True)
    with pytest.raises(RuntimeError, match="differentiate twice"):
        grad.sum().backward()


@pytest.mark.parametrize("holder", HOLDERS)
def test_the_callers_table_is_never_written(holder):
    """The entering table is a caller's state -- forward and backward leave it be."""
    inputs = make_inputs()
    before = inputs["memory"].clone()
    leaves = _leaves(inputs)
    y, m = chunked(*(leaves[name] for name in KERNEL_ARGS), chunk_size=8, holder=holder)
    (y.sum() + m.square().sum()).backward()
    assert torch.equal(leaves["memory"], before)
    assert m.untyped_storage().data_ptr() != leaves["memory"].untyped_storage().data_ptr()


def test_under_a_transform_the_functional_holder_is_chosen():
    """The default picks the holder ``torch.func`` can batch; forcing the arena
    under one is refused with the reason, not with a transform's own error."""
    inputs = make_inputs()
    tables = torch.stack([inputs["memory"], inputs["memory"].flip(-2)])
    rest = args(inputs)[1:]

    def run(memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return chunk_sparse_delta(memory, *rest, chunk_size=8)

    y, m = torch.func.vmap(run)(tables)
    for index in range(2):
        y_one, m_one = run(tables[index])
        assert max_diff(y[index], y_one) < EXACT
        assert max_diff(m[index], m_one) < EXACT

    with pytest.raises(ValueError, match="torch.func transform"):
        torch.func.vmap(
            lambda memory: chunk_sparse_delta(memory, *rest, chunk_size=8, in_place=True)
        )(tables)


def test_recompute_is_refused_under_a_transform():
    """Recomputation rides on saved-tensor hooks, which ``torch.func`` refuses;
    the kernel says so itself rather than surfacing the transform's error."""
    inputs = make_inputs()
    tables = torch.stack([inputs["memory"], inputs["memory"].flip(-2)])
    rest = args(inputs)[1:]
    with pytest.raises(ValueError, match="recompute_pairwise"):
        torch.func.vmap(
            lambda memory: chunk_sparse_delta(
                memory, *rest, chunk_size=8, recompute_pairwise=True
            )
        )(tables)


def test_without_gradients_recompute_is_no_reason_to_refuse():
    """Without gradients the dial has nothing to rebuild, so a no-grad pass
    under ``vmap`` -- parameter sets evaluated at once -- runs, and is the plain
    path.  The boundary is grad mode *inside* the transform: ``torch.func.grad``
    turns gradients on whatever surrounds it, so it is still refused."""
    inputs = make_inputs()
    tables = torch.stack([inputs["memory"], inputs["memory"].flip(-2)])
    rest = args(inputs)[1:]

    def run(recompute: bool):
        return lambda memory: chunk_sparse_delta(
            memory, *rest, chunk_size=8, recompute_pairwise=recompute
        )

    with torch.no_grad():
        y, m = torch.func.vmap(run(True))(tables)
        y_plain, m_plain = torch.func.vmap(run(False))(tables)
        with pytest.raises(ValueError, match="recompute_pairwise"):
            torch.func.grad(lambda memory: run(True)(memory)[0].sum())(inputs["memory"])
    assert torch.equal(y, y_plain) and torch.equal(m, m_plain)


@pytest.mark.parametrize("holder", HOLDERS)
def test_nothing_table_sized_is_kept_for_backward(holder):
    """Autograd keeps gathered rows and pairwise terms -- never a table.

    A table far larger than anything per-chunk, so a retained copy of it could
    not hide among the rest.
    """
    inputs = make_inputs(n_slots=400)
    leaves = _leaves(inputs)
    table_bytes = leaves["memory"].nbytes

    def run() -> None:
        chunked(*(leaves[name] for name in KERNEL_ARGS), chunk_size=8, holder=holder)

    largest = max(nbytes for _, nbytes in _saved_storages(run))
    assert largest < table_bytes


@pytest.mark.parametrize("holder", HOLDERS)
def test_each_chunk_keeps_its_gathered_write_rows_once(holder):
    """At the size of a chunk's write set, autograd keeps the gathered rows -- once.

    `W != R`, so a storage of exactly the write set's size is either the rows
    gathered at the write indices or the rows written back, which no backward
    needs.  ``index_copy`` kept the latter -- a fifth of everything a training
    step retained, measured at the defaults -- and this is what keeps it gone.
    """
    batch, heads, seq_len, chunk, n_writes, d_v = 2, 2, 48, 8, 4, 5
    inputs = make_inputs(n_writes=n_writes, n_reads=3, seq_len=seq_len)
    leaves = _leaves(inputs)
    write_set = batch * heads * chunk * n_writes * d_v

    def run() -> None:
        chunked(*(leaves[name] for name in KERNEL_ARGS), chunk_size=chunk, holder=holder)

    kept = [numel for numel, _ in _saved_storages(run) if numel == write_set]
    assert len(kept) == seq_len // chunk


@pytest.mark.parametrize("holder", HOLDERS)
def test_recomputing_the_pairwise_terms_changes_what_is_kept_and_nothing_else(holder):
    """Rebuilt in the backward, the table-free terms are the same arithmetic on
    the same inputs: outputs and gradients agree bit for bit.  What changes is
    what autograd keeps: of everything pairwise-sized, only the carry -- one
    array per group, needed by the table-touching step -- stays.

    `W != R`, so a group's write-side pairwise arrays and read-side ones have
    different sizes and each can be counted.
    """
    batch, heads, seq_len, chunk, group, n_writes, n_reads = 2, 2, 48, 8, 3, 4, 3
    inputs = make_inputs(
        batch=batch, heads=heads, seq_len=seq_len, n_writes=n_writes, n_reads=n_reads
    )
    probe = torch.randn(batch, heads, seq_len, 5, generator=torch.Generator().manual_seed(12),
                        dtype=torch.float64)
    cells = {k: batch * heads * group * chunk * k * chunk for k in (n_writes, n_reads)}
    n_groups = seq_len // (chunk * group)

    results = {}
    for recompute in (False, True):
        leaves = _leaves(inputs)
        out: dict[str, torch.Tensor] = {}

        def run() -> None:
            out["y"], out["m"] = chunked(
                *(leaves[name] for name in KERNEL_ARGS), chunk_size=chunk, group=group,
                holder=holder, recompute_pairwise=recompute,
            )

        kept = [numel for numel, _ in _saved_storages(run)]
        ((out["y"] * probe).sum() + out["m"].square().sum()).backward()
        grads = [leaves[name].grad for name in DIFFERENTIABLE]
        results[recompute] = (out["y"], out["m"], grads, kept)

    (y, m, grads, kept), (y_rc, m_rc, grads_rc, kept_rc) = results[False], results[True]
    assert torch.equal(y, y_rc) and torch.equal(m, m_rc)
    for name, a, b in zip(DIFFERENTIABLE, grads, grads_rc):
        assert torch.equal(a, b), name
    assert kept.count(cells[n_writes]) > n_groups and kept.count(cells[n_reads]) > 0
    assert kept_rc.count(cells[n_writes]) == n_groups  # the carry
    assert kept_rc.count(cells[n_reads]) == 0


def test_a_stack_recomputes_through_the_arena():
    """Non-reentrant checkpointing replays the forward and must land on the same
    gradients -- the arena's indices are saved where checkpointing can see them.
    With each layer rebuilding its pairwise terms as well, the two recomputes
    nest, and still land on the same gradients."""

    def block(index: int) -> Block:
        config = SparseDeltaMemoryConfig(
            d_model=D_MODEL, n_heads=2, n_slots=16, initial_memory="learned",
            n_writes=3, n_reads=4, chunk_size=4,
        )
        return Block(D_MODEL, SparseDeltaMemory(config), norm_eps=1e-5, d_mlp=48)

    torch.manual_seed(0)
    trunk = Stack(D_MODEL, 2, block, norm_eps=1e-5).double().train()
    with torch.no_grad():
        for b in trunk.blocks:
            b.mixer.initial_memory.normal_()
    x = sequence()

    grads = {}
    for recompute in (False, True):
        for pairwise in (False, True):
            trunk.recompute = recompute
            for b in trunk.blocks:
                b.mixer.recompute_pairwise = pairwise
            trunk.zero_grad(set_to_none=True)
            trunk(x).square().sum().backward()
            grads[recompute, pairwise] = {
                name: p.grad.clone() for name, p in trunk.named_parameters()
            }
    for case in grads:
        for name in grads[False, False]:
            assert torch.equal(grads[case][name], grads[False, False][name]), (case, name)


# ── preconditions ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("path", ["sequential", "chunk"])
def test_a_position_writing_one_slot_twice_is_refused(path):
    inputs = make_inputs()
    write_idx = inputs["write_idx"].clone()
    write_idx[1, 0, 17, 2] = write_idx[1, 0, 17, 0]
    inputs["write_idx"] = write_idx
    with pytest.raises(ValueError, match="same slot twice"):
        if path == "sequential":
            sequential_sparse_delta(*args(inputs))
        else:
            chunk_sparse_delta(*args(inputs), chunk_size=8)


def test_inconsistent_shapes_are_refused():
    inputs = make_inputs()
    inputs["write_val"] = inputs["write_val"][..., :-1]
    with pytest.raises(ValueError, match="share a shape"):
        chunk_sparse_delta(*args(inputs), chunk_size=8)


def test_chunk_size_must_be_positive():
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_sparse_delta(*args(make_inputs()), chunk_size=0)


def test_group_must_be_positive():
    with pytest.raises(ValueError, match="group"):
        chunk_sparse_delta(*args(make_inputs()), chunk_size=8, group=0)


# ══ the layer ══════════════════════════════════════════════════════════════

from dataclasses import FrozenInstanceError  # noqa: E402

from lumen import Block, Stack  # noqa: E402
from lumen.sdm import (  # noqa: E402
    SparseDeltaMemory,
    SparseDeltaMemoryConfig,
    SparseDeltaMemoryState,
)

D_MODEL = 24


def make_layer(
    initial_memory: str = "learned",
    *,
    seed: int = 0,
    trained_table: bool = True,
    dtype: torch.dtype = torch.float64,
    **overrides,
) -> SparseDeltaMemory:
    """A small layer; a learned table is given random contents by default.

    A zero-initialised learned table IS the zero layer (tested below), so a
    streaming test run on one would pass without ever reading a learned value.
    """
    torch.manual_seed(seed)
    fields = dict(
        d_model=D_MODEL,
        n_heads=2,
        n_slots=16,
        initial_memory=initial_memory,
        n_writes=3,
        n_reads=4,
        chunk_size=4,
    )
    fields.update(overrides)
    layer = SparseDeltaMemory(SparseDeltaMemoryConfig(**fields)).to(dtype)
    if trained_table and layer.initial_memory is not None:
        with torch.no_grad():
            layer.initial_memory.normal_()
    return layer


def sequence(batch: int = 2, seq_len: int = 22, seed: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, seq_len, D_MODEL, generator=generator, dtype=torch.float64)


# ── configuration ─────────────────────────────────────────────────────────


def test_initial_memory_is_required():
    """Two models behind one field, so there is no honest default."""
    with pytest.raises(TypeError, match="initial_memory"):
        SparseDeltaMemoryConfig(d_model=24, n_heads=2, n_slots=16)


@pytest.mark.parametrize(
    "overrides, match",
    [
        (dict(n_slots=15), "perfect square"),
        (dict(n_slots=0), "perfect square"),
        (dict(n_heads=5), "divisor"),
        (dict(initial_memory="random"), "initial_memory"),
        (dict(n_writes=17), "n_writes"),
        (dict(n_reads=0), "n_reads"),
        (dict(key_norm="sparsemax"), "key_norm"),
        (dict(decay_weighting="slot"), "decay_weighting"),
        (dict(decay_weighting="key", key_norm="l2"), "softmax keys only"),
        (dict(beta_max=2.1), "beta_max"),
        (dict(beta_max=0.0), "beta_max"),
        (dict(chunk_size=0), "chunk_size"),
        (dict(norm_eps=0.0), "norm_eps"),
        (dict(dropout=1.0), "dropout"),
    ],
)
def test_config_rejects_bad_values(overrides, match):
    # Small W and R, so that the defaults (64) do not trip the n_writes check
    # before the field under test is reached.
    fields = dict(
        d_model=24, n_heads=2, n_slots=16, initial_memory="zero", n_writes=3, n_reads=3
    )
    fields.update(overrides)
    with pytest.raises(ValueError, match=match):
        SparseDeltaMemoryConfig(**fields)


# ── addressing ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("n_select", [1, 3, 4, 7, 16])
def test_product_keys_find_the_exact_top_k(n_select):
    """Against the brute force over all `N` outer-sum scores.

    `n_select` crosses `√N = 4`, where the per-half candidate count switches
    from `K` to `√N`.
    """
    layer = make_layer(n_writes=n_select)
    root = layer.config.sub_keys
    scores = layer.write_proj(sequence())
    slot, weight = layer._address(scores, n_select)

    halves = scores.view(2, 22, 2, 2, root).permute(0, 2, 1, 3, 4)
    every = (halves[..., 0, :, None] + halves[..., 1, None, :]).flatten(-2)
    best, expected = every.topk(n_select, dim=-1)

    assert torch.equal(slot.sort(-1).values, expected.sort(-1).values)
    ordered = slot.argsort(-1)
    brute = expected.argsort(-1)
    torch.testing.assert_close(
        weight.gather(-1, ordered), torch.softmax(best, -1).gather(-1, brute)
    )
    # Distinct within every position: the kernels' precondition, by construction.
    assert bool((slot.sort(-1).values.diff(dim=-1) > 0).all())


def test_key_norms_produce_what_they_claim():
    x = sequence()
    softmax = make_layer(key_norm="softmax")
    _, weight = softmax._address(softmax.write_proj(x), 3)
    torch.testing.assert_close(weight.sum(-1), torch.ones_like(weight[..., 0]))
    assert bool((weight >= 0).all())

    l2 = make_layer(key_norm="l2")
    _, weight = l2._address(l2.write_proj(x), 3)
    torch.testing.assert_close(weight.norm(dim=-1), torch.ones_like(weight[..., 0]))


# ── forward, step, streaming ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "overrides",
    [
        dict(initial_memory="zero"),
        dict(initial_memory="learned"),
        dict(initial_memory="learned", key_norm="l2"),
        dict(initial_memory="learned", decay_weighting="key"),
    ],
    ids=["zero", "learned", "l2", "key-decay"],
)
def test_layer_forward_shape_and_finiteness(overrides):
    layer = make_layer(**overrides)
    y = layer(sequence())
    assert y.shape == (2, 22, D_MODEL)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("initial_memory", ["zero", "learned"])
def test_step_matches_forward(initial_memory):
    layer = make_layer(initial_memory)
    x = sequence()
    y_full = layer(x)

    state = layer.init_state(2)
    steps = []
    for t in range(x.shape[1]):
        y_t, state = layer.step(x[:, t : t + 1], state)
        steps.append(y_t)
    assert max_diff(torch.cat(steps, dim=1), y_full) < EXACT


def test_prefill_then_step_matches_one_pass():
    layer = make_layer()
    x = sequence()
    y_full = layer(x)
    y_prefix, state = layer(x[:, :13], return_state=True)
    tail = []
    for t in range(13, x.shape[1]):
        y_t, state = layer.step(x[:, t : t + 1], state)
        tail.append(y_t)
    assert max_diff(torch.cat([y_prefix, *tail], dim=1), y_full) < EXACT


@pytest.mark.parametrize("split", [3, 8, 17])
def test_chunked_forward_matches_one_pass(split):
    layer = make_layer()
    x = sequence()
    y_full, s_full = layer(x, return_state=True)
    y_a, state = layer(x[:, :split], return_state=True)
    y_b, s_split = layer(x[:, split:], state=state, return_state=True)
    assert max_diff(torch.cat([y_a, y_b], dim=1), y_full) < EXACT
    assert max_diff(s_split.memory, s_full.memory) < EXACT


def test_read_is_a_step_with_the_write_switched_off():
    """`β = 0` and `α = 1` through the biases: a step becomes a read, bit for bit.

    Both saturate exactly in fp64 -- `sigmoid(−1e4)` and `softplus(−1e4)` are
    0 -- so the step writes back exactly the rows it found and reads them with
    the address :meth:`read` forms.
    """
    layer = make_layer()
    with torch.no_grad():
        layer.b_proj.bias.fill_(-1e4)
        layer.a_proj.bias.fill_(-1e4)
    x = sequence(seq_len=1)
    state = layer.init_state(2)
    _, state = layer(sequence(seq_len=9, seed=5), state=state, return_state=True)

    before = state.memory.clone()
    y_read = layer.read(x, state)
    y_step, after = layer.step(x, state)
    assert torch.equal(y_read, y_step)
    assert torch.equal(after.memory, before)
    assert torch.equal(state.memory, before)


def test_a_block_holds_it_and_streams():
    torch.manual_seed(0)
    config = SparseDeltaMemoryConfig(
        d_model=D_MODEL, n_heads=2, n_slots=16, initial_memory="learned",
        n_writes=3, n_reads=4, chunk_size=4,
    )
    block = Block(D_MODEL, SparseDeltaMemory(config), norm_eps=1e-5, d_mlp=48).double()
    x = sequence()
    y_full = block(x)
    state = block.init_state(2)
    steps = []
    for t in range(x.shape[1]):
        y_t, state = block.step(x[:, t : t + 1], state)
        steps.append(y_t)
    assert max_diff(torch.cat(steps, dim=1), y_full) < EXACT


def test_a_stack_keeps_the_gate_bias_and_the_empty_table():
    """The stack's init pass redraws weights, never biases, never bare tables."""

    def block(index: int) -> Block:
        config = SparseDeltaMemoryConfig(
            d_model=D_MODEL, n_heads=2, n_slots=16, initial_memory="learned",
            n_writes=3, n_reads=4,
        )
        return Block(D_MODEL, SparseDeltaMemory(config), norm_eps=1e-5, d_mlp=0)

    trunk = Stack(D_MODEL, 2, block, norm_eps=1e-5)
    for b in trunk.blocks:
        assert torch.equal(b.mixer.a_proj.bias, torch.full_like(b.mixer.a_proj.bias, -3.0))
        assert torch.equal(b.mixer.initial_memory, torch.zeros_like(b.mixer.initial_memory))
    assert all(p in trunk.residual_out_projections() for b in trunk.blocks
               for p in b.mixer.residual_out_projections())


# ── the initial table ─────────────────────────────────────────────────────


def test_a_learned_table_at_init_is_the_zero_layer_exactly():
    """Same seed, same projections, same (empty) table: the same outputs, bit for bit."""
    zero = make_layer("zero", trained_table=False)
    learned = make_layer("learned", trained_table=False)
    for name, parameter in zero.named_parameters():
        assert torch.equal(parameter, dict(learned.named_parameters())[name]), name
    x = sequence()
    y_zero, s_zero = zero(x, return_state=True)
    y_learned, s_learned = learned(x, return_state=True)
    assert torch.equal(y_zero, y_learned)
    assert torch.equal(s_zero.memory, s_learned.memory)


def test_a_learned_table_is_live_once_moved():
    learned = make_layer("learned", trained_table=False)
    x = sequence()
    before = learned(x)
    with torch.no_grad():
        learned.initial_memory.normal_()
    assert not torch.allclose(learned(x), before)
    assert learned.initial_memory_parameters() == (learned.initial_memory,)
    assert make_layer("zero").initial_memory_parameters() == ()


def test_backward_reaches_every_parameter_including_an_empty_table():
    """A zero table still trains: `∂y/∂M₀[n] = q_n` wherever a read lands."""
    layer = make_layer("learned", trained_table=False)
    layer(sequence()).square().sum().backward()
    for name, parameter in layer.named_parameters():
        assert parameter.grad is not None, name
        assert bool(parameter.grad.abs().sum() > 0), name


def test_the_layer_rebuilds_its_pairwise_terms_only_when_asked():
    """A memory dial: off by default, not in the ``state_dict``, invisible in
    what the layer computes, and visible in what it keeps.  (Exactly what is
    kept, and several groups at once, are the kernel's gates; this is the layer
    passing the dial down.  Equal gradients alone would pass on a layer that
    ignored it.)"""
    layer = make_layer("learned").train()
    assert layer.recompute_pairwise is False
    keys = set(layer.state_dict())
    x = sequence(seq_len=45)
    grads, kept = {}, {}
    for on in (False, True):
        layer.recompute_pairwise = on
        layer.zero_grad(set_to_none=True)
        out: dict[str, torch.Tensor] = {}
        kept[on] = sum(
            nbytes for _, nbytes in _saved_storages(lambda: out.update(y=layer(x)))
        )
        out["y"].square().sum().backward()
        grads[on] = {name: p.grad.clone() for name, p in layer.named_parameters()}
    assert set(layer.state_dict()) == keys
    for name in grads[False]:
        assert torch.equal(grads[True][name], grads[False][name]), name
    assert kept[True] < kept[False]


def test_the_layer_rebuilds_its_pairwise_terms_only_while_training():
    """The dial is for training.  Outside it -- an evaluation that still wants
    gradients, say -- the layer keeps exactly what it keeps with the dial off,
    and computes the same thing."""
    layer = make_layer("learned").eval()
    x = sequence(seq_len=45)
    ys, kept = {}, {}
    for on in (False, True):
        layer.recompute_pairwise = on
        out: dict[str, torch.Tensor] = {}
        kept[on] = sorted(_saved_storages(lambda: out.update(y=layer(x))))
        ys[on] = out["y"]
    assert kept[True] == kept[False]
    assert torch.equal(ys[True], ys[False])


def test_a_layer_trained_with_the_dial_on_still_serves_under_a_transform():
    """Recomputation is refused under ``torch.func`` with gradients on (§3.12),
    so the dial must not follow a layer out of training.  Switched to eval, a
    layer that trained with it on runs under ``vmap`` over stacked parameter
    sets and agrees with each set run alone.  Left in training, it is refused,
    by name -- unless gradients are off, where the dial has nothing to do and
    the layer runs as well.

    Distinct parameter sets, so a transform that broadcast one set across the
    others could not pass.
    """
    models = [make_layer("learned", seed=seed).eval() for seed in (0, 1)]
    params, buffers = torch.func.stack_module_state(models)
    base = make_layer("learned", seed=2)
    base.recompute_pairwise = True
    x = sequence()

    def run(p: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.func.functional_call(base, (p, b), (x,))

    base.train()
    with pytest.raises(ValueError, match="recompute_pairwise"):
        torch.func.vmap(run)(params, buffers)
    with torch.no_grad():
        in_training = torch.func.vmap(run)(params, buffers)

    base.eval()
    batched = torch.func.vmap(run)(params, buffers)
    for model, y, y_training in zip(models, batched, in_training):
        expected = model(x)
        torch.testing.assert_close(y, expected, rtol=0, atol=1e-12)
        torch.testing.assert_close(y_training, expected, rtol=0, atol=1e-12)


def test_zero_and_learned_checkpoints_do_not_silently_mix():
    zero, learned = make_layer("zero"), make_layer("learned")
    with pytest.raises(RuntimeError, match="initial_memory"):
        learned.load_state_dict(zero.state_dict())
    with pytest.raises(RuntimeError, match="initial_memory"):
        zero.load_state_dict(learned.state_dict())


def test_checkpoint_round_trips():
    layer, fresh = make_layer(seed=0), make_layer(seed=1)
    fresh.load_state_dict(layer.state_dict())
    x = sequence()
    assert torch.equal(fresh(x), layer(x))


# ── the state ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("initial_memory", ["zero", "learned"])
def test_init_state_is_a_broadcast_view(initial_memory):
    """No stream pays for a table before its first write."""
    layer = make_layer(initial_memory)
    state = layer.init_state(5)
    assert state.memory.shape == (5, 2, 16, D_MODEL // 2)
    assert state.memory.stride(0) == 0
    assert state.memory.dtype == torch.float64
    if initial_memory == "learned":
        assert state.memory.data_ptr() == layer.initial_memory.data_ptr()


def test_state_is_frozen_and_step_returns_a_successor():
    layer = make_layer()
    state = layer.init_state(2)
    _, state = layer(sequence(seq_len=5), state=state, return_state=True)
    before = state.memory.clone()
    _, successor = layer.step(sequence(seq_len=1, seed=9), state)
    assert successor is not state
    assert torch.equal(state.memory, before)
    with pytest.raises(FrozenInstanceError):
        state.memory = before


def test_state_size_is_flat_in_generated_length():
    layer = make_layer()
    state = layer.init_state(1)
    shape = state.memory.shape
    for t in range(40):
        _, state = layer.step(sequence(batch=1, seq_len=1, seed=t), state)
        assert state.memory.shape == shape


def test_step_refuses_more_than_one_position():
    layer = make_layer()
    with pytest.raises(ValueError, match="one position"):
        layer.step(sequence(seq_len=2), layer.init_state(2))


def test_residual_out_projections_is_the_output():
    layer = make_layer()
    assert layer.residual_out_projections() == (layer.o_proj,)


# ── on a device ───────────────────────────────────────────────────────────
# CI is CPU-only; these run locally with `pytest -m gpu`.  The kernels have no
# compute-capability floor to test -- they are torch ops -- so what is checked
# is that the arithmetic and the determinism claim survive the move.


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_chunkwise_matches_sequential_on_cuda(dtype):
    inputs = {
        name: tensor.cuda()
        for name, tensor in make_inputs(dtype=dtype, seq_len=61).items()
    }
    y_ref, m_ref = sequential_sparse_delta(*args(inputs))
    y, m = chunk_sparse_delta(*args(inputs), chunk_size=8)
    bound = EXACT if dtype == torch.float64 else FP32
    assert max_diff(y, y_ref) < bound
    assert max_diff(m, m_ref) < bound


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_the_chunkwise_forward_is_deterministic_on_cuda():
    """No scatter-add into a repeated destination, so no atomics to reorder.

    The claim the partner layout and the scratch rows exist to make true, held
    where it could fail: on a CPU every reduction order is already fixed.
    """
    inputs = {
        name: tensor.cuda()
        for name, tensor in make_inputs(dtype=torch.float32, seq_len=61).items()
    }
    first = chunk_sparse_delta(*args(inputs), chunk_size=8)
    second = chunk_sparse_delta(*args(inputs), chunk_size=8)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("layout", ["collide", "same", "disjoint", "miss", "fixture"])
def test_sorted_partners_are_the_dense_compare_on_cuda(layout):
    """The device's sort and search, against the device's compare."""
    w_idx, r_idx = (t.cuda() for t in chunk_indices(layout, chunk=16))
    for got, spec in zip(_partners(w_idx, r_idx), _partners_dense(w_idx, r_idx)):
        assert torch.equal(got, spec)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_the_holders_agree_on_cuda(dtype):
    """Forward bit for bit; gradients to round-off -- on the device whose
    ``index_add_`` reorders its sums from run to run."""
    inputs = {
        name: tensor.cuda()
        for name, tensor in make_inputs(dtype=dtype, seq_len=61).items()
    }
    results, grads = {}, {}
    for holder in HOLDERS:
        leaves = _leaves(inputs)
        y, m = chunked(*(leaves[name] for name in KERNEL_ARGS), chunk_size=8, holder=holder)
        (y.square().sum() + m.square().sum()).backward()
        results[holder] = (y.detach(), m.detach())
        grads[holder] = [leaves[name].grad for name in DIFFERENTIABLE]
    for a, b in zip(results["in_place"], results["functional"]):
        assert torch.equal(a, b)
    bound = EXACT if dtype == torch.float64 else FP32
    for name, a, b in zip(DIFFERENTIABLE, grads["in_place"], grads["functional"]):
        assert max_diff(a, b) < bound, name


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("holder", HOLDERS)
def test_the_backward_is_reproducible_on_cuda_when_asked(holder):
    """A gather's backward is a scatter-add into repeated rows, so by default its
    sums reorder run to run.  Under ``torch.use_deterministic_algorithms`` they
    do not -- the claim the module docstring makes, held on the device.

    ``warn_only``: cuBLAS asks for a workspace setting that must be in the
    environment before it initialises, which a test cannot arrange; the
    matmuls here reproduce without it, and the equality below is the check.
    """
    inputs = {
        name: tensor.cuda()
        for name, tensor in make_inputs(dtype=torch.float32, seq_len=61).items()
    }

    def grads() -> list[torch.Tensor]:
        leaves = _leaves(inputs)
        y, m = chunked(*(leaves[name] for name in KERNEL_ARGS), chunk_size=8, holder=holder)
        (y.square().sum() + m.square().sum()).backward()
        return [leaves[name].grad for name in DIFFERENTIABLE]

    was = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            runs = [grads() for _ in range(3)]
    finally:
        torch.use_deterministic_algorithms(was)
    for run in runs[1:]:
        for name, a, b in zip(DIFFERENTIABLE, runs[0], run):
            assert torch.equal(a, b), name


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("holder", HOLDERS)
def test_recomputing_the_pairwise_terms_changes_nothing_on_cuda(holder):
    """On the device as on the host: with the backward made reproducible, the
    rebuilt terms give the same gradients bit for bit, over several groups."""
    inputs = {
        name: tensor.cuda()
        for name, tensor in make_inputs(dtype=torch.float32, seq_len=61).items()
    }

    def run(recompute: bool) -> list[torch.Tensor]:
        leaves = _leaves(inputs)
        y, m = chunked(
            *(leaves[name] for name in KERNEL_ARGS), chunk_size=8, group=2,
            holder=holder, recompute_pairwise=recompute,
        )
        (y.square().sum() + m.square().sum()).backward()
        return [y.detach(), m.detach()] + [leaves[name].grad for name in DIFFERENTIABLE]

    was = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            plain, rebuilt = run(False), run(True)
    finally:
        torch.use_deterministic_algorithms(was)
    for name, a, b in zip(("y", "m", *DIFFERENTIABLE), plain, rebuilt):
        assert torch.equal(a, b), name


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_the_layer_streams_on_cuda_without_a_device_argument():
    """Build, move, init, step -- with no caller ever naming a device."""
    layer = make_layer(dtype=torch.float32).cuda()
    x = sequence().float().cuda()
    state = layer.init_state(2)
    assert state.memory.device == x.device

    steps = []
    for t in range(x.shape[1]):
        y_t, state = layer.step(x[:, t : t + 1], state)
        steps.append(y_t)
    assert max_diff(torch.cat(steps, dim=1), layer(x)) < FP32
