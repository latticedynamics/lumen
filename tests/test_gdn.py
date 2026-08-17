"""Gated DeltaNet — the acceptance conditions of the design record, as tests.

Organised the way the record is: layout validity, then the kernel against its
oracle, then the layer, then streaming, then the properties that are claimed
"by construction" and therefore have to be held to it.

Tolerance convention (design record §3.7): the chunkwise and sequential paths
are the *same algorithm reached by two routes*, so in fp64 they agree to
round-off and the gate is tight.  In fp32 the chunkwise form reassociates the
sum and the difference is arithmetic, not structural — so fp32 gets a bound
consistent with fp32, never the fp64 one.
"""

from __future__ import annotations

import itertools
import math

import pytest
import torch
import torch.nn.functional as F

from lumen.gdn import (
    GatedDeltaNet,
    GatedDeltaNetConfig,
    GatedDeltaNetState,
    HeadLayout,
    ShortConv,
)
from lumen.gdn.reference import (
    assign_values,
    chunk_gated_delta,
    inv_unit,
    inv_unit_lower,
    inv_unit_lower_small,
    sequential_gated_delta,
)

EXACT = 1e-9  # fp64: structural agreement
FP32 = 5e-5  # fp32: round-off on these shapes, deliberately not tighter

LAYOUTS = {
    "shared_key": HeadLayout.shared_key(8),
    "diagonal": HeadLayout.diagonal(8),
    "shared_value": HeadLayout.shared_value(8),
    "crossed_2x4": HeadLayout.crossed(2, 4),
    "crossed_4x2": HeadLayout.crossed(4, 2),
    "single": HeadLayout.diagonal(1),
}


def make_inputs(
    layout, seq_len=128, batch=2, d_k=16, d_v=12, dtype=torch.float64, decay="shared"
):
    """Kernel-shaped inputs obeying the layer's own invariants.

    ``decay`` selects the state axis of `log_alpha`: ``"shared"`` gives one
    timescale per key group (axis 1, the shipped arrangement) and ``"per_state"``
    gives every state its own as a geometric band of half-lives — §3.4's
    motivating case, and the one that has to reach the oracle.

    **The decay is unclamped, deliberately.**  This fixture used to end with
    ``.clamp(min=-8.0 / chunk)`` -- `max_chunk_decay` hard-coded into the test
    inputs -- so every equivalence test ran inside the range the old `1/gamma`
    factorisation could survive, and none of them ever visited the range it
    could not.  With the clamp gone from the layer, leaving it here would have
    made these tests pass without exercising anything the change opened up.
    Unclamped `-softplus(randn)` accumulates to roughly `-110` over 128
    positions, which is past fp32's `exp` ceiling of 88.7 and would have been
    `nan` under the old form.
    """
    g_k = layout.n_key_groups
    m = layout.states_per_key_group
    q = F.normalize(torch.randn(batch, g_k, m, seq_len, d_k, dtype=dtype), dim=-1)
    k = F.normalize(torch.randn(batch, g_k, 1, seq_len, d_k, dtype=dtype), dim=-1)
    v = assign_values(
        torch.randn(batch, seq_len, layout.n_value_groups, d_v, dtype=dtype), layout
    )
    beta = 2.0 * torch.sigmoid(torch.randn(batch, g_k, 1, seq_len, dtype=dtype))

    states = 1 if decay == "shared" else m
    log_alpha = -F.softplus(torch.randn(batch, g_k, states, seq_len, dtype=dtype))
    if decay != "shared":
        # A geometric band across the states: two decades of half-life, which is
        # the spread someone turns this on to get.  Uniform rates would let the
        # per-state path pass by accident.
        band = torch.logspace(0, 2, m, dtype=dtype).view(1, 1, m, 1)
        log_alpha = log_alpha * band
    return q, k, v, beta, log_alpha


# ── head layout ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", list(LAYOUTS))
def test_layout_is_rectangular_and_consistent(name):
    layout = LAYOUTS[name]
    assert layout.n_heads == len(layout.key_group) == len(layout.value_group)
    assert layout.n_heads == layout.n_key_groups * layout.states_per_key_group
    assert len(layout.rows) == layout.n_key_groups
    assert all(len(row) == layout.states_per_key_group for row in layout.rows)


def test_every_named_layout_reindexes_for_free():
    """The design record claims the named layouts cost no copy.  Hold it to that."""
    for layout in LAYOUTS.values():
        assert layout.reindex_is_free, layout.describe()


def test_diagonal_is_not_a_crossed_layout():
    """The structural claim the whole naming decision rests on (§3.1).

    The crossed family is a Cartesian product and the ordinary arrangement is a
    diagonal; they coincide only at H = 1.  If this ever passes for H > 1, the
    argument for parameterising the assignment has evaporated.
    """
    assert HeadLayout.diagonal(1) == HeadLayout.crossed(1, 1)
    for h in (2, 4, 8):
        diagonal = HeadLayout.diagonal(h)
        crossed = {
            HeadLayout.crossed(g_k, h // g_k)
            for g_k in range(1, h + 1)
            if h % g_k == 0
        }
        assert diagonal not in crossed
        # and it is not crossed(h, h) either -- that has h**2 states
        assert HeadLayout.crossed(h, h).n_heads == h * h != diagonal.n_heads


def test_layout_rejects_malformed_assignments():
    with pytest.raises(ValueError, match="densely from 0"):
        HeadLayout(key_group=(0, 2), value_group=(0, 1))
    with pytest.raises(ValueError, match="non-decreasing"):
        HeadLayout(key_group=(1, 0), value_group=(0, 1))
    with pytest.raises(ValueError, match="rectangle rule"):
        HeadLayout(key_group=(0, 0, 1), value_group=(0, 1, 0))
    with pytest.raises(ValueError, match="same memory computed twice"):
        HeadLayout(key_group=(0, 0), value_group=(0, 0))
    with pytest.raises(ValueError, match="same length"):
        HeadLayout(key_group=(0, 1), value_group=(0,))


def test_layout_describe_round_trips():
    assert "diagonal" in HeadLayout.diagonal(4).describe()
    assert "shared_key" in HeadLayout.shared_key(4).describe()
    assert "shared_value" in HeadLayout.shared_value(4).describe()
    assert "crossed" in HeadLayout.crossed(2, 3).describe()


# ── the triangular inverse ────────────────────────────────────────────────


def test_three_inverses_agree():
    torch.manual_seed(0)
    n = (torch.randn(2, 3, 64, 64, dtype=torch.float64) * 0.3).tril(-1)
    trsm, blocked, small = inv_unit(n), inv_unit_lower(n), inv_unit_lower_small(n)
    assert (trsm - blocked).abs().max() < EXACT
    assert (blocked - small).abs().max() < EXACT

    eye = torch.eye(64, dtype=torch.float64)
    assert ((eye + n) @ trsm - eye.expand_as(n)).abs().max() < EXACT


def test_inverse_survives_correlated_keys():
    """The trap the reference's docstring warns about, made a test.

    The Neumann/binary-expansion identity is algebraically exact and blows up
    numerically once keys are correlated.  Forward substitution must not.
    """
    torch.manual_seed(0)
    key = F.normalize(torch.randn(1, 1, 64, 8, dtype=torch.float64), dim=-1)
    # heavily correlated: every key near the same direction
    key = F.normalize(key + 4.0 * key[:, :, :1, :], dim=-1)
    beta = 1.9 * torch.ones(1, 1, 64, 1, dtype=torch.float64)
    n = ((beta * key) @ key.transpose(-1, -2)).tril(-1)

    inverse = inv_unit(n)
    assert torch.isfinite(inverse).all()
    assert inverse.abs().max() < 1e3, "forward substitution should stay bounded"

    eye = torch.eye(64, dtype=torch.float64)
    assert ((eye + n) @ inverse - eye).abs().max() < 1e-9


# ── kernel vs oracle ──────────────────────────────────────────────────────


@pytest.mark.parametrize("name", list(LAYOUTS))
def test_chunkwise_matches_sequential_fp64(name):
    """Acceptance §6.1 — the chunkwise path is the recurrence, at every layout."""
    torch.manual_seed(0)
    layout = LAYOUTS[name]
    q, k, v, beta, log_alpha = make_inputs(layout)

    out_chunk, state_chunk = chunk_gated_delta(q, k, v, beta, log_alpha, 32)
    out_seq, state_seq = sequential_gated_delta(q, k, v, beta, log_alpha)

    assert (out_chunk - out_seq).abs().max() < EXACT
    assert (state_chunk - state_seq).abs().max() < EXACT


@pytest.mark.parametrize("chunk", [1, 2, 8, 32, 128])
def test_chunkwise_is_independent_of_chunk_size(chunk):
    """Chunking is an implementation detail; the answer must not know about it."""
    torch.manual_seed(0)
    layout = LAYOUTS["shared_key"]
    q, k, v, beta, log_alpha = make_inputs(layout, seq_len=128)

    out, state = chunk_gated_delta(q, k, v, beta, log_alpha, chunk)
    out_seq, state_seq = sequential_gated_delta(q, k, v, beta, log_alpha)

    assert (out - out_seq).abs().max() < EXACT
    assert (state - state_seq).abs().max() < EXACT


def test_chunkwise_matches_sequential_fp32():
    """The shipped dtype, held to a bound the arithmetic actually allows."""
    torch.manual_seed(0)
    layout = LAYOUTS["shared_key"]
    q, k, v, beta, log_alpha = make_inputs(layout, dtype=torch.float32)

    out_chunk, _ = chunk_gated_delta(q, k, v, beta, log_alpha, 32)
    out_seq, _ = sequential_gated_delta(q, k, v, beta, log_alpha)
    assert (out_chunk - out_seq).abs().max() < FP32


def test_state_carries_across_calls():
    """Splitting a sequence and threading the state == one pass over the whole."""
    torch.manual_seed(0)
    layout = LAYOUTS["crossed_2x4"]
    q, k, v, beta, log_alpha, = make_inputs(layout, seq_len=128)

    whole, final = chunk_gated_delta(q, k, v, beta, log_alpha, 32)

    split = 64
    first, mid = chunk_gated_delta(
        q[..., :split, :], k[..., :split, :], v[..., :split, :],
        beta[..., :split], log_alpha[..., :split], 32,
    )
    second, end = chunk_gated_delta(
        q[..., split:, :], k[..., split:, :], v[..., split:, :],
        beta[..., split:], log_alpha[..., split:], 32, mid,
    )

    assert (torch.cat([first, second], dim=-2) - whole).abs().max() < EXACT
    assert (end - final).abs().max() < EXACT


def test_zero_beta_is_pure_decay():
    """A sanity anchor with a closed form: no writes means the state only fades."""
    torch.manual_seed(0)
    layout = LAYOUTS["shared_key"]
    q, k, v, beta, log_alpha = make_inputs(layout, seq_len=64)
    beta = torch.zeros_like(beta)

    memory = torch.randn(2, 1, 8, 16, 12, dtype=torch.float64)
    _, final = chunk_gated_delta(q, k, v, beta, log_alpha, 16, memory)

    expected = memory * log_alpha.sum(-1)[..., None, None].exp().transpose(2, 2)
    assert (final - expected).abs().max() < EXACT


# ── per-state decay (§3.4's seam, taken) ──────────────────────────────────


def test_the_oracle_was_per_state_ready_before_the_fast_path():
    """`recurrent_gated_delta` needed no code for this, and that is a claim.

    `alpha[..., None, None]` broadcasts against a state of `(B, G_k, m, d_k, d_v)`
    whether the axis is 1 or m.  Relying on that silently would be how it quietly
    stops being true, so: an m-wide alpha whose rows agree must reproduce the
    shared answer *exactly* — same values, same elementwise multiply, no
    tolerance earned or needed.
    """
    torch.manual_seed(0)
    layout = LAYOUTS["shared_key"]
    q, k, v, beta, log_alpha = make_inputs(layout, seq_len=32)

    widened = log_alpha.expand(-1, -1, layout.states_per_key_group, -1).contiguous()
    shared_out, shared_state = sequential_gated_delta(q, k, v, beta, log_alpha)
    wide_out, wide_state = sequential_gated_delta(q, k, v, beta, widened)

    assert torch.equal(shared_out, wide_out)
    assert torch.equal(shared_state, wide_state)


@pytest.mark.parametrize("name", list(LAYOUTS))
def test_chunkwise_matches_sequential_with_per_state_decay(name):
    """The gate the whole feature answers to, at every layout.

    Same bound as the shared case: the chunkwise and sequential paths are the
    same recurrence reached by two routes, and per-state decay does not make
    that any less true.
    """
    torch.manual_seed(0)
    layout = LAYOUTS[name]
    q, k, v, beta, log_alpha = make_inputs(layout, decay="per_state")
    assert log_alpha.shape[2] == layout.states_per_key_group

    out_chunk, state_chunk = chunk_gated_delta(q, k, v, beta, log_alpha, 32)
    out_seq, state_seq = sequential_gated_delta(q, k, v, beta, log_alpha)

    assert (out_chunk - out_seq).abs().max() < EXACT
    assert (state_chunk - state_seq).abs().max() < EXACT


@pytest.mark.parametrize("chunk", [1, 2, 8, 32, 128])
def test_per_state_decay_is_independent_of_chunk_size(chunk):
    torch.manual_seed(0)
    layout = LAYOUTS["crossed_2x4"]
    q, k, v, beta, log_alpha = make_inputs(layout, seq_len=128, decay="per_state")

    out, state = chunk_gated_delta(q, k, v, beta, log_alpha, chunk)
    out_seq, state_seq = sequential_gated_delta(q, k, v, beta, log_alpha)

    assert (out - out_seq).abs().max() < EXACT
    assert (state - state_seq).abs().max() < EXACT


def test_per_state_decay_carries_state_across_calls():
    torch.manual_seed(0)
    layout = LAYOUTS["shared_key"]
    q, k, v, beta, log_alpha = make_inputs(layout, seq_len=128, decay="per_state")

    whole, final = chunk_gated_delta(q, k, v, beta, log_alpha, 32)

    split = 64
    first, mid = chunk_gated_delta(
        q[..., :split, :], k[..., :split, :], v[..., :split, :],
        beta[..., :split], log_alpha[..., :split], 32,
    )
    second, end = chunk_gated_delta(
        q[..., split:, :], k[..., split:, :], v[..., split:, :],
        beta[..., split:], log_alpha[..., split:], 32, mid,
    )

    assert (torch.cat([first, second], dim=-2) - whole).abs().max() < EXACT
    assert (end - final).abs().max() < EXACT


def test_per_state_decay_reduces_to_shared_when_the_rows_agree():
    """Turning the axis on without spreading the rates must change nothing.

    Not bit-exact, and deliberately so: the two paths reach the same object by
    opposite routes — damp-then-solve versus solve-then-damp — so agreement here
    is the homomorphism `inv((I+N)*rel) = inv(I+N)*rel` holding numerically, not
    a shape change.  That is exactly the claim the per-state route rests on.
    """
    torch.manual_seed(0)
    layout = LAYOUTS["crossed_2x4"]
    q, k, v, beta, log_alpha = make_inputs(layout, seq_len=64)

    widened = log_alpha.expand(-1, -1, layout.states_per_key_group, -1).contiguous()
    shared_out, shared_state = chunk_gated_delta(q, k, v, beta, log_alpha, 16)
    wide_out, wide_state = chunk_gated_delta(q, k, v, beta, widened, 16)

    assert (shared_out - wide_out).abs().max() < EXACT
    assert (shared_state - wide_state).abs().max() < EXACT


def test_per_state_decay_actually_gives_each_state_its_own_horizon():
    """Otherwise every test above passes on a feature that does nothing.

    Drive one state to forget fast and another slowly, with no writes at all, and
    the surviving memory must follow each state's own rate rather than a shared
    one.  Checked against the closed form, per state.
    """
    torch.manual_seed(0)
    layout = LAYOUTS["shared_key"]  # one key group, eight states
    m = layout.states_per_key_group
    q, k, v, beta, _ = make_inputs(layout, seq_len=64)
    beta = torch.zeros_like(beta)

    rates = -torch.logspace(-2, 0, m, dtype=torch.float64).view(1, 1, m, 1)
    log_alpha = rates.expand(2, 1, m, 64).contiguous()

    memory = torch.randn(2, 1, m, 16, 12, dtype=torch.float64)
    _, final = chunk_gated_delta(q, k, v, beta, log_alpha, 16, memory)

    expected = memory * log_alpha.sum(-1)[..., None, None].exp()
    assert (final - expected).abs().max() < EXACT

    # and the states genuinely differ: the fastest has forgotten, the slowest
    # has barely started.  Without this the closed form above is satisfiable by
    # a layer that applied state 0's rate to everything.
    survival = (final.abs().sum(dim=(-1, -2)) / memory.abs().sum(dim=(-1, -2)))[0, 0]
    assert survival[0] > 0.5, "the slowest state should still remember"
    assert survival[-1] < 1e-20, "the fastest state should have forgotten"


def test_the_triangular_solve_stays_shared_under_per_state_decay(monkeypatch):
    """The rectangle rule's `O(C^3)` paid `G_k` times, not `H` times (§3.1).

    Per-state decay had an obvious implementation that pays it `H` times — fold
    each state's `rel` in and solve `m` times.  This module takes the other route
    precisely to avoid that, so the cost claim is worth holding to structurally
    rather than in a benchmark that nobody runs.  The solve's batch must not
    widen with the state axis.
    """
    from lumen.gdn import reference

    seen: list[tuple[int, ...]] = []
    original = reference.inv_unit

    def recording(matrix):
        seen.append(tuple(matrix.shape))
        return original(matrix)

    monkeypatch.setattr(reference, "inv_unit", recording)

    torch.manual_seed(0)
    layout = LAYOUTS["shared_key"]  # G_k = 1, m = 8: the worst case for this
    for decay in ("shared", "per_state"):
        seen.clear()
        q, k, v, beta, log_alpha = make_inputs(layout, seq_len=64, decay=decay)
        reference.chunk_gated_delta(q, k, v, beta, log_alpha, 16)

        assert len(seen) == 1, "more than one solve per call"
        assert seen[0][2] == 1, (
            f"{decay}: the solve's state axis is {seen[0][2]}, not 1 — the "
            f"inverse is being built per state and the rectangle rule is gone"
        )


def test_the_kernel_refuses_a_wide_log_alpha_it_would_mishandle():
    """A wrong answer that does not raise is the worst failure mode here.

    Before the state axis was plumbed through, a wide `log_alpha` flowed into
    several `[:, :, 0]` slices that applied **state 0's decay to every state**
    and returned a plausible tensor of the right shape.  The seam was documented
    as open, so acting on it was exactly what a reader was invited to do.
    """
    torch.manual_seed(0)
    layout = LAYOUTS["shared_key"]
    q, k, v, beta, log_alpha = make_inputs(layout, seq_len=32)

    with pytest.raises(ValueError, match="log_alpha's state axis"):
        chunk_gated_delta(q, k, v, beta, log_alpha.expand(-1, -1, 3, -1), 16)

    with pytest.raises(ValueError, match="beta must be per key group"):
        chunk_gated_delta(
            q, k, v, beta.expand(-1, -1, layout.states_per_key_group, -1), log_alpha, 16
        )


def test_beta_max_two_is_the_stability_boundary_of_the_undamped_solve():
    """Why solve-then-damp is safe, held to as a property rather than a comment.

    Forward substitution on `I + N` is the delta rule: `Z` advances by
    `(I - beta k k^T)`, whose spectral norm is `max(1, |1 - beta|)` — exactly 1
    for every beta in [0, 2].  So the undamped inverse is bounded independent of
    the chunk size, which is what lets the decay be applied *after* the solve and
    the solve stay shared across states.

    Past 2 the operator expands and the growth is geometric in `C`.  That is not
    only a per-state concern: `rel -> 1` as `alpha -> 1`, so slow forgetting
    leaves the shared path's solve undamped too.
    """
    def peak(chunk: int, beta_max: float) -> float:
        torch.manual_seed(0)
        base = F.normalize(torch.randn(8, dtype=torch.float64), dim=-1)
        keys = F.normalize(
            base + 0.01 * torch.randn(chunk, 8, dtype=torch.float64), dim=-1
        )
        matrix = (beta_max * (keys @ keys.T)).tril(-1)
        return inv_unit(matrix).abs().amax().item()

    # Bounded, and flat in the chunk size, right up to the ceiling.
    for beta_max in (0.5, 1.0, 1.9, 2.0):
        peaks = [peak(chunk, beta_max) for chunk in (16, 64, 256)]
        assert max(peaks) < 2.0, f"beta_max={beta_max} is not bounded by 2"
        assert max(peaks) - min(peaks) < 0.01, (
            f"beta_max={beta_max} grows with the chunk size: {peaks}"
        )

    # And past it, geometric -- so the ceiling is load-bearing, not decorative.
    assert peak(256, 2.5) > 1e20


def test_assign_values_is_a_view_for_named_layouts():
    """The 'costs no copy' claim, checked against storage rather than asserted."""
    for name, layout in LAYOUTS.items():
        values = torch.randn(2, 16, layout.n_value_groups, 12)
        assigned = assign_values(values, layout)
        assert assigned.shape == (
            2, layout.n_key_groups, layout.states_per_key_group, 16, 12
        )
        assert assigned.data_ptr() == values.data_ptr(), name


def test_assign_values_broadcast_survives_chunk_reshape():
    """A crossed layout must not silently materialise G_k copies of the values.

    The kernel splits the time axis; if that forced a copy, the whole
    cheapness argument for sharing a key across states would be wrong.
    """
    layout = HeadLayout.crossed(4, 4)
    values = torch.randn(2, 128, 4, 12)
    assigned = assign_values(values, layout)
    chunked = assigned.reshape(2, 4, 4, 8, 16, 12)
    assert chunked.data_ptr() == values.data_ptr()
    assert chunked.untyped_storage().nbytes() < chunked.numel() * 4


def test_assign_values_places_the_right_value_in_each_state():
    """Indexing, checked by identity rather than by shape."""
    layout = HeadLayout.crossed(2, 3)
    values = torch.arange(3.0).view(1, 1, 3, 1).expand(1, 5, 3, 4).contiguous()
    assigned = assign_values(values, layout)
    for head, (key_group, value_group) in enumerate(
        zip(layout.key_group, layout.value_group)
    ):
        i, j = key_group, head % layout.states_per_key_group
        assert torch.equal(
            assigned[0, i, j, 0], torch.full((4,), float(value_group))
        )


# ── the layer ─────────────────────────────────────────────────────────────


def make_layer(layout=None, **overrides):
    layout = layout or LAYOUTS["shared_key"]
    config = GatedDeltaNetConfig(
        **{
            "d_model": 64,
            "layout": layout,
            "expand_k": 2.0,
            "expand_v": 1.0,
            "chunk_size": 16,
            **overrides,
        }
    )
    return GatedDeltaNet(config).double()


@pytest.mark.parametrize("name", list(LAYOUTS))
def test_layer_forward_shape_and_finiteness(name):
    torch.manual_seed(0)
    layout = LAYOUTS[name]
    d_model = 64 if layout.n_heads > 1 else 8
    layer = make_layer(layout, d_model=d_model)
    x = torch.randn(2, 32, d_model, dtype=torch.float64)
    y = layer(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_config_rejects_bad_values():
    with pytest.raises(ValueError, match="power of two"):
        GatedDeltaNetConfig(d_model=64, layout=HeadLayout.shared_key(8), chunk_size=24)
    with pytest.raises(ValueError, match="divide evenly"):
        GatedDeltaNetConfig(d_model=12, layout=HeadLayout.shared_key(8), expand_k=1.0)
    with pytest.raises(ValueError, match="expand_k must be > 0"):
        GatedDeltaNetConfig(d_model=64, layout=HeadLayout.shared_key(8), expand_k=0.0)
    with pytest.raises(ValueError, match="dropout"):
        GatedDeltaNetConfig(d_model=64, layout=HeadLayout.shared_key(8), dropout=1.0)
    with pytest.raises(TypeError, match="layout must be a HeadLayout"):
        GatedDeltaNetConfig(d_model=64, layout=8)


def test_the_common_case_is_one_call():
    config = GatedDeltaNetConfig(d_model=512, layout=HeadLayout.shared_key(8))
    assert config.d_k == 128 and config.d_v == 64
    assert GatedDeltaNet(config)(torch.randn(1, 8, 512)).shape == (1, 8, 512)


def test_n_heads_is_derived_and_cannot_disagree_with_the_layout():
    """It used to be a second field cross-checked into agreement (0.2 §6.4).

    Deriving makes disagreement *unrepresentable* rather than *detected*, which
    is the stronger guarantee -- there is no state in which the two are both
    present and wrong, so there is nothing left to validate.  Assigning to it
    fails because a property has no setter, which is the same fact from the
    other direction.
    """
    config = GatedDeltaNetConfig(d_model=64, layout=HeadLayout.crossed(2, 4))
    assert config.n_heads == 8 == config.layout.n_heads

    with pytest.raises(AttributeError):
        config.n_heads = 4  # type: ignore[misc]

    # And the head count follows the layout wherever it goes.
    for layout in LAYOUTS.values():
        assert GatedDeltaNetConfig(d_model=64, layout=layout).n_heads == layout.n_heads


def test_layout_is_required_because_a_head_count_does_not_imply_one():
    """`n_heads=8` used to mean `shared_key(8)` silently.  That was a decision.

    The layout module's whole argument is that "8 heads" does not determine an
    arrangement -- diagonal, shared-key and crossed(2,4) are all eight heads and
    all different models.  A default picked one of them without saying so.
    """
    with pytest.raises(TypeError):
        GatedDeltaNetConfig(d_model=64)  # type: ignore[call-arg]

    eight = [HeadLayout.shared_key(8), HeadLayout.diagonal(8), HeadLayout.crossed(2, 4)]
    assert all(layout.n_heads == 8 for layout in eight)
    assert len({layout.describe() for layout in eight}) == 3, "same count, three models"


def test_layer_backward_reaches_every_parameter():
    torch.manual_seed(0)
    layer = make_layer(centre=True)
    x = torch.randn(2, 32, 64, dtype=torch.float64, requires_grad=True)
    layer(x).square().mean().backward()

    missing = [n for n, p in layer.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached: {missing}"
    assert torch.isfinite(x.grad).all()


# ── streaming ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["shared_key", "diagonal", "crossed_2x4"])
def test_step_matches_forward(name):
    """Acceptance §6.2 — decode one position at a time == one parallel pass."""
    torch.manual_seed(0)
    layer = make_layer(LAYOUTS[name])
    x = torch.randn(2, 32, 64, dtype=torch.float64)

    parallel = layer(x)

    state = layer.init_state(2, dtype=torch.float64)
    stepped = []
    for t in range(x.shape[1]):
        y, state = layer.step(x[:, t:t + 1], state)
        stepped.append(y)

    assert (torch.cat(stepped, dim=1) - parallel).abs().max() < EXACT


def test_prefill_then_step_matches_one_pass():
    torch.manual_seed(0)
    layer = make_layer()
    x = torch.randn(2, 48, 64, dtype=torch.float64)

    whole = layer(x)

    prefill, state = layer(x[:, :32], return_state=True)
    rest = []
    for t in range(32, 48):
        y, state = layer.step(x[:, t:t + 1], state)
        rest.append(y)

    assert (torch.cat([prefill, *rest], dim=1) - whole).abs().max() < EXACT


@pytest.mark.parametrize("split", [16, 32, 48])
def test_chunked_forward_matches_one_pass(split):
    torch.manual_seed(0)
    layer = make_layer()
    x = torch.randn(2, 64, 64, dtype=torch.float64)

    whole = layer(x)
    first, state = layer(x[:, :split], return_state=True)
    second = layer(x[:, split:], state=state)

    assert (torch.cat([first, second], dim=1) - whole).abs().max() < EXACT


def test_conv_cache_is_load_bearing():
    """Drop the short-conv cache on resume and the seam must visibly break.

    Otherwise this test would be proving nothing about a state field that is
    easy to leave out and whose absence corrupts only a few positions.
    """
    torch.manual_seed(0)
    layer = make_layer(conv_size=4)
    x = torch.randn(2, 64, 64, dtype=torch.float64)

    whole = layer(x)
    _, state = layer(x[:, :32], return_state=True)

    with_cache = layer(x[:, 32:], state=state)
    without = layer(x[:, 32:], state=GatedDeltaNetState(memory=state.memory, conv=None))

    assert (with_cache - whole[:, 32:]).abs().max() < EXACT
    assert (without - whole[:, 32:]).abs().max() > 1e-6


def test_state_is_frozen_and_step_returns_a_successor():
    """Branching a stream must not have two branches sharing a buffer."""
    torch.manual_seed(0)
    layer = make_layer()
    x = torch.randn(2, 1, 64, dtype=torch.float64)

    state = layer.init_state(2, dtype=torch.float64)
    before = state.memory.clone()

    _, branch_a = layer.step(x, state)
    _, branch_b = layer.step(x * 2.0, state)

    assert torch.equal(state.memory, before), "step mutated the state it was given"
    assert not torch.equal(branch_a.memory, branch_b.memory)
    with pytest.raises(Exception):
        state.memory = branch_a.memory  # type: ignore[misc]


def test_memory_is_flat_in_generated_length():
    """Acceptance §6.4 — the point of a fixed-size memory."""
    torch.manual_seed(0)
    layer = make_layer()
    state = layer.init_state(2, dtype=torch.float64)
    x = torch.randn(2, 1, 64, dtype=torch.float64)

    sizes = []
    for step in range(64):
        _, state = layer.step(x, state)
        if step in (7, 63):
            sizes.append(state.memory.numel() + sum(c.numel() for c in state.conv))
    assert sizes[0] == sizes[1]


# ── claims made "by construction" ─────────────────────────────────────────


def test_zero_init_centring_is_an_exact_identity():
    """Acceptance §6.5.

    The design record says a checkpoint from an uncentred model loads and
    behaves identically *by construction*.  A construction claim is worth a
    test rather than a comment — and bit-identity, not a tolerance, because
    subtracting an exact zero is exact.
    """
    torch.manual_seed(0)
    plain = make_layer(centre=False)
    torch.manual_seed(0)
    centred = make_layer(centre=True)

    # Same init stream: zeros() draws no randomness, so every shared parameter
    # must be identical without any copying.
    plain_params = dict(plain.named_parameters())
    for name, param in centred.named_parameters():
        if name in plain_params:
            assert torch.equal(param, plain_params[name]), name

    x = torch.randn(2, 32, 64, dtype=torch.float64)
    with torch.no_grad():
        assert torch.equal(plain(x), centred(x))


def test_centring_is_live_once_moved():
    """The flag must not be inert -- otherwise the test above proves nothing."""
    torch.manual_seed(0)
    layer = make_layer(centre=True)
    x = torch.randn(2, 32, 64, dtype=torch.float64)

    with torch.no_grad():
        before = layer(x)
        layer.k_centre.normal_(0.0, 0.5)
        after = layer(x)

    assert (before - after).abs().max() > 1e-3


def test_zero_init_per_state_decay_is_the_shared_layer():
    """The construction claim for `decay="state"`, and its honest tolerance.

    `centre` gets `torch.equal` because subtracting an exact zero is exact.
    This one does not, and the difference is worth being precise about rather
    than papering over: the offset is zero so the *decay values* are identical,
    but a per-state `log_alpha` sends the kernel down the solve-then-damp route
    while the shared one damps first.  Same object, opposite routes, so the
    guarantee is "mathematically the same layer, agreeing to the fp64 gate" and
    not bit-identity.  Claiming the stronger one would be false.
    """
    torch.manual_seed(0)
    plain = make_layer(decay="key_group")
    torch.manual_seed(0)
    per_state = make_layer(decay="state")

    plain_params = dict(plain.named_parameters())
    for name, param in per_state.named_parameters():
        if name in plain_params:
            assert torch.equal(param, plain_params[name]), name
    assert torch.equal(per_state.a_offset, torch.zeros_like(per_state.a_offset))

    x = torch.randn(2, 32, 64, dtype=torch.float64)
    with torch.no_grad():
        assert (plain(x) - per_state(x)).abs().max() < EXACT


def test_per_state_decay_is_live_once_the_offset_moves():
    """The flag must not be inert -- otherwise the test above proves nothing."""
    torch.manual_seed(0)
    layer = make_layer(decay="state")
    x = torch.randn(2, 32, 64, dtype=torch.float64)

    with torch.no_grad():
        before = layer(x)
        layer.a_offset.normal_(0.0, 1.0)
        after = layer(x)

    assert (before - after).abs().max() > 1e-3


def test_per_state_decay_gives_the_layer_distinct_horizons():
    """What someone turns this on for: the states must forget at different rates.

    Checked on `log_alpha` itself rather than on the output, because that is
    where the claim lives -- a spread in the outputs could come from anywhere.
    """
    torch.manual_seed(0)
    layer = make_layer(LAYOUTS["shared_key"], decay="state")
    m = layer.config.layout.states_per_key_group

    with torch.no_grad():
        # A geometric band is an init of this parameter, not another code path.
        layer.a_offset.copy_(torch.linspace(-2.0, 2.0, m).view(1, m))

    x = torch.randn(2, 32, 64, dtype=torch.float64)
    _, _, _, _, log_alpha, _ = layer._features(x)

    assert log_alpha.shape[2] == m
    assert log_alpha.max() <= 0.0, "alpha must stay in (0, 1]"

    half_lives = math.log(2) / -log_alpha.mean(dim=(0, 1, 3))
    assert half_lives[0] > 10.0 * half_lives[-1], (
        f"the band is not producing distinct horizons: {half_lives.tolist()}"
    )


def test_per_state_decay_backward_reaches_the_offset():
    torch.manual_seed(0)
    layer = make_layer(decay="state")
    x = torch.randn(2, 32, 64, dtype=torch.float64, requires_grad=True)
    layer(x).square().mean().backward()

    missing = [n for n, p in layer.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached: {missing}"
    assert layer.a_offset.grad.abs().max() > 0.0, "the offset is not being trained"


def test_per_state_decay_costs_one_scalar_per_state():
    layout = LAYOUTS["shared_key"]
    shared, per_state = make_layer(layout), make_layer(layout, decay="state")
    added = sum(p.numel() for p in per_state.parameters()) - sum(
        p.numel() for p in shared.parameters()
    )
    assert added == layout.n_heads, (
        "per-state decay should cost H scalars, not a wider projection"
    )


def test_per_state_decay_streams():
    """The interface commitment holds for both arrangements, not just the old one."""
    torch.manual_seed(0)
    layer = make_layer(LAYOUTS["crossed_2x4"], decay="state")
    with torch.no_grad():
        layer.a_offset.normal_(0.0, 1.0)
    x = torch.randn(2, 32, 64, dtype=torch.float64)

    parallel = layer(x)
    state = layer.init_state(2, dtype=torch.float64)
    stepped = []
    for t in range(x.shape[1]):
        y, state = layer.step(x[:, t:t + 1], state)
        stepped.append(y)

    assert (torch.cat(stepped, dim=1) - parallel).abs().max() < EXACT


def test_decay_arrangements_do_not_silently_mix_checkpoints():
    """Same rule as centring: crossing the boundary must be loud."""
    shared, per_state = make_layer(decay="key_group"), make_layer(decay="state")
    with pytest.raises(RuntimeError, match="[Uu]nexpected key"):
        shared.load_state_dict(per_state.state_dict())
    with pytest.raises(RuntimeError, match="[Mm]issing key"):
        per_state.load_state_dict(shared.state_dict())


@pytest.mark.parametrize("name", ["shared_key", "diagonal", "crossed_2x4", "single"])
def test_state_gated_reaches_the_oracle(name):
    """`state_gated` is a wider projection, not a different recurrence."""
    torch.manual_seed(0)
    layer = make_layer(LAYOUTS[name], d_model=64 if LAYOUTS[name].n_heads > 1 else 8,
                       decay="state_gated")
    with torch.no_grad():
        layer.a_proj.weight.normal_(0.0, 0.5)  # break the shared-gate init
    x = torch.randn(2, 32, layer.config.d_model, dtype=torch.float64)

    q, k, v, beta, log_alpha, _ = layer._features(x)
    assert log_alpha.shape[2] == LAYOUTS[name].states_per_key_group

    out, state = layer._scan(q, k, v, beta, log_alpha, None)
    out_seq, state_seq = sequential_gated_delta(q, k, v, beta, log_alpha)
    assert (out - out_seq).abs().max() < EXACT
    assert (state - state_seq).abs().max() < EXACT


def test_state_gated_starts_in_the_key_group_arrangement():
    """The weaker of the two init guarantees, stated as the weaker one.

    `state` is an EXACT identity with `key_group` — it adds zero to the same
    projection.  `state_gated` cannot be: a wider `nn.Linear` draws a different
    number of values, so its rates are not the rates the narrow layer would have
    had from the same seed.  What it *does* guarantee is the ARRANGEMENT — every
    state in a key group on one gate, so the layer has to earn its way out.
    Asserting the stronger claim here would be false, and asserting nothing
    would let the init silently stop mattering.
    """
    torch.manual_seed(0)
    gated = make_layer(decay="state_gated")
    torch.manual_seed(0)
    shared = make_layer(decay="key_group")
    x = torch.randn(2, 32, 64, dtype=torch.float64)

    *_, log_alpha, _ = gated._features(x)
    assert torch.equal(log_alpha, log_alpha[:, :, :1].expand_as(log_alpha)), (
        "states within a key group do not start on one gate"
    )
    # ...and it is its own layer, not a re-spelling of the narrow one.
    with torch.no_grad():
        assert not torch.allclose(gated(x), shared(x))


def test_apply_init_structure_restores_the_arrangement_after_a_redraw():
    """The guarantee above, as something a later caller can re-establish.

    The test above builds the layer and looks at it, which is the one situation
    where nothing writes to `a_proj` after the constructor does.  A trunk-level
    init does — `lumen#8` — so the arrangement has to be a hook that survives
    being redrawn over, not three lines that ran once.

    `crossed(4, 2)` rather than `shared_key`, so "every row equal" and "equal
    within a key group" are different assertions and only the second passes.
    """
    torch.manual_seed(0)
    layer = make_layer(LAYOUTS["crossed_4x2"], decay="state_gated")
    groups = layer.config.layout.n_key_groups
    rows = layer.a_proj.weight.view(groups, -1, layer.a_proj.in_features)
    assert torch.equal(rows, rows[:, :1].expand_as(rows))
    assert not torch.equal(rows[0, 0], rows[1, 0]), "key groups share one gate"

    with torch.no_grad():
        torch.nn.init.normal_(layer.a_proj.weight, std=0.02)
    rows = layer.a_proj.weight.view(groups, -1, layer.a_proj.in_features)
    assert not torch.equal(rows, rows[:, :1].expand_as(rows)), (
        "probe is invalid: the redraw did not disturb the arrangement"
    )

    layer.apply_init_structure()
    rows = layer.a_proj.weight.view(groups, -1, layer.a_proj.in_features)
    assert torch.equal(rows, rows[:, :1].expand_as(rows))
    assert not torch.equal(rows[0, 0], rows[1, 0])


@pytest.mark.parametrize("decay", ["key_group", "state"])
def test_apply_init_structure_is_a_no_op_for_the_other_arrangements(decay):
    """Absence of structure is the normal case, so the hook has to be safe.

    Neither of these has an arrangement among its weights to lose: `state`'s
    per-state offset is a separate zeroed parameter and `key_group`'s rates are
    one row each.  A caller invokes the hook without asking which `decay` it
    holds, so "no structure" must mean "changes nothing" rather than "does
    something defensible".
    """
    torch.manual_seed(0)
    layer = make_layer(decay=decay)
    before = {name: p.clone() for name, p in layer.named_parameters()}
    layer.apply_init_structure()
    for name, parameter in layer.named_parameters():
        assert torch.equal(before[name], parameter), name


def test_state_gated_is_live_and_independently_modulated():
    """The property that separates it from `state`: each rate moves on its own.

    Under `state` the eight rates share one data-dependent signal, so perturbing
    the input moves them together.  Under `state_gated` they are eight functions
    and need not agree.
    """
    torch.manual_seed(0)
    layer = make_layer(decay="state_gated")
    with torch.no_grad():
        layer.a_proj.weight.normal_(0.0, 1.0)
    x = torch.randn(2, 64, 64, dtype=torch.float64)

    *_, log_alpha, _ = layer._features(x)
    # across time, do the states' rates vary independently rather than in lockstep?
    series = log_alpha[0, 0]                      # (m, T)
    centred = series - series.mean(dim=-1, keepdim=True)
    corr = torch.corrcoef(centred)
    off = corr[~torch.eye(corr.shape[0], dtype=torch.bool)]
    assert off.abs().max() < 0.99, (
        f"state rates move in lockstep (max |corr| {off.abs().max():.3f}); "
        f"that is the `state` arrangement, not `state_gated`"
    )


def test_the_two_per_state_arrangements_cost_different_things():
    """`state` buys baselines for H scalars; `state_gated` buys gates for weights."""
    layout = LAYOUTS["shared_key"]
    base = sum(p.numel() for p in make_layer(layout).parameters())
    band = sum(p.numel() for p in make_layer(layout, decay="state").parameters())
    gated = sum(p.numel() for p in make_layer(layout, decay="state_gated").parameters())

    config = make_layer(layout).config
    extra = layout.n_heads - layout.n_key_groups
    assert band - base == layout.n_heads
    # a wider projection: weights and biases for the states beyond the first
    assert gated - base == extra * (config.d_model + 1)
    assert gated > band, "the gated arrangement should be the expensive one"


def test_state_gated_backward_reaches_every_gate():
    torch.manual_seed(0)
    layer = make_layer(decay="state_gated")
    x = torch.randn(2, 32, 64, dtype=torch.float64, requires_grad=True)
    layer(x).square().mean().backward()

    grad = layer.a_proj.weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    # every state's gate must receive its own signal, not just the first
    assert (grad.abs().sum(dim=-1) > 0).all(), "some state's gate got no gradient"


def test_the_three_decay_arrangements_do_not_mix_checkpoints():
    layers = {name: make_layer(decay=name)
              for name in ("key_group", "state", "state_gated")}
    for a, b in itertools.permutations(layers, 2):
        with pytest.raises(RuntimeError):
            layers[a].load_state_dict(layers[b].state_dict())


def test_config_refuses_a_beta_max_the_solve_cannot_survive():
    """The ceiling is load-bearing, so it is enforced rather than documented.

    `beta_max > 2` makes the delta update expand instead of reflect, and the
    UT/WY inverse grows geometrically in `chunk_size`.  The relative decay hides
    it while forgetting is fast and stops hiding it as `alpha -> 1`, so it is a
    value that works until a model learns to remember.
    """
    with pytest.raises(ValueError, match="beta_max must be <= 2"):
        GatedDeltaNetConfig(d_model=64, layout=HeadLayout.shared_key(8), beta_max=2.5)
    # and the boundary itself is allowed, since that is where reflections live
    assert GatedDeltaNetConfig(
        d_model=64, layout=HeadLayout.shared_key(8), beta_max=2.0
    ).beta_max == 2.0

    with pytest.raises(ValueError, match="decay must be"):
        GatedDeltaNetConfig(d_model=64, layout=HeadLayout.shared_key(8), decay="per_head")


def test_centring_cost_is_two_centres_per_head():
    layout = LAYOUTS["shared_key"]
    plain, centred = make_layer(layout), make_layer(layout, centre=True)
    added = sum(p.numel() for p in centred.parameters()) - sum(
        p.numel() for p in plain.parameters()
    )
    config = centred.config
    assert added == (config.n_heads + layout.n_key_groups) * config.d_k


def test_beta_max_reaches_reflections():
    """beta_max = 2 is what lets 1 - beta go negative (a reflection, not a
    contraction).  Checked on the state operator rather than on the constant."""
    torch.manual_seed(0)
    layer = make_layer()
    x = torch.randn(4, 32, 64, dtype=torch.float64)
    _, _, _, beta, _, _ = layer._features(x)
    assert beta.max() > 1.0, "beta never exceeds 1, so no update is a reflection"
    assert beta.min() > 0.0
    assert beta.max() < layer.config.beta_max


@pytest.mark.parametrize("chunk_size", [8, 16, 32, 64, 128])
def test_chunk_size_does_not_decide_the_shortest_half_life(chunk_size):
    """**The test the decay reformulation exists for.**

    `_features` used to floor `log_alpha` at ``-max_chunk_decay / chunk_size``,
    to keep `1/gamma` in range for a decay absorption that no longer forms
    `1/gamma`.  The side effect was that the shortest expressible half-life was

        chunk_size * ln2 / max_chunk_decay

    — *linear in the chunk size*.  A parameter whose entire job is to tile the
    sequence for the GPU was silently deciding what the layer could represent:
    at the old defaults a head could not forget faster than every 5.5 positions,
    and moving to ``chunk_size = 128`` for performance would have doubled that
    to 11 without anyone choosing it.

    Driving `alpha` low must now produce the same half-life at every chunk size,
    and one below the old floor.  A numerics change with no behavioural test is
    indistinguishable from a refactor; this is the behaviour.
    """
    torch.manual_seed(0)
    layer = make_layer(chunk_size=chunk_size, conv_size=0)
    with torch.no_grad():
        layer.a_proj.weight.zero_()
        layer.a_proj.bias.fill_(2.0)  # softplus(2) ~ 2.13 -> alpha ~ 0.119

    x = torch.randn(1, 8, 64, dtype=torch.float64)
    _, _, _, _, log_alpha, _ = layer._features(x)

    assert log_alpha.max() <= 0.0, "alpha must stay in (0, 1]"
    half_life = math.log(2) / -log_alpha.mean().item()
    old_floor = chunk_size * math.log(2) / 8.0

    assert half_life == pytest.approx(0.3259, abs=1e-3), "must not depend on chunk_size"
    assert half_life < old_floor, (
        f"half-life {half_life:.3f} is not below the old floor {old_floor:.3f}; "
        f"the clamp's removal is not observable and the change is a refactor"
    )


@pytest.mark.parametrize("chunk", [1, 2, 8, 32, 128])
def test_chunk_size_is_inert_on_unclamped_decay(chunk):
    """Gate 3 — outputs agree across chunk sizes where the old form could not.

    Distinct from `test_chunkwise_is_independent_of_chunk_size` only in what the
    fixture now supplies: unclamped decay, accumulating past `-100` inside a
    128-chunk.  Under the `1/gamma` factorisation that range was unreachable by
    construction, which is why this could not have been asserted before.
    """
    torch.manual_seed(0)
    layout = LAYOUTS["shared_key"]
    q, k, v, beta, log_alpha = make_inputs(layout, seq_len=128)
    assert -log_alpha.cumsum(-1)[..., -1].min() > 50.0, "fixture is not exercising the range"

    out, state = chunk_gated_delta(q, k, v, beta, log_alpha, chunk)
    out_seq, state_seq = sequential_gated_delta(q, k, v, beta, log_alpha)

    assert (out - out_seq).abs().max() < EXACT
    assert (state - state_seq).abs().max() < EXACT


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_no_overflow_at_extreme_decay(dtype):
    """Gate 4 — `-sum(log alpha) = 200` in one chunk, well past fp32's ceiling.

    fp32 `exp` overflows at 88.7, so the old `inv_gamma = (-cumulative).exp()`
    produced `inf` here, then `inf * 0 = nan`, and poisoned the batch.  The
    relative form's only failure mode is underflow to zero, and underflow to
    zero is the right answer: it means that position has fully decayed.
    """
    torch.manual_seed(0)
    layout = LAYOUTS["shared_key"]
    q, k, v, beta, _ = make_inputs(layout, seq_len=128, dtype=dtype)
    log_alpha = torch.full_like(beta, -200.0 / 128)

    out, state = chunk_gated_delta(q, k, v, beta, log_alpha, 128)

    assert torch.isfinite(out).all(), "extreme decay must not produce inf or nan"
    assert torch.isfinite(state).all()

    out_seq, _ = sequential_gated_delta(q, k, v, beta, log_alpha)
    assert (out - out_seq).abs().max() < (EXACT if dtype is torch.float64 else FP32)


def test_decode_conv_matches_the_library_call():
    """The T=1 short-conv path is a hand-rolled sum; hold it to `conv1d`.

    It exists because grouped-convolution dispatch dominates a decode step, and
    it is confined to decode so that training numerics stay bit-identical.  It
    is *not* bit-identical itself -- a different accumulation order -- so the
    bound here is round-off, and the point of the test is that it stays there.
    """
    torch.manual_seed(0)
    conv = ShortConv(64, 4).double()
    x = torch.randn(2, 1, 64, dtype=torch.float64)
    cache = torch.randn(2, 64, 3, dtype=torch.float64)

    fast, cache_fast = conv(x, cache)

    u = torch.cat([cache, x.transpose(1, 2)], dim=-1)
    reference = F.conv1d(u, conv.conv.weight, groups=64).transpose(1, 2)

    assert (fast - reference).abs().max() < EXACT
    assert torch.equal(cache_fast, u[..., -3:])


def test_short_conv_is_causal():
    """It carries relative offsets, and it must not carry the future."""
    torch.manual_seed(0)
    layer = make_layer(conv_size=4)
    x = torch.randn(1, 32, 64, dtype=torch.float64)
    perturbed = x.clone()
    perturbed[:, 20:] += 10.0

    with torch.no_grad():
        assert torch.allclose(layer(x)[:, :20], layer(perturbed)[:, :20], atol=EXACT)


# ── depending on it ───────────────────────────────────────────────────────
#
# Not properties of the algorithm -- properties of the component, which is what
# a project taking a dependency on it actually needs to hold.


@pytest.mark.parametrize("seq_len", [1, 7, 15, 16, 17, 33, 100])
def test_any_sequence_length_works_and_is_exact(seq_len):
    """A corpus is whatever length it is.  Padding must be exact, not close.

    Padded positions carry beta=0 and log_alpha=0, so the recurrence there is
    the identity and they sit behind the causal mask.  Checked against the
    oracle rather than against "it did not raise".
    """
    torch.manual_seed(0)
    layout = LAYOUTS["shared_key"]
    q, k, v, beta, log_alpha = make_inputs(layout, seq_len=seq_len)

    out, state = chunk_gated_delta(q, k, v, beta, log_alpha, 16)
    out_seq, state_seq = sequential_gated_delta(q, k, v, beta, log_alpha)

    assert out.shape[-2] == seq_len
    assert (out - out_seq).abs().max() < EXACT
    assert (state - state_seq).abs().max() < EXACT


def test_layer_accepts_a_ragged_sequence():
    torch.manual_seed(0)
    layer = make_layer(chunk_size=16)
    x = torch.randn(2, 37, 64, dtype=torch.float64)
    y, state = layer(x, return_state=True)
    assert y.shape == x.shape

    # and the state is genuinely the state after position 37, not after 48
    continued = layer(torch.randn(2, 5, 64, dtype=torch.float64), state=state)
    assert torch.isfinite(continued).all()


def test_checkpoint_round_trips():
    """Save, build fresh from the same config, load, and get the same layer."""
    torch.manual_seed(0)
    layer = make_layer(centre=True)
    x = torch.randn(2, 32, 64, dtype=torch.float64)

    fresh = GatedDeltaNet(layer.config).double()
    fresh.load_state_dict(layer.state_dict())

    with torch.no_grad():
        assert torch.equal(layer(x), fresh(x))


def test_centred_and_uncentred_checkpoints_do_not_silently_mix():
    """Loading across the centring boundary must be explicit, not quiet."""
    centred, plain = make_layer(centre=True), make_layer(centre=False)
    with pytest.raises(RuntimeError, match="[Uu]nexpected key"):
        plain.load_state_dict(centred.state_dict())


def test_init_state_follows_the_module():
    """`layer.to(...).init_state(batch)` must not need a second argument."""
    layer = make_layer()  # .double()
    state = layer.init_state(2)
    assert state.memory.dtype == torch.float64
    assert state.memory.device == next(layer.parameters()).device
    assert all(c.dtype == torch.float64 for c in state.conv)


def test_it_is_interchangeable_with_undertow():
    """The interface commitment: a block can hold either mixer.

    This is the reason the output side is shaped the way it is, so it is worth
    a test rather than a docstring promise.
    """
    from lumen.undertow import UndertowAttention, UndertowConfig

    x = torch.randn(2, 32, 64)
    mixers = [
        GatedDeltaNet(GatedDeltaNetConfig(d_model=64, layout=HeadLayout.shared_key(8), chunk_size=16)),
        UndertowAttention(UndertowConfig(d_model=64, n_heads=8, window=8)),
    ]
    for mixer in mixers:
        assert mixer(x).shape == x.shape
        state = mixer.init_state(2)
        y, state = mixer.step(x[:, :1], state)
        assert y.shape == (2, 1, 64)
        y, state = mixer(x[:, :8], state=state, return_state=True)
        assert y.shape == (2, 8, 64)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_layer_runs_on_cuda_end_to_end():
    layer = GatedDeltaNet(
        GatedDeltaNetConfig(d_model=64, layout=HeadLayout.shared_key(8), chunk_size=16)
    ).cuda()
    x = torch.randn(2, 100, 64, device="cuda")
    y, state = layer(x, return_state=True)
    assert y.shape == x.shape and state.memory.is_cuda
    y, state = layer.step(x[:, :1], state)
    assert y.is_cuda


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_chunkwise_matches_sequential_on_cuda():
    torch.manual_seed(0)
    layout = LAYOUTS["shared_key"]
    q, k, v, beta, log_alpha = make_inputs(layout)
    device = torch.device("cuda")
    q, k, v, beta, log_alpha = (t.to(device) for t in (q, k, v, beta, log_alpha))

    out_chunk, _ = chunk_gated_delta(q, k, v, beta, log_alpha, 32)
    out_seq, _ = sequential_gated_delta(q, k, v, beta, log_alpha)
    assert (out_chunk - out_seq).abs().max().item() < EXACT
