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


def make_inputs(layout, seq_len=128, chunk=32, batch=2, d_k=16, d_v=12, dtype=torch.float64):
    """Kernel-shaped inputs obeying the layer's own invariants."""
    g_k = layout.n_key_groups
    m = layout.states_per_key_group
    q = F.normalize(torch.randn(batch, g_k, m, seq_len, d_k, dtype=dtype), dim=-1)
    k = F.normalize(torch.randn(batch, g_k, 1, seq_len, d_k, dtype=dtype), dim=-1)
    v = assign_values(
        torch.randn(batch, seq_len, layout.n_value_groups, d_v, dtype=dtype), layout
    )
    beta = 2.0 * torch.sigmoid(torch.randn(batch, g_k, 1, seq_len, dtype=dtype))
    log_alpha = -F.softplus(
        torch.randn(batch, g_k, 1, seq_len, dtype=dtype)
    ).clamp(min=-8.0 / chunk)
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
    q, k, v, beta, log_alpha = make_inputs(layout, seq_len=128, chunk=128)

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
    q, k, v, beta, log_alpha = make_inputs(layout, seq_len=64, chunk=16)
    beta = torch.zeros_like(beta)

    memory = torch.randn(2, 1, 8, 16, 12, dtype=torch.float64)
    _, final = chunk_gated_delta(q, k, v, beta, log_alpha, 16, memory)

    expected = memory * log_alpha.sum(-1)[..., None, None].exp().transpose(2, 2)
    assert (final - expected).abs().max() < EXACT


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
            "n_heads": layout.n_heads,
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
        GatedDeltaNetConfig(d_model=64, n_heads=8, chunk_size=24)
    with pytest.raises(ValueError, match="divide evenly"):
        GatedDeltaNetConfig(d_model=12, n_heads=8, expand_k=1.0)
    with pytest.raises(ValueError, match="expand_k must be > 0"):
        GatedDeltaNetConfig(d_model=64, n_heads=8, expand_k=0.0)
    with pytest.raises(ValueError, match="dropout"):
        GatedDeltaNetConfig(d_model=64, n_heads=8, dropout=1.0)
    with pytest.raises(ValueError, match="n_heads must be >= 1"):
        GatedDeltaNetConfig(d_model=64, n_heads=0)


def test_the_common_case_is_one_call():
    """A library people depend on should not need a layout object to get going."""
    config = GatedDeltaNetConfig(d_model=512, n_heads=8)
    assert config.layout == HeadLayout.shared_key(8)
    assert config.d_k == 128 and config.d_v == 64
    assert GatedDeltaNet(config)(torch.randn(1, 8, 512)).shape == (1, 8, 512)


def test_layout_and_n_heads_are_cross_checked():
    """Over-determined on purpose: disagreement is an error, not a silent model."""
    ok = GatedDeltaNetConfig(d_model=64, n_heads=8, layout=HeadLayout.crossed(2, 4))
    assert ok.layout.n_heads == 8
    with pytest.raises(ValueError, match="pass one or the other"):
        GatedDeltaNetConfig(d_model=64, n_heads=8, layout=HeadLayout.crossed(2, 2))


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


def test_decay_is_clamped_within_a_chunk():
    """max_chunk_decay bounds 1/gamma, which the decay absorption divides by."""
    torch.manual_seed(0)
    layer = make_layer(max_chunk_decay=4.0, chunk_size=16)
    x = 50.0 * torch.randn(2, 32, 64, dtype=torch.float64)
    _, _, _, _, log_alpha, _ = layer._features(x)

    assert log_alpha.max() <= 0.0
    worst = -log_alpha.reshape(2, -1, 2, 16).sum(-1).max()
    assert worst <= 4.0 + 1e-12


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
    q, k, v, beta, log_alpha = make_inputs(layout, seq_len=seq_len, chunk=16)

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
        GatedDeltaNet(GatedDeltaNetConfig(d_model=64, n_heads=8, chunk_size=16)),
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
        GatedDeltaNetConfig(d_model=64, n_heads=8, chunk_size=16)
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
