"""Undertow correctness tests.

The load-bearing ones are :func:`test_windowed_matches_dense_everywhere` and
:func:`test_profile_placement_equivalence`.  The first says the fast path
computes what the obvious path computes; the second says a claim the design
record makes about the layer is actually true of the code.

Everything here runs on CPU in fp32.  There is no GPU in CI by design, so
anything that cannot be checked on a laptop is not checked automatically —
which is a reason to keep the reference path exact rather than a reason to
lower the bar.
"""

from __future__ import annotations

import math

import pytest
import torch

from lumen.undertow import UndertowAttention, UndertowConfig
from lumen.undertow.reference import (
    decay_profile,
    dense_attention,
    log_decay_profile,
    window_validity,
    windowed_aggregate,
    windowed_attention,
    windowed_scores,
)

TOL = 1e-6


def _qkv(
    batch: int = 2, heads: int = 4, seq_len: int = 24, dim: int = 16, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    shape = (batch, heads, seq_len, dim)
    return tuple(  # type: ignore[return-value]
        torch.randn(shape, generator=generator, dtype=torch.float32) for _ in range(3)
    )


# ── config validation ──────────────────────────────────────────────────────


def test_config_rejects_indivisible_heads() -> None:
    with pytest.raises(ValueError, match="divide evenly"):
        UndertowConfig(d_model=10, n_heads=4, window=8)


def test_config_rejects_plateau_at_or_past_window() -> None:
    with pytest.raises(ValueError, match="plateau"):
        UndertowConfig(d_model=16, n_heads=4, window=8, plateau=8)
    with pytest.raises(ValueError, match="plateau"):
        UndertowConfig(d_model=16, n_heads=4, window=8, plateau=-1)


def test_config_rejects_degenerate_window() -> None:
    with pytest.raises(ValueError, match="window"):
        UndertowConfig(d_model=16, n_heads=4, window=0)


def test_config_is_frozen() -> None:
    config = UndertowConfig(d_model=16, n_heads=4, window=8)
    with pytest.raises(Exception):
        config.window = 16  # type: ignore[misc]


def test_plateau_zero_is_legal() -> None:
    """plateau=0 means ramp from the very first step — degenerate but valid."""
    config = UndertowConfig(d_model=16, n_heads=4, window=8, plateau=0)
    assert config.plateau == 0


# ── the profile ────────────────────────────────────────────────────────────


def test_profile_is_strictly_positive() -> None:
    """log() has to be finite everywhere — the equivalence argument needs it."""
    for window, plateau in [(8, 6), (32, 24), (4, 0), (64, 1)]:
        profile = decay_profile(window, plateau)
        assert (profile > 0).all(), f"non-positive entry at W={window}, P={plateau}"
        assert torch.isfinite(log_decay_profile(window, plateau)).all()


def test_profile_self_position_is_full_strength() -> None:
    """Slot W-1 is distance 0 and must never be attenuated."""
    for window, plateau in [(8, 6), (32, 24), (4, 0)]:
        assert decay_profile(window, plateau)[-1].item() == pytest.approx(1.0)


def test_profile_is_non_decreasing_toward_self() -> None:
    profile = decay_profile(32, 24)
    assert torch.all(profile[1:] >= profile[:-1] - 1e-12)


def test_profile_plateau_region_is_exactly_one() -> None:
    window, plateau = 32, 24
    profile = decay_profile(window, plateau)
    # Distances 0 .. plateau-1 occupy the last `plateau` slots.
    assert torch.allclose(profile[-plateau:], torch.ones(plateau), atol=1e-12)
    # Beyond the plateau the ramp is strictly below full strength.  Note the
    # boundary itself is excluded — see the continuity test below.
    assert (profile[: window - plateau - 1] < 1.0).all()


def test_profile_is_continuous_at_the_plateau_boundary() -> None:
    """The ramp *starts* at full strength; there is no step at the boundary.

    At δ = plateau the cosine argument is zero, so p = ½(1+cos 0) = 1 and the
    profile meets the plateau exactly.  Worth pinning: a ramp that began at
    some value below 1 would put a discontinuity in the attention bias at a
    distance the model crosses constantly.
    """
    window, plateau = 32, 24
    profile = decay_profile(window, plateau)
    boundary = window - plateau - 1  # the slot holding distance == plateau
    assert profile[boundary].item() == pytest.approx(1.0, abs=1e-12)
    assert profile[boundary - 1].item() < 1.0


def test_hard_window_profile_is_all_ones() -> None:
    profile = decay_profile(16, None)
    assert torch.equal(profile, torch.ones(16))
    assert torch.equal(log_decay_profile(16, None), torch.zeros(16))


# ── window bookkeeping ─────────────────────────────────────────────────────


def test_validity_masks_exactly_the_prefix() -> None:
    seq_len, window = 6, 4
    valid = window_validity(seq_len, window)
    assert valid.shape == (seq_len, window)
    # Query 0 reaches only itself; query window-1 onward is fully covered.
    assert valid[0].tolist() == [False, False, False, True]
    assert valid[window - 1 :].all()
    # Row t has min(t+1, window) valid slots.
    for t in range(seq_len):
        assert valid[t].sum().item() == min(t + 1, window)


def test_windowed_scores_shape_and_alignment() -> None:
    q, k, _ = _qkv(seq_len=12, dim=8)
    window = 5
    scores = windowed_scores(q, k, window, scale=math.sqrt(8))
    assert scores.shape == (2, 4, 12, window)

    # Slot W-1 must be the self-score; check against an explicit dot product.
    expected = (q[0, 0, 7] * k[0, 0, 7]).sum() / math.sqrt(8)
    assert scores[0, 0, 7, window - 1].item() == pytest.approx(expected.item(), abs=1e-5)

    # Slot 0 of query 7 is key 7-(W-1) = 3.
    expected_far = (q[0, 0, 7] * k[0, 0, 3]).sum() / math.sqrt(8)
    assert scores[0, 0, 7, 0].item() == pytest.approx(expected_far.item(), abs=1e-5)


def test_aggregate_inverts_scores_indexing() -> None:
    """A one-hot weight on slot w must select key t-(W-1)+w."""
    _, _, v = _qkv(seq_len=10, dim=8)
    window = 4
    weights = torch.zeros(2, 4, 10, window)
    weights[..., window - 1] = 1.0  # select self
    assert torch.allclose(windowed_aggregate(weights, v, window), v, atol=1e-12)


# ── the load-bearing equivalence: windowed == dense, everywhere ────────────


@pytest.mark.parametrize(
    "seq_len,window,plateau",
    [
        (24, 8, 6),  # the ordinary case
        (24, 8, None),  # hard window, no ramp
        (24, 1, None),  # degenerate: attends only to self
        (24, 24, 18),  # window spans the whole sequence
        (5, 8, 6),  # T < window: clamped
        (32, 32, 0),  # ramp from the first step
        (16, 3, 2),  # tiny window
    ],
)
def test_windowed_matches_dense_everywhere(
    seq_len: int, window: int, plateau: int | None
) -> None:
    """The O(T·W) path reproduces the O(T²) oracle at *every* position.

    Including the partial-window prefix.  That region is only comparable
    because absent slots are masked to -inf rather than zero-padded; with
    padding it has to be excluded, and an excluded region is somewhere a bug
    can live indefinitely.
    """
    q, k, v = _qkv(seq_len=seq_len, dim=16, seed=7)
    effective = min(window, seq_len)

    fast = windowed_attention(q, k, v, effective, plateau)
    slow = dense_attention(q, k, v, window, plateau)

    diff = (fast - slow).abs().max().item()
    assert diff < TOL, f"windowed vs dense max diff {diff:.3e} (tol {TOL:.0e})"


def test_prefix_is_not_silently_agreeing() -> None:
    """Guard the guard: the prefix rows must actually differ from the interior.

    If the prefix were trivially equal for some structural reason, the test
    above would be passing for free there.  It is not.
    """
    q, k, v = _qkv(seq_len=16, dim=16, seed=3)
    out = windowed_attention(q, k, v, 8, 6)
    assert not torch.allclose(out[:, :, 0], out[:, :, 8], atol=1e-3)


def test_padding_would_have_been_wrong() -> None:
    """Characterise the bug the -inf mask prevents.

    Zero-padding leaves absent slots scoring exp(0)=1, so they take real mass.
    This reproduces that and asserts it disagrees with the oracle — so if the
    masking is ever removed, a test fails rather than the numbers quietly
    drifting.
    """
    q, k, v = _qkv(seq_len=16, dim=16, seed=11)
    window, plateau = 8, 6

    scores = windowed_scores(q, k, window, scale=math.sqrt(16))
    scores = scores + log_decay_profile(window, plateau)
    padded = windowed_aggregate(torch.softmax(scores, dim=-1), v, window)

    oracle = dense_attention(q, k, v, window, plateau)
    prefix_diff = (padded - oracle)[:, :, : window - 1].abs().max().item()
    interior_diff = (padded - oracle)[:, :, window - 1 :].abs().max().item()

    assert prefix_diff > 1e-3, "unmasked padding should corrupt the prefix"
    assert interior_diff < TOL, "and should leave the interior untouched"


# ── the design record's claim about profile placement ──────────────────────


def test_profile_placement_equivalence() -> None:
    """Design record §3.1: where the profile enters does not change the layer.

    ``softmax(s + log p)`` and ``softmax(s) · p`` differ by a positive scalar
    per (batch, head, position), which the per-head RMSNorm cancels.  This
    holds the code to that claim, since it is the reason the layer ships one
    form and no switch.
    """
    torch.manual_seed(19)
    config = UndertowConfig(d_model=64, n_heads=8, window=8, plateau=6)
    layer = UndertowAttention(config).eval()
    x = torch.randn(2, 20, 64)

    with torch.no_grad():
        shipped = layer(x)

        # The other placement, built from the same parts.
        q, k, v, gate = layer._project(x)
        q, k, v = q.float(), k.float(), v.float()
        scores = layer._window_scores(q, k, config.window)
        valid = window_validity(x.shape[1], config.window)
        scores = scores.masked_fill(~valid, float("-inf"))
        weights = torch.softmax(scores, dim=-1) * decay_profile(
            config.window, config.plateau
        )
        alternative = layer._out(
            layer._window_aggregate(weights, v, config.window), gate
        )

    diff = (shipped - alternative).abs().max().item()
    assert diff < 1e-5, f"placements diverged by {diff:.3e} — §3.1 is violated"


def test_placement_equivalence_needs_the_head_norm() -> None:
    """The equivalence is conditional, and the condition is the normalisation.

    Compared *before* ``_out``, the two placements differ — which is what makes
    the test above a statement about the layer rather than about the softmax.
    """
    torch.manual_seed(23)
    config = UndertowConfig(d_model=64, n_heads=8, window=8, plateau=6)
    layer = UndertowAttention(config).eval()
    x = torch.randn(2, 20, 64)

    with torch.no_grad():
        q, k, v, _ = layer._project(x)
        q, k, v = q.float(), k.float(), v.float()
        log_bias = layer._window_aggregate(
            layer._window_weights(q, k, config.window), v, config.window
        )
        scores = layer._window_scores(q, k, config.window)
        scores = scores.masked_fill(
            ~window_validity(x.shape[1], config.window), float("-inf")
        )
        post = layer._window_aggregate(
            torch.softmax(scores, dim=-1)
            * decay_profile(config.window, config.plateau),
            v,
            config.window,
        )

    assert not torch.allclose(log_bias, post, atol=1e-4)


# ── the layer ──────────────────────────────────────────────────────────────


def test_forward_shape_and_finiteness() -> None:
    config = UndertowConfig(d_model=64, n_heads=8, window=8, plateau=6)
    layer = UndertowAttention(config).eval()
    x = torch.randn(3, 17, 64)
    y = layer(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_short_sequence_clamps_rather_than_raising() -> None:
    """T < window is a short batch, not a user error."""
    config = UndertowConfig(d_model=32, n_heads=4, window=16, plateau=12)
    layer = UndertowAttention(config).eval()
    for seq_len in (1, 2, 15):
        y = layer(torch.randn(2, seq_len, 32))
        assert y.shape == (2, seq_len, 32)
        assert torch.isfinite(y).all()


def test_zero_init_is_an_exact_identity_noop() -> None:
    """A spliced layer must contribute exactly nothing at step 0."""
    config = UndertowConfig(d_model=32, n_heads=4, window=8, zero_init=True)
    layer = UndertowAttention(config).eval()
    y = layer(torch.randn(2, 12, 32))
    assert torch.equal(y, torch.zeros_like(y))


def test_causality_no_leak_from_the_future() -> None:
    """Perturbing position t must not change any output before t."""
    torch.manual_seed(31)
    config = UndertowConfig(d_model=32, n_heads=4, window=6, plateau=4)
    layer = UndertowAttention(config).eval()

    x = torch.randn(1, 14, 32)
    perturbed = x.clone()
    perturbed[0, 9] += 5.0

    with torch.no_grad():
        before, after = layer(x), layer(perturbed)

    assert torch.allclose(before[:, :9], after[:, :9], atol=1e-6)
    assert not torch.allclose(before[:, 9], after[:, 9], atol=1e-4)


def test_window_bounds_the_reach() -> None:
    """A perturbation must not reach further back than the window allows."""
    torch.manual_seed(37)
    config = UndertowConfig(d_model=32, n_heads=4, window=4)
    layer = UndertowAttention(config).eval()

    x = torch.randn(1, 20, 32)
    perturbed = x.clone()
    perturbed[0, 5] += 5.0

    with torch.no_grad():
        before, after = layer(x), layer(perturbed)

    # Positions 5..8 can see position 5; position 9 onward cannot.
    assert not torch.allclose(before[:, 5:9], after[:, 5:9], atol=1e-4)
    assert torch.allclose(before[:, 9:], after[:, 9:], atol=1e-6)


def test_gradients_flow_to_every_projection() -> None:
    config = UndertowConfig(d_model=32, n_heads=4, window=8, plateau=6)
    layer = UndertowAttention(config)
    layer(torch.randn(2, 12, 32)).sum().backward()

    for name, parameter in layer.named_parameters():
        assert parameter.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(parameter.grad).all(), f"{name} gradient not finite"
        assert parameter.grad.abs().sum() > 0, f"{name} gradient is all zero"


# ── streaming ──────────────────────────────────────────────────────────────


def _layer(window: int = 8, plateau: int | None = 6, seed: int = 5) -> UndertowAttention:
    torch.manual_seed(seed)
    return UndertowAttention(
        UndertowConfig(d_model=64, n_heads=8, window=window, plateau=plateau)
    ).eval()


def test_init_state_shapes_and_emptiness() -> None:
    layer = _layer(window=8)
    state = layer.init_state(batch=3)
    assert state.keys.shape == (3, 8, 7, 8)
    assert state.values.shape == state.keys.shape
    assert state.seen == 0
    assert state.filled == 0
    assert torch.equal(state.keys, torch.zeros_like(state.keys))


@pytest.mark.parametrize("window,plateau", [(8, 6), (8, None), (1, None), (5, 0)])
def test_step_matches_forward_including_the_prefix(
    window: int, plateau: int | None
) -> None:
    """The Phase 3 gate.

    ``forward`` masks the partial-window prefix by positional validity;
    ``step`` masks it by counting filled buffer slots. Those are different
    pieces of code, so agreement over the first W-1 positions is a real check
    rather than a formality — which is exactly why the range is not skipped.
    """
    layer = _layer(window=window, plateau=plateau)
    x = torch.randn(2, 20, 64)

    with torch.no_grad():
        parallel = layer(x)
        state = layer.init_state(batch=2)
        stepped = []
        for t in range(x.shape[1]):
            y_t, state = layer.step(x[:, t : t + 1], state)
            stepped.append(y_t)
        sequential = torch.cat(stepped, dim=1)

    assert state.seen == x.shape[1]

    prefix = max(window - 1, 1)
    prefix_diff = (parallel[:, :prefix] - sequential[:, :prefix]).abs().max().item()
    total_diff = (parallel - sequential).abs().max().item()

    assert prefix_diff < 1e-5, f"prefix diverged by {prefix_diff:.3e}"
    assert total_diff < 1e-5, f"step vs forward diverged by {total_diff:.3e}"


def test_unmasked_step_would_have_been_wrong() -> None:
    """Characterise the bug the count-based mask prevents.

    Until the buffer fills, its front slots hold zeros from ``init_state``.
    Leaving them unmasked lets them take real attention mass — the streaming
    twin of zero-padding in ``forward``.  Reproduced here so that removing the
    mask fails a test, rather than quietly shifting the first W-1 outputs of
    every generated sequence.
    """
    layer = _layer(window=8, plateau=6)
    x = torch.randn(2, 20, 64)
    window = layer.config.window

    with torch.no_grad():
        parallel = layer(x)
        state = layer.init_state(batch=2)
        collected = []
        for t in range(x.shape[1]):
            q, k, v, gate = layer._project(x[:, t : t + 1])
            keys = torch.cat([state.keys, k.float()], dim=2)
            values = torch.cat([state.values, v.float()], dim=2)
            scores = torch.matmul(q.float(), keys.transpose(-2, -1)) / layer.scale
            scores = scores + layer.log_profile
            # The mask belongs here.  Its absence is the bug under test.
            o = torch.matmul(torch.softmax(scores, dim=-1), values)
            collected.append(layer._out(o, gate))
            state = layer._next_state(keys, values, state.seen + 1)
        unmasked = torch.cat(collected, dim=1)

    prefix_diff = (parallel[:, : window - 1] - unmasked[:, : window - 1]).abs().max()
    interior_diff = (parallel[:, window - 1 :] - unmasked[:, window - 1 :]).abs().max()

    assert prefix_diff.item() > 1e-5, "unmasked buffer slots should corrupt the prefix"
    assert interior_diff.item() < 1e-5, "and should leave the saturated region alone"


def test_step_rejects_multiple_positions() -> None:
    layer = _layer()
    state = layer.init_state(batch=2)
    with pytest.raises(ValueError, match="one position at a time"):
        layer.step(torch.randn(2, 3, 64), state)


def test_state_size_is_constant_in_generated_length() -> None:
    """The property the whole fixed window exists to buy."""
    layer = _layer(window=8)
    state = layer.init_state(batch=2)
    x = torch.randn(2, 1, 64)

    shapes = set()
    with torch.no_grad():
        for _ in range(200):
            _, state = layer.step(x, state)
            shapes.add((tuple(state.keys.shape), tuple(state.values.shape)))

    assert len(shapes) == 1, f"state changed shape during generation: {shapes}"
    assert state.seen == 200
    assert state.filled == 7  # saturates at window-1 and stays there


def test_prefill_then_step_matches_one_pass() -> None:
    """Prefill a prompt in parallel, continue token by token, same answer."""
    layer = _layer(window=8, plateau=6)
    x = torch.randn(2, 24, 64)
    split = 15

    with torch.no_grad():
        whole = layer(x)
        head, state = layer(x[:, :split], return_state=True)
        tail = []
        for t in range(split, x.shape[1]):
            y_t, state = layer.step(x[:, t : t + 1], state)
            tail.append(y_t)
        joined = torch.cat([head, *tail], dim=1)

    diff = (whole - joined).abs().max().item()
    assert diff < 1e-5, f"prefill+step diverged from one pass by {diff:.3e}"


@pytest.mark.parametrize("splits", [(4, 20), (8, 16), (1, 23), (12, 12), (23, 1)])
def test_chunked_forward_matches_one_pass(splits: tuple[int, int]) -> None:
    """Chunked prefill: a state handed to forward reaches back correctly.

    Includes a first chunk shorter than the window, where the second chunk's
    early queries must reach past a partly-filled buffer.
    """
    layer = _layer(window=8, plateau=6)
    first, second = splits
    x = torch.randn(2, first + second, 64)

    with torch.no_grad():
        whole = layer(x)
        head, state = layer(x[:, :first], return_state=True)
        tail, state = layer(x[:, first:], state=state, return_state=True)
        joined = torch.cat([head, tail], dim=1)

    assert state.seen == first + second
    diff = (whole - joined).abs().max().item()
    assert diff < 1e-5, f"chunked forward diverged by {diff:.3e}"


def test_state_is_frozen_so_streams_can_branch() -> None:
    """Two continuations from one state must not share a buffer."""
    layer = _layer(window=8)
    state = layer.init_state(batch=2)
    with torch.no_grad():
        _, state = layer.step(torch.randn(2, 1, 64), state)
        _, branch_a = layer.step(torch.randn(2, 1, 64), state)
        _, branch_b = layer.step(torch.randn(2, 1, 64), state)

    with pytest.raises(Exception):
        state.seen = 99  # type: ignore[misc]
    assert branch_a.keys.data_ptr() != branch_b.keys.data_ptr()
    assert not torch.allclose(branch_a.keys, branch_b.keys)


def test_forward_without_return_state_is_unchanged() -> None:
    """The default path must not have shifted under the new keyword args."""
    layer = _layer(window=8, plateau=6)
    x = torch.randn(2, 20, 64)
    with torch.no_grad():
        plain = layer(x)
        with_state, _ = layer(x, return_state=True)
    assert torch.equal(plain, with_state)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_generation_memory_is_flat() -> None:
    """Peak allocation must not grow with how much has been generated."""
    layer = _layer(window=8).cuda()
    x = torch.randn(2, 1, 64, device="cuda")

    peaks = []
    for length in (64, 256, 1024):
        state = layer.init_state(batch=2, device=torch.device("cuda"))
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for _ in range(length):
                _, state = layer.step(x, state)
        torch.cuda.synchronize()
        peaks.append(torch.cuda.max_memory_allocated())

    # 16x the tokens must not move peak allocation appreciably.
    growth = peaks[-1] / peaks[0]
    assert growth < 1.1, f"peak memory grew {growth:.2f}x over 16x the tokens"


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_windowed_matches_dense_on_cuda() -> None:
    """The exit gate is CPU *and* CUDA — different reduction order, same answer.

    Deselected in CI, which has no GPU.  Run locally before tagging.
    """
    q, k, v = _qkv(seq_len=48, dim=32, seed=13)
    q, k, v = q.cuda(), k.cuda(), v.cuda()

    for window, plateau in [(8, 6), (16, None), (32, 24)]:
        fast = windowed_attention(q, k, v, window, plateau)
        slow = dense_attention(q, k, v, window, plateau)
        diff = (fast - slow).abs().max().item()
        assert diff < TOL, f"W={window} P={plateau}: max diff {diff:.3e}"


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_layer_agrees_across_devices() -> None:
    """Same weights, same input, two devices — the layer is not device-dependent."""
    torch.manual_seed(41)
    config = UndertowConfig(d_model=64, n_heads=8, window=8, plateau=6)
    layer = UndertowAttention(config).eval()
    x = torch.randn(2, 20, 64)

    with torch.no_grad():
        on_cpu = layer(x)
        on_gpu = layer.cuda()(x.cuda()).cpu()

    diff = (on_cpu - on_gpu).abs().max().item()
    assert diff < 1e-5, f"cpu vs cuda max diff {diff:.3e}"


def test_log_profile_buffer_is_not_persistent() -> None:
    """It is a function of the config; serialising it invites disagreement."""
    config = UndertowConfig(d_model=32, n_heads=4, window=8, plateau=6)
    layer = UndertowAttention(config)
    assert "log_profile" not in layer.state_dict()
