"""`backend="fla"` is a faster path, not a different model.

The claim this file exists to hold: fla's Triton chunk kernel and Lumen's
reference compute **the same function**, forward and backward, to fp32
round-off — which is `docs/design/GATED_DELTANET.md` §3.7's *"one algorithm
reached by two routes"*, the case that earns the tight tolerance.

Consolidation that silently changes numerics invalidates every experimental
record downstream, so the tolerance is the load-bearing part here, not the
speed.  Speed is measured in `lumen.bench` and argued in the design record; a
backend that were merely *fast* would not be allowed in.

Two conventions are pinned here because both fail **silently** if reversed, and
a silent wrong answer is the one failure this component can least afford:

* `beta` goes in post-sigmoid, exactly as the reference takes it.
* `scale=1.0`, because the reference takes `q` already unit-norm.

The config-surface tests need no GPU and run in CI.  The numerical ones need
both CUDA and a working fla, and are marked accordingly.
"""

from __future__ import annotations

import pytest
import torch

from lumen.gdn import GatedDeltaNet, GatedDeltaNetConfig, HeadLayout
from lumen.gdn import fla_backend
from lumen.gdn.reference import chunk_gated_delta, sequential_gated_delta

# fp32 round-off on these shapes.  Deliberately not tighter: the two routes
# reassociate the same sums, so demanding more would be specifying a tolerance
# the arithmetic cannot meet -- which §3.7 calls worse than specifying a loose
# one.  The reference's own distance from the fp64 oracle is the yardstick, and
# both tests below check fla against *that* rather than against a bare number.
FP32_ROUNDOFF = 5e-6

# `triton` and `gpu` are the project's markers, so CI's
# `-m "not gpu and not triton"` deselects these rather than collecting and
# skipping them.  The skipif is still needed for a local box that has one and
# not the other.
requires_fla = pytest.mark.triton
requires_cuda = pytest.mark.gpu

needs_fla = pytest.mark.skipif(
    not fla_backend.HAS_FLA, reason="flash-linear-attention is not installed"
)
needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)


# ── the config surface: no device needed ──────────────────────────────────


def _config(**overrides: object) -> GatedDeltaNetConfig:
    base: dict[str, object] = dict(
        d_model=256, layout=HeadLayout.diagonal(4), expand_k=1.0, expand_v=4.0
    )
    base.update(overrides)
    return GatedDeltaNetConfig(**base)  # type: ignore[arg-type]


def test_reference_is_the_default() -> None:
    """Opt-in, never auto-detected -- even with fla importable and a GPU present."""
    assert _config().backend == "reference"


def test_an_unknown_backend_is_refused() -> None:
    with pytest.raises(ValueError, match="backend must be one of"):
        _config(backend="triton")


@pytest.mark.parametrize(
    "layout",
    [HeadLayout.shared_key(4), HeadLayout.shared_value(4), HeadLayout.crossed(2, 2)],
    ids=["shared_key", "shared_value", "crossed"],
)
def test_fla_refuses_layouts_it_cannot_express(layout: HeadLayout) -> None:
    """fla implements one key per head; anything else is refused at construction.

    Refused where the config is *written*, not at the first forward -- an
    unsupported arrangement is a fact about the configuration, and discovering
    it several thousand steps into a run is the expensive way to learn it.
    """
    with pytest.raises(ValueError, match="one key per head"):
        _config(layout=layout, backend="fla")


def test_fla_refuses_state_gated_decay() -> None:
    with pytest.raises(ValueError, match="own modulation"):
        _config(decay="state_gated", backend="fla")


def test_state_decay_is_accepted_because_it_is_the_same_object_here() -> None:
    """On a diagonal layout `G_k == H` and `m == 1`, so per-group == per-state.

    This is not leniency.  Refusing `decay="state"` would refuse a
    configuration that is bit-identical to one being accepted, which would make
    the backend's supported set a statement about spelling rather than about
    what the kernel computes.
    """
    assert _config(decay="state", backend="fla").backend == "fla"


def test_the_backend_is_in_the_repr() -> None:
    """Which object produced a checkpoint has to be visible without the config."""
    assert "backend=reference" in repr(GatedDeltaNet(_config()))


@requires_fla
@needs_fla
def test_fla_refuses_a_read_cache_on_any_device() -> None:
    """fla's kernel returns only the final state, so it cannot build a read cache.

    Refused loudly, and on a CPU tensor -- where fla would otherwise fall back
    to the reference and could have produced one.  A capability present on the
    CPU box and absent on the GPU the same config trains on is a trap, and the
    forward without a cache is still allowed to fall back exactly as before.
    """
    layer = GatedDeltaNet(_config(backend="fla"))
    x = torch.randn(2, 32, 256)
    assert layer(x).shape == x.shape  # the fallback itself is untouched
    with pytest.raises(RuntimeError, match="cannot build a read cache"):
        layer(x, return_cache=True)


# ── the numerics: this is the part that licenses the switch ───────────────


def _draw(
    batch: int, seq: int, heads: int, d_k: int, d_v: int, device: str
) -> tuple[torch.Tensor, ...]:
    """Inputs in Lumen kernel shape `(B, G_k, m, T, d)`, diagonal layout."""
    torch.manual_seed(0)
    shape = (batch, heads, 1)
    q = torch.randn(*shape, seq, d_k, device=device)
    k = torch.randn(*shape, seq, d_k, device=device)
    q = torch.nn.functional.normalize(q, dim=-1).requires_grad_()
    k = torch.nn.functional.normalize(k, dim=-1).requires_grad_()
    v = torch.randn(*shape, seq, d_v, device=device, requires_grad=True)
    # beta spans [0, 2]: past 1 the update is a reflection rather than a
    # contraction, which is where the two implementations could most easily
    # disagree and where `beta_max=2` actually puts a trained model.
    beta = (2.0 * torch.rand(*shape, seq, device=device)).requires_grad_()
    log_alpha = (-torch.rand(*shape, seq, device=device) * 0.1).requires_grad_()
    return q, k, v, beta, log_alpha


@requires_cuda
@requires_fla
@needs_cuda
@needs_fla
def test_fla_is_as_close_to_the_oracle_as_the_reference_is() -> None:
    """Both routes, scored against the same fp64 sequential oracle.

    The comparison is against Lumen's own oracle rather than against Lumen's
    chunkwise path, because that is the question: not "do these two agree" but
    "do they both compute the recurrence".
    """
    q, k, v, beta, log_alpha = _draw(2, 256, 4, 64, 128, "cuda")

    out_ref, state_ref = chunk_gated_delta(q, k, v, beta, log_alpha, 64)
    out_fla, state_fla = fla_backend.chunk_gated_delta_fla(q, k, v, beta, log_alpha)
    out_oracle, state_oracle = sequential_gated_delta(
        q.double(), k.double(), v.double(), beta.double(), log_alpha.double()
    )

    def distance(a: torch.Tensor, b: torch.Tensor) -> float:
        return ((a.double() - b.double()).norm() / b.double().norm()).item()

    reference_error = distance(out_ref, out_oracle)
    fla_error = distance(out_fla, out_oracle)

    assert fla_error < FP32_ROUNDOFF
    # The real claim: not merely "small", but *no worse than the path already
    # trusted*. A bare threshold would still pass if fla were ten times further
    # out and the threshold happened to be loose.
    assert fla_error < reference_error * 10
    assert distance(state_fla, state_oracle) < FP32_ROUNDOFF


@requires_cuda
@requires_fla
@needs_cuda
@needs_fla
def test_the_gradients_agree_too() -> None:
    """The half that decides a training run, and the half a forward check misses.

    An accelerated forward bolted to a subtly different backward is how a kernel
    passes every numerical test and trains a different model.
    """
    q, k, v, beta, log_alpha = _draw(2, 256, 4, 64, 128, "cuda")
    inputs = [q, k, v, beta, log_alpha]

    out_ref, _ = chunk_gated_delta(q, k, v, beta, log_alpha, 64)
    out_fla, _ = fla_backend.chunk_gated_delta_fla(q, k, v, beta, log_alpha)

    # One upstream signal into both, or the comparison is of two different
    # questions.
    seed = torch.randn_like(out_ref)
    grads_ref = torch.autograd.grad(out_ref, inputs, seed, retain_graph=True)
    grads_fla = torch.autograd.grad(out_fla, inputs, seed, retain_graph=True)

    for name, a, b in zip(("q", "k", "v", "beta", "log_alpha"), grads_fla, grads_ref):
        error = ((a.double() - b.double()).norm() / b.double().norm()).item()
        assert error < FP32_ROUNDOFF, f"d{name} disagrees by {error:.3e}"


@requires_cuda
@requires_fla
@needs_cuda
@needs_fla
def test_an_incoming_state_is_carried() -> None:
    """Streaming is not an afterthought: a resumed chunk must match one pass.

    This is the test that would catch an `initial_state` wired to the wrong
    axis -- which produces plausible finite output and a silently different
    model, exactly the failure the layout adapter is most exposed to.
    """
    q, k, v, beta, log_alpha = _draw(2, 128, 4, 64, 128, "cuda")
    split = 64

    whole, _ = fla_backend.chunk_gated_delta_fla(q, k, v, beta, log_alpha)

    head = [t[..., :split, :] if t.ndim == 5 else t[..., :split] for t in (q, k, v, beta, log_alpha)]
    tail = [t[..., split:, :] if t.ndim == 5 else t[..., split:] for t in (q, k, v, beta, log_alpha)]
    first, carried = fla_backend.chunk_gated_delta_fla(*head)
    second, _ = fla_backend.chunk_gated_delta_fla(*tail, carried)

    rejoined = torch.cat([first, second], dim=-2)
    assert (rejoined - whole).abs().max() < FP32_ROUNDOFF


@requires_cuda
@requires_fla
@needs_cuda
@needs_fla
def test_the_layer_agrees_with_itself_across_backends() -> None:
    """End to end, one set of weights, both backends -- the consumer's question.

    Everything above tests the kernel in isolation.  This tests that the layer
    *wires* it correctly: the beta convention, the scale, and the shape
    adapter, all of which produce plausible output when wrong.
    """
    config = dict(d_model=256, layout=HeadLayout.diagonal(4), expand_k=1.0, expand_v=4.0)
    torch.manual_seed(0)
    reference = GatedDeltaNet(GatedDeltaNetConfig(**config)).cuda()

    accelerated = GatedDeltaNet(GatedDeltaNetConfig(**config, backend="fla")).cuda()
    accelerated.load_state_dict(reference.state_dict())

    x = torch.randn(2, 256, 256, device="cuda")
    with torch.no_grad():
        got_ref = reference(x)
        got_fla = accelerated(x)

    error = ((got_fla - got_ref).norm() / got_ref.norm()).item()
    assert error < FP32_ROUNDOFF, f"backends disagree by {error:.3e}"


@requires_fla
@needs_fla
def test_a_cpu_tensor_falls_back_instead_of_failing() -> None:
    """A device gap is a fallback; a missing capability is an error.

    The distinction is Undertow's (§3.4) and is kept identical here on purpose:
    the same word in two components has to mean the same thing.
    """
    layer = GatedDeltaNet(GatedDeltaNetConfig(
        d_model=256, layout=HeadLayout.diagonal(4), expand_k=1.0, expand_v=4.0,
        backend="fla",
    ))
    assert layer(torch.randn(1, 32, 256)).shape == (1, 32, 256)
