"""Block — the acceptance conditions of the design record, as tests.

The load-bearing ones are :func:`test_streaming_agrees_with_parallel` and
:func:`test_checkpoint_keys_are_the_inherited_ones`. The first says the two
paths through the block compute the same thing; the second says a claim the
record makes about interoperation is actually true of the code, and it is the
one that cannot be fixed later — a key is an interface with archived
checkpoints on the other side of it.

Everything here runs on CPU in fp32.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
import torch.nn as nn

from lumen.block import Block, BlockState
from lumen.gdn import GatedDeltaNet, GatedDeltaNetConfig, HeadLayout
from lumen.nn import SwiGLU
from lumen.undertow import UndertowAttention, UndertowConfig

D_MODEL = 64
EXACT = 1e-6


def a_mixer(d_model: int = D_MODEL) -> GatedDeltaNet:
    return GatedDeltaNet(
        GatedDeltaNetConfig(
            d_model=d_model, layout=HeadLayout.shared_key(4), expand_k=2.0
        )
    )


def a_local(d_model: int = D_MODEL) -> UndertowAttention:
    return UndertowAttention(
        UndertowConfig(d_model=d_model, n_heads=4, window=8)
    )


#: The four arrangements a block can be in.  They differ in residual-write
#: count, which is the axis along which a wrong residual arrangement shows up —
#: so a test that only covers one of them covers almost nothing.
COMPOSITIONS = {
    "mixer": dict(d_mlp=0, local=None),
    "mixer+mlp": dict(d_mlp=128, local=None),
    "local+mixer": dict(d_mlp=0, local="yes"),
    "local+mixer+mlp": dict(d_mlp=128, local="yes"),
}


def build(name: str, seed: int = 0) -> Block:
    torch.manual_seed(seed)
    spec = COMPOSITIONS[name]
    local = a_local() if spec["local"] else None
    return Block(
        D_MODEL, a_mixer(), norm_eps=1e-5, d_mlp=spec["d_mlp"], local=local
    ).eval()


ALL = list(COMPOSITIONS)


# ── construction ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ALL)
def test_shape_is_preserved(name):
    x = torch.randn(2, 12, D_MODEL)
    assert build(name)(x).shape == x.shape


def test_norm_eps_and_d_mlp_have_no_defaults():
    """Both are open questions, so neither may be answered by omission.

    `latticedynamics/lumen#6` holds the argument. A default here would close a
    question this repository deliberately left open, silently, at every call
    site that did not think about it.
    """
    with pytest.raises(TypeError):
        Block(D_MODEL, a_mixer(), d_mlp=0)          # type: ignore[call-arg]
    with pytest.raises(TypeError):
        Block(D_MODEL, a_mixer(), norm_eps=1e-5)    # type: ignore[call-arg]


def test_rejects_a_negative_mlp_width():
    with pytest.raises(ValueError):
        Block(D_MODEL, a_mixer(), norm_eps=1e-5, d_mlp=-1)


@pytest.mark.parametrize("name", ALL)
def test_checkpoint_keys_are_the_inherited_ones(name):
    """The keys are the interface, and these are the lineage's.

    Adopting them verbatim is what lets a stack built here load weights trained
    by the implementation this was extracted from. Renaming any of them —
    including tidying ``norm2`` into something that matches ``norm_local`` — is
    a permanent remap entry against archived runs, bought with nothing
    measurable. This test is what makes that a decision rather than a drift.
    """
    keys = {
        key
        for key in build(name).state_dict()
        if not key.startswith(("local.", "mixer."))
    }
    expected = {"norm.weight"}
    if COMPOSITIONS[name]["local"]:
        expected.add("norm_local.weight")
    if COMPOSITIONS[name]["d_mlp"]:
        expected |= {
            "norm2.weight",
            "mlp.up.weight",
            "mlp.gate.weight",
            "mlp.down_proj.weight",
        }
    assert keys == expected


# ── the residual arrangement ──────────────────────────────────────────────


@pytest.mark.parametrize("name", ALL)
def test_the_residual_is_outside_the_norm(name):
    """Silence every sub-layer and the block must be exactly the identity.

    Zeroing the residual writes leaves ``x + 0`` at each stage, so anything
    other than a bit-identical passthrough means the arrangement is not
    ``x = x + f(norm(x))`` — a pre-norm block that normalised the *stream*
    rather than the sub-layer input would fail this, and would be very hard to
    notice any other way.
    """
    block = build(name)
    x = torch.randn(2, 12, D_MODEL)
    with torch.no_grad():
        for projection in block.residual_out_projections():
            for parameter in projection.parameters():
                parameter.zero_()
        assert torch.equal(block(x), x)


@pytest.mark.parametrize("name", ALL)
def test_residual_writes_counts_sub_layers_not_projections(name):
    """Two different quantities that agree only by accident.

    The depth-scaling divisor's `2` means *residual writes*; the init sweep
    needs *projections*. They coincide for every arrangement here and would
    stop coinciding for a sub-layer with two output projections, which is
    exactly when a caller that conflated them would be quietly wrong.
    """
    block = build(name)
    expected = 1 + bool(COMPOSITIONS[name]["local"]) + bool(COMPOSITIONS[name]["d_mlp"])
    assert block.residual_writes() == expected


@pytest.mark.parametrize("name", ALL)
def test_residual_out_projections_reach_every_sub_layer(name):
    """The block answers for its sub-layers, so a stack never walks into them."""
    block = build(name)
    declared = {
        id(parameter)
        for projection in block.residual_out_projections()
        for parameter in projection.parameters()
    }
    swept = {
        id(parameter)
        for key, parameter in block.named_parameters()
        if key.endswith(("o_proj.weight", "down_proj.weight"))
    }
    assert declared and declared == swept


def test_a_sub_layer_that_cannot_say_where_it_writes_is_refused():
    """Loudly. A silently skipped write is a model that trains and is wrong."""
    block = Block(D_MODEL, nn.Linear(D_MODEL, D_MODEL), norm_eps=1e-5, d_mlp=0)
    with pytest.raises(TypeError, match="residual_out_projections"):
        block.residual_out_projections()


# ── polymorphism ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "mixer",
    [a_mixer, a_local, lambda: SwiGLU(D_MODEL, 128)],
    ids=["gdn", "undertow", "swiglu"],
)
def test_the_mixer_slot_holds_anything_including_a_stateless_module(mixer):
    """Holding either mixer without knowing which — and a stateless one too.

    The stateless case is the interesting one: a layer that trades its state for
    a plain feed-forward needs no support in this class, because the block reads
    statefulness off the object. That is what turns per-layer heterogeneity from
    a feature into something a caller already can do.
    """
    torch.manual_seed(0)
    block = Block(D_MODEL, mixer(), norm_eps=1e-5, d_mlp=0).eval()
    x = torch.randn(2, 12, D_MODEL)
    assert block(x).shape == x.shape

    state = block.init_state(2)
    y, successor = block.step(x[:, :1], state)
    assert y.shape == (2, 1, D_MODEL)
    assert isinstance(successor, BlockState)


def test_a_stateless_mixer_has_a_state_slot_that_is_none():
    """``None`` means absent-or-stateless, and never "a zeroed state"."""
    block = Block(D_MODEL, SwiGLU(D_MODEL, 128), norm_eps=1e-5, d_mlp=0)
    assert block.init_state(2).mixer is None


# ── streaming ─────────────────────────────────────────────────────────────


def test_block_state_is_frozen():
    """Forking a stream must not let one branch mutate the other's state."""
    state = BlockState(local=None, mixer=None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.mixer = 1  # type: ignore[misc]


@pytest.mark.parametrize("name", ALL)
def test_streaming_agrees_with_parallel(name):
    """Prefill a prefix, step the rest, and land where one pass lands.

    The block-level version of the check the mixers already hold. It is the one
    that catches a state threaded to the wrong sub-layer, which is otherwise
    invisible: outputs stay the right shape and only the *history* is wrong.
    """
    block = build(name)
    x = torch.randn(2, 10, D_MODEL)

    whole = block(x)

    split = 6
    prefix, state = block(x[:, :split], return_state=True)
    outputs = [prefix]
    for t in range(split, x.shape[1]):
        y, state = block.step(x[:, t : t + 1], state)
        outputs.append(y)

    assert (torch.cat(outputs, dim=1) - whole).abs().max() < EXACT


@pytest.mark.parametrize("name", ALL)
def test_state_threads_without_the_caller_knowing_the_shape(name):
    """Split a sequence, thread the state across the seam, get one pass back.

    The caller writes the same three lines whether the block has one stateful
    sub-layer or two. That is the property the block exists to provide, and it
    is why ``init_state`` lives here rather than in every consumer.
    """
    block = build(name)
    x = torch.randn(2, 16, D_MODEL)

    whole = block(x)
    first, state = block(x[:, :8], return_state=True)
    second = block(x[:, 8:], state=state)

    assert (torch.cat([first, second], dim=1) - whole).abs().max() < EXACT


@pytest.mark.parametrize("name", ALL)
def test_step_returns_a_successor_rather_than_mutating(name):
    """Lineage branching: two futures from one state must not interfere."""
    block = build(name)
    x = torch.randn(2, 1, D_MODEL)
    state = block.init_state(2)

    a, after_a = block.step(x, state)
    b, after_b = block.step(x, state)

    assert torch.equal(a, b), "the shared prior state was mutated by the first step"
    assert after_a is not state and after_b is not state


def test_step_refuses_more_than_one_position():
    block = build("mixer")
    with pytest.raises(ValueError, match="one position at a time"):
        block.step(torch.randn(2, 3, D_MODEL), block.init_state(2))


@pytest.mark.parametrize("name", ALL)
def test_init_state_follows_the_block(name):
    """A batch of N independent streams, which is a primary inference path.

    Not a decode convenience: a consumer stepping hundreds of independent
    streams in lockstep uses this as its main path, and nothing else in the
    library exercises it at a batch size above 1.
    """
    block = build(name)
    state = block.init_state(7)
    y, _ = block.step(torch.randn(7, 1, D_MODEL), state)
    assert y.shape == (7, 1, D_MODEL)
