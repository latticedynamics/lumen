"""State containers as pytree nodes — can a ``torch`` transform walk a state?

The load-bearing ones are :func:`test_undertow_state_keeps_its_count_across_a_round_trip`
and :func:`test_vmap_over_stacked_parameter_sets`. The first is the regression
that the obvious-looking registration silently loses — a state that forgets
``seen`` masks the wrong slots and returns numbers rather than raising. The
second is the capability the registration exists to buy: many distinct parameter
sets through one module definition in a single batched call, without a second,
batched implementation of the layer.

CPU-only on purpose. Every traversal here is over the reference path and needs
no device, which puts the whole file in what CI already covers.
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.utils._pytree as pytree
from torch.func import functional_call, stack_module_state, vmap

from lumen.block import Block
from lumen.gdn import GatedDeltaNet, GatedDeltaNetConfig, HeadLayout
from lumen.pytree import REGISTERED, register_state_pytrees
from lumen.stack import Stack
from lumen.undertow import UndertowAttention, UndertowConfig, UndertowState

D_MODEL = 32
N_LAYERS = 2
BATCH = 3
SEQ = 8
SETS = 4

needs_registration = pytest.mark.skipif(
    len(REGISTERED) < 4,
    reason=f"torch {torch.__version__} registered only {REGISTERED}",
)


def a_mixer() -> GatedDeltaNet:
    return GatedDeltaNet(
        GatedDeltaNetConfig(
            d_model=D_MODEL,
            layout=HeadLayout.shared_key(2),
            expand_k=1.0,
            expand_v=1.0,
            chunk_size=4,
            conv_size=4,
        )
    )


def a_local() -> UndertowAttention:
    return UndertowAttention(UndertowConfig(d_model=D_MODEL, n_heads=2, window=4))


def gdn_only(index: int) -> Block:
    return Block(D_MODEL, a_mixer(), norm_eps=1e-5, d_mlp=0)


def undertow_only(index: int) -> Block:
    return Block(D_MODEL, a_local(), norm_eps=1e-5, d_mlp=0)


def both(index: int) -> Block:
    return Block(D_MODEL, a_mixer(), norm_eps=1e-5, d_mlp=64, local=a_local())


FACTORIES = [
    pytest.param(gdn_only, id="gdn"),
    pytest.param(undertow_only, id="undertow"),
    pytest.param(both, id="undertow-local-plus-gdn"),
]


# ── registration itself ───────────────────────────────────────────────────


def test_import_registers_every_state() -> None:
    """The whole set, at import, with no opt-in from the consumer."""
    assert REGISTERED == (
        "GatedDeltaNetState",
        "BlockState",
        "StackState",
        "UndertowState",
    )


def test_registering_again_is_a_quiet_no_op() -> None:
    """Idempotent, and silent about it.

    The two ``torch`` entry points disagree about repetition — one overwrites
    with a warning, the other raises — and a consumer calling the public
    function after import should meet neither.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert register_state_pytrees() == REGISTERED
        assert register_state_pytrees() == REGISTERED


# ── the containers ────────────────────────────────────────────────────────


@needs_registration
def test_undertow_state_keeps_its_count_across_a_round_trip() -> None:
    """``seen`` is context, and dropping it is a silent wrong answer.

    A registration that flattens ``seen`` away rebuilds the state as ``seen=0``.
    Nothing raises: the window masks the wrong slots and returns plausible
    numbers, which is why this is asserted rather than left to the vmap test.
    """
    layer = a_local()
    state = layer.init_state(BATCH)
    _, state = layer(torch.randn(BATCH, SEQ, D_MODEL), state=state, return_state=True)
    assert state.seen == SEQ

    leaves, spec = pytree.tree_flatten(state)
    assert [type(leaf).__name__ for leaf in leaves] == ["Tensor", "Tensor"]

    restored = pytree.tree_unflatten(leaves, spec)
    assert restored.seen == SEQ
    assert restored.filled == state.filled
    torch.testing.assert_close(restored.keys, state.keys)
    torch.testing.assert_close(restored.values, state.values)


@needs_registration
@pytest.mark.parametrize("factory", FACTORIES)
def test_every_leaf_of_a_stack_state_is_a_tensor(factory) -> None:
    """Absent slots are context, not ``None`` leaves.

    ``BlockState.local`` on a block without one, and ``GatedDeltaNetState.conv``
    on a layer without a convolution, are the two that exist. A ``None`` leaf is
    rejected by ``vmap`` exactly as a non-tensor one is.
    """
    trunk = Stack(D_MODEL, N_LAYERS, factory, norm_eps=1e-5)
    leaves, spec = pytree.tree_flatten(trunk.init_state(BATCH))

    assert leaves, "a stateful trunk flattened to nothing"
    assert all(torch.is_tensor(leaf) for leaf in leaves)
    assert isinstance(pytree.tree_unflatten(leaves, spec).blocks, tuple)


@needs_registration
@pytest.mark.parametrize("factory", FACTORIES)
def test_a_state_survives_flattening_unchanged(factory) -> None:
    """Structure and values both, so a transform can be a no-op."""
    trunk = Stack(D_MODEL, N_LAYERS, factory, norm_eps=1e-5)
    state = trunk.init_state(BATCH)

    identical = pytree.tree_map(lambda leaf: leaf, state)
    assert pytree.tree_structure(identical) == pytree.tree_structure(state)

    original, restored = pytree.tree_leaves(state), pytree.tree_leaves(identical)
    for before, after in zip(original, restored):
        torch.testing.assert_close(before, after)


@needs_registration
@pytest.mark.parametrize("factory", FACTORIES)
def test_paths_are_available_on_every_state(factory) -> None:
    """``tree_map_with_path`` works, or the four states are not alike after all.

    The keyed flatten is optional to ``torch`` and not optional here: without it
    the path-carrying traversals work on three states and fail on the fourth,
    which is the asymmetry between mixers this module exists to remove.
    """
    trunk = Stack(D_MODEL, N_LAYERS, factory, norm_eps=1e-5)
    paths = pytree.tree_flatten_with_path(trunk.init_state(BATCH))[0]

    assert paths and all(path for path, _ in paths)


# ── the capability it buys ────────────────────────────────────────────────


@needs_registration
@pytest.mark.parametrize("factory", FACTORIES)
def test_vmap_over_stacked_parameter_sets(factory) -> None:
    """`SETS` parameter sets × `BATCH` streams, one call, state carrying both.

    This is the case the registration exists for: a population-based consumer
    evaluating many distinct weight draws without a second, batched
    implementation of the trunk — a forked implementation being exactly what
    consolidation is for avoiding.
    """
    models = [Stack(D_MODEL, N_LAYERS, factory, norm_eps=1e-5) for _ in range(SETS)]
    params, buffers = stack_module_state(models)
    base = Stack(D_MODEL, N_LAYERS, factory, norm_eps=1e-5).to("meta")

    def one_set(parameters, buffered, x, state):
        return functional_call(
            base, (parameters, buffered), (x,), {"state": state, "return_state": True}
        )

    x = torch.randn(SETS, BATCH, SEQ, D_MODEL)
    state = pytree.tree_map(
        lambda leaf: leaf.unsqueeze(0).expand(SETS, *leaf.shape).contiguous(),
        models[0].init_state(BATCH),
    )

    y, successor = vmap(one_set)(params, buffers, x, state)

    assert y.shape == (SETS, BATCH, SEQ, D_MODEL)
    assert len(successor.blocks) == N_LAYERS
    assert all(leaf.shape[0] == SETS for leaf in pytree.tree_leaves(successor))


@needs_registration
def test_vmapped_sets_agree_with_the_models_run_one_at_a_time() -> None:
    """The batched call is the same arithmetic, not merely the same shapes.

    Without this the test above passes on a transform that quietly broadcast one
    parameter set across all of them.
    """
    models = [Stack(D_MODEL, N_LAYERS, gdn_only, norm_eps=1e-5) for _ in range(SETS)]
    params, buffers = stack_module_state(models)
    base = Stack(D_MODEL, N_LAYERS, gdn_only, norm_eps=1e-5).to("meta")

    x = torch.randn(SETS, BATCH, SEQ, D_MODEL)

    batched = vmap(
        lambda p, b, xi: functional_call(base, (p, b), (xi,))
    )(params, buffers, x)
    one_by_one = torch.stack([model(x[i]) for i, model in enumerate(models)])

    torch.testing.assert_close(batched, one_by_one, rtol=1e-5, atol=1e-5)


@needs_registration
def test_a_vmapped_state_still_streams() -> None:
    """The successor is a real state, not a shape that merely survived.

    Feeding it back is what a population-based consumer actually does between
    chunks, so a state that flattens and unflattens but cannot be continued
    would pass every test above and fail in use.
    """
    models = [Stack(D_MODEL, N_LAYERS, both, norm_eps=1e-5) for _ in range(SETS)]
    params, buffers = stack_module_state(models)
    base = Stack(D_MODEL, N_LAYERS, both, norm_eps=1e-5).to("meta")

    def one_set(parameters, buffered, x, state):
        return functional_call(
            base, (parameters, buffered), (x,), {"state": state, "return_state": True}
        )

    state = pytree.tree_map(
        lambda leaf: leaf.unsqueeze(0).expand(SETS, *leaf.shape).contiguous(),
        models[0].init_state(BATCH),
    )

    first = torch.randn(SETS, BATCH, SEQ, D_MODEL)
    _, state = vmap(one_set)(params, buffers, first, state)

    second = torch.randn(SETS, BATCH, SEQ, D_MODEL)
    y, state = vmap(one_set)(params, buffers, second, state)

    assert y.shape == (SETS, BATCH, SEQ, D_MODEL)
    # Both chunks were consumed, so the window state counted both.
    assert state.blocks[0].local.seen == 2 * SEQ
