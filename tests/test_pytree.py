"""State containers as pytree nodes — can a ``torch`` transform walk a state?

The load-bearing ones are :func:`test_undertow_state_keeps_its_count_across_a_round_trip`
and :func:`test_vmap_over_stacked_parameter_sets`. The first is the regression
that the obvious-looking registration silently loses — a state that forgets
``seen`` masks the wrong slots and returns numbers rather than raising. The
second is the capability the registration exists to buy: many distinct parameter
sets through one module definition in a single batched call, without a second,
batched implementation of the layer.

The read caches are not states, and are tested here because they meet the same
boundary: :func:`test_vmapped_rereads_of_vmapped_writes_agree_with_a_loop` is
their counterpart to the second, with the writes and the re-reads in separate
calls — the arrangement an unregistered cache cannot cross.

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
from lumen.gdn import (
    GatedDeltaNet,
    GatedDeltaNetConfig,
    GatedDeltaNetReadCache,
    HeadLayout,
)
from lumen.pytree import REGISTERED, register_state_pytrees
from lumen.sdm import SparseDeltaMemory, SparseDeltaMemoryConfig
from lumen.stack import Stack
from lumen.undertow import UndertowAttention, UndertowConfig, UndertowState

D_MODEL = 32
N_LAYERS = 2
BATCH = 3
SEQ = 8
SETS = 4

needs_registration = pytest.mark.skipif(
    len(REGISTERED) < 7,
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


def a_memory() -> SparseDeltaMemory:
    # Learned, so the initial table is a parameter and is stacked per set like
    # every other weight -- the arrangement a population-based consumer runs.
    return SparseDeltaMemory(
        SparseDeltaMemoryConfig(
            d_model=D_MODEL,
            n_heads=2,
            n_slots=16,
            initial_memory="learned",
            n_writes=3,
            n_reads=3,
            chunk_size=4,
        )
    )


def gdn_only(index: int) -> Block:
    return Block(D_MODEL, a_mixer(), norm_eps=1e-5, d_mlp=0)


def undertow_only(index: int) -> Block:
    return Block(D_MODEL, a_local(), norm_eps=1e-5, d_mlp=0)


def sdm_only(index: int) -> Block:
    return Block(D_MODEL, a_memory(), norm_eps=1e-5, d_mlp=0)


def both(index: int) -> Block:
    return Block(D_MODEL, a_mixer(), norm_eps=1e-5, d_mlp=64, local=a_local())


FACTORIES = [
    pytest.param(gdn_only, id="gdn"),
    pytest.param(undertow_only, id="undertow"),
    pytest.param(sdm_only, id="sdm"),
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
        "SparseDeltaMemoryState",
        "ReadCache",
        "GatedDeltaNetReadCache",
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
    """``tree_map_with_path`` works, or the states are not alike after all.

    The keyed flatten is optional to ``torch`` and not optional here: without it
    the path-carrying traversals work on the dataclass states and fail on one,
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
@pytest.mark.parametrize(
    "factory", [pytest.param(gdn_only, id="gdn"), pytest.param(sdm_only, id="sdm")]
)
def test_vmapped_sets_agree_with_the_models_run_one_at_a_time(factory) -> None:
    """The batched call is the same arithmetic, not merely the same shapes.

    Without this the test above passes on a transform that quietly broadcast one
    parameter set across all of them.
    """
    models = [Stack(D_MODEL, N_LAYERS, factory, norm_eps=1e-5) for _ in range(SETS)]
    params, buffers = stack_module_state(models)
    base = Stack(D_MODEL, N_LAYERS, factory, norm_eps=1e-5).to("meta")

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


@needs_registration
def test_a_vmapped_sparse_delta_memory_streams_like_one_pass() -> None:
    """Two vmapped chunks with the table carried equal one vmapped pass.

    The sparse delta memory's kernel keeps every shape a function of the
    configuration alone -- no boolean selection, no host sync -- precisely so
    this works.  A kernel that picked its writers with ``w_idx[first]`` would be
    refused by ``vmap`` outright; this holds that the carried table is also the
    right one.
    """
    models = [Stack(D_MODEL, N_LAYERS, sdm_only, norm_eps=1e-5) for _ in range(SETS)]
    params, buffers = stack_module_state(models)
    base = Stack(D_MODEL, N_LAYERS, sdm_only, norm_eps=1e-5).to("meta")

    def one_set(parameters, buffered, x, state):
        return functional_call(
            base, (parameters, buffered), (x,), {"state": state, "return_state": True}
        )

    state = pytree.tree_map(
        lambda leaf: leaf.unsqueeze(0).expand(SETS, *leaf.shape).contiguous(),
        models[0].init_state(BATCH),
    )
    x = torch.randn(SETS, BATCH, 2 * SEQ, D_MODEL)

    whole, _ = vmap(one_set)(params, buffers, x, state)
    head, carried = vmap(one_set)(params, buffers, x[:, :, :SEQ], state)
    tail, _ = vmap(one_set)(params, buffers, x[:, :, SEQ:], carried)

    torch.testing.assert_close(torch.cat([head, tail], dim=2), whole)


# ── the read caches ───────────────────────────────────────────────────────


class Rereading(torch.nn.Module):
    """A layer whose ``forward`` is its ``reread``.

    ``functional_call`` only ever calls ``forward``, so this is the smallest
    module that lets it reach the other method with stacked weights.
    """

    def __init__(self, layer: GatedDeltaNet) -> None:
        super().__init__()
        self.layer = layer

    def forward(
        self, x: torch.Tensor, cache: GatedDeltaNetReadCache
    ) -> torch.Tensor:
        return self.layer.reread(x, cache)


def a_cache(
    continued: bool,
) -> tuple[GatedDeltaNet, torch.Tensor, torch.Tensor, GatedDeltaNetReadCache]:
    """A layer, the input it wrote, the output it gave, and the cache it wrote.

    A continued pass is the one whose cache carries a ``conv`` tuple.  A first
    pass carries ``None`` there, which has to stay context rather than a leaf.
    """
    layer = a_mixer()
    state = None
    if continued:
        _, state = layer(torch.randn(BATCH, SEQ, D_MODEL), return_state=True)
    x = torch.randn(BATCH, SEQ, D_MODEL)
    y, cache = layer(x, state=state, return_cache=True)
    return layer, x, y, cache


PASSES = [pytest.param(False, id="first-pass"), pytest.param(True, id="continued")]


@needs_registration
@pytest.mark.parametrize("continued", PASSES)
def test_a_read_cache_keeps_its_length_across_a_round_trip(continued: bool) -> None:
    """``seq_len`` is context, as ``seen`` is, and the rebuilt cache still reads.

    A cache that lost its length would not answer wrongly -- ``reread`` checks
    the query against it -- but it would fail at the re-read, naming a shape,
    far from the flatten that dropped it.  So it is asserted at the flatten.
    """
    layer, x, y, cache = a_cache(continued)

    leaves, spec = pytree.tree_flatten(cache)
    assert all(torch.is_tensor(leaf) for leaf in leaves)
    assert len(leaves) == 5 + (3 if continued else 0)

    restored = pytree.tree_unflatten(leaves, spec)
    assert restored.trajectory.seq_len == SEQ
    assert (restored.conv is None) == (not continued)
    # A re-read with the original input is the original output, exactly -- so
    # a rebuilt cache that is a cache in every respect passes this at zero.
    torch.testing.assert_close(layer.reread(x, restored), y, rtol=0, atol=0)


@needs_registration
@pytest.mark.parametrize("continued", PASSES)
def test_paths_reach_every_leaf_of_a_read_cache(continued: bool) -> None:
    """Through the outer dataclass and the inner hand-rolled flatten alike.

    The two halves of a cache are registered by different entry points.  A
    path that stopped at ``.trajectory`` would mean the inner keyed flatten did
    not take, and a ``.conv`` on a first pass would mean ``None`` became a leaf.
    """
    *_, cache = a_cache(continued)
    paths = [
        pytree.keystr(path) for path, _ in pytree.tree_flatten_with_path(cache)[0]
    ]

    trajectory = [f".trajectory.{name}" for name in ("unfolded", "u", "k_t", "gamma", "rel")]
    conv = [f".conv[{index}]" for index in range(3)] if continued else []
    assert paths == trajectory + conv


@needs_registration
@pytest.mark.parametrize("continued", PASSES)
def test_vmapped_rereads_of_vmapped_writes_agree_with_a_loop(continued: bool) -> None:
    """`SETS` writers return `SETS` caches; `SETS` re-readers take them back.

    The write and the re-read in separate ``vmap`` calls, which is the
    arrangement that needs the registration: a cache made and consumed inside
    one call never crosses the boundary, and one shared cache crosses it
    unbatched.  Compared against the layers run one at a time, so a transform
    that broadcast one set's cache across all of them would not pass.
    """
    layers = [a_mixer() for _ in range(SETS)]
    params, buffers = stack_module_state(layers)
    base = a_mixer().to("meta")
    rereading = Rereading(base)

    def prefixed(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {f"layer.{name}": tensor for name, tensor in tensors.items()}

    def advance(parameters, buffered, x):
        return functional_call(base, (parameters, buffered), (x,), {"return_state": True})[1]

    def write(parameters, buffered, x, state):
        kwargs = {"state": state, "return_cache": True}
        return functional_call(base, (parameters, buffered), (x,), kwargs)[1]

    def reread(parameters, buffered, x, cache):
        weights = (prefixed(parameters), prefixed(buffered))
        return functional_call(rereading, weights, (x, cache))

    prefix = torch.randn(SETS, BATCH, SEQ, D_MODEL)
    x = torch.randn(SETS, BATCH, SEQ, D_MODEL)
    fresh = torch.randn(SETS, BATCH, SEQ, D_MODEL)

    state = vmap(advance)(params, buffers, prefix) if continued else None
    # A first pass has no state to map over, and `vmap` wants that said.
    in_dims = (0, 0, 0, 0 if continued else None)
    caches = vmap(write, in_dims=in_dims)(params, buffers, x, state)

    assert isinstance(caches, GatedDeltaNetReadCache)
    assert caches.trajectory.seq_len == SEQ
    assert (caches.conv is not None) == continued
    assert all(leaf.shape[0] == SETS for leaf in pytree.tree_leaves(caches))

    batched = vmap(reread)(params, buffers, fresh, caches)

    looped = []
    for index, layer in enumerate(layers):
        own = layer(prefix[index], return_state=True)[1] if continued else None
        _, cache = layer(x[index], state=own, return_cache=True)
        looped.append(layer.reread(fresh[index], cache))

    torch.testing.assert_close(batched, torch.stack(looped), rtol=1e-5, atol=1e-5)
