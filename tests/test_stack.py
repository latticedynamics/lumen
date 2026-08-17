"""Stack — the trunk, and the initialisation policy that is its whole reason.

The load-bearing ones are :func:`test_streaming_agrees_with_parallel`,
:func:`test_two_stacks_under_one_seed_are_identical` and
:func:`test_deliberate_biases_survive_initialisation`. The first says the two
paths agree; the second is the contract a consumer reconstructing weights from a
stored seed depends on; the third says the simplification this class makes to
the lineage's init is actually equivalent rather than merely tidier.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch
import torch.nn as nn

from lumen.block import Block, BlockState
from lumen.gdn import GatedDeltaNet, GatedDeltaNetConfig, HeadLayout
from lumen.nn import SwiGLU
from lumen.stack import Stack, StackState
from lumen.undertow import UndertowAttention, UndertowConfig

D_MODEL = 64
EXACT = 1e-6


def a_mixer() -> GatedDeltaNet:
    return GatedDeltaNet(
        GatedDeltaNetConfig(
            d_model=D_MODEL, layout=HeadLayout.shared_key(4), expand_k=2.0
        )
    )


def a_local() -> UndertowAttention:
    return UndertowAttention(UndertowConfig(d_model=D_MODEL, n_heads=4, window=8))


def plain(d_mlp: int = 0, local: bool = False):
    def factory(index: int) -> Block:
        return Block(
            D_MODEL,
            a_mixer(),
            norm_eps=1e-5,
            d_mlp=d_mlp,
            local=a_local() if local else None,
        )

    return factory


def build(n_layers: int = 3, seed: int = 0, **kwargs) -> Stack:
    torch.manual_seed(seed)
    return Stack(D_MODEL, n_layers, plain(**kwargs), norm_eps=1e-5).eval()


# ── construction ──────────────────────────────────────────────────────────


def test_shape_is_preserved():
    x = torch.randn(2, 12, D_MODEL)
    assert build()(x).shape == x.shape


def test_rejects_an_empty_stack():
    with pytest.raises(ValueError, match="at least 1"):
        Stack(D_MODEL, 0, plain(), norm_eps=1e-5)


def test_norm_eps_has_no_default():
    with pytest.raises(TypeError):
        Stack(D_MODEL, 2, plain())  # type: ignore[call-arg]


def test_checkpoint_keys_are_the_inherited_ones():
    """``blocks.{i}.*`` and ``norm_f.weight``, which is what a transplant needs.

    The trunk is carried between different bodies by loading a state dict, and
    that operation does not degrade under a naming difference — it either raises
    or partially matches and quietly loads a subset.
    """
    keys = set(build(n_layers=2).state_dict())
    assert "norm_f.weight" in keys
    assert all(k.startswith(("blocks.0.", "blocks.1.", "norm_f.")) for k in keys)


def test_the_factory_is_called_once_per_index_in_order():
    seen: list[int] = []

    def factory(index: int) -> Block:
        seen.append(index)
        return Block(D_MODEL, a_mixer(), norm_eps=1e-5, d_mlp=0)

    Stack(D_MODEL, 4, factory, norm_eps=1e-5)
    assert seen == [0, 1, 2, 3]


def test_heterogeneity_needs_no_support_from_this_class():
    """One layer gets a stateless SwiGLU where its mixer would be.

    A two-line factory, and the stack has no branch for it. This is the whole
    argument for taking instances rather than a config: the arrangement a
    consumer wants to experiment with is something it can already express.
    """
    torch.manual_seed(0)

    def factory(index: int) -> Block:
        mixer = SwiGLU(D_MODEL, 128) if index == 1 else a_mixer()
        return Block(D_MODEL, mixer, norm_eps=1e-5, d_mlp=0)

    stack = Stack(D_MODEL, 3, factory, norm_eps=1e-5).eval()
    x = torch.randn(2, 12, D_MODEL)
    assert stack(x).shape == x.shape
    assert stack.init_state(2).blocks[1].mixer is None


def test_the_final_norm_is_applied():
    """It is the stack's, not the last block's, and it must actually run."""
    stack = build(n_layers=1)
    x = torch.randn(2, 8, D_MODEL)
    with torch.no_grad():
        through_blocks = stack.blocks[0](x)
        assert torch.equal(stack(x), stack.norm_f(through_blocks))


# ── initialisation ────────────────────────────────────────────────────────


def assert_drawn_at(weight: torch.Tensor, target: float, name: str) -> None:
    """Compare a sample standard deviation against its own sampling error.

    A flat relative tolerance is wrong here and fails for the right reason: the
    package's smallest projections are ``Linear(d_model, 1)`` — a gated delta
    net's decay and write-strength heads — so their weights hold ``d_model``
    elements and a sample std is off by ``σ/sqrt(2n)`` from sampling alone. At
    ``n = 64`` that is 9% at one sigma, which a 15% bound calls a bug.
    """
    n = weight.numel()
    sigma = target / math.sqrt(2 * n)
    assert abs(weight.std().item() - target) < 5 * sigma, (
        f"{name}: std {weight.std().item():.5f} is more than 5 sampling sigmas "
        f"({sigma:.5f}) from {target:.5f} over {n} elements"
    )


def test_residual_writes_get_the_depth_scaled_deviation():
    """``init_std / sqrt(2·n_layers)`` on the writes, ``init_std`` elsewhere."""
    n_layers = 6
    stack = build(n_layers=n_layers, d_mlp=128)
    expected = 0.02 / math.sqrt(2 * n_layers)

    written = {id(p) for proj in stack.residual_out_projections() for p in proj.parameters()}
    for projection in stack.residual_out_projections():
        assert_drawn_at(projection.weight, expected, "a residual write")

    others = [
        (name, module.weight)
        for name, module in stack.named_modules()
        if isinstance(module, nn.Linear) and id(module.weight) not in written
    ]
    assert others, "probe is invalid: every Linear is a residual write"
    for name, weight in others:
        assert_drawn_at(weight, 0.02, name)


def test_deliberate_biases_survive_initialisation():
    """The simplification to the lineage's init, checked rather than assumed.

    That init zeroed a bias only when the bias was already zero — a
    value-dependent branch that happens to preserve deliberate settings. This
    class does not touch biases at all, which is equivalent *provided* every
    bias in the package is deliberate. It is: a gated delta net's ``a_proj``
    (−3.0, so the layer starts by forgetting slowly) and ``b_proj`` (zero).
    This test fails if a biased projection with an ordinary default ever
    appears, which is exactly when the two policies would stop agreeing.
    """
    stack = build(n_layers=2)
    biases = {
        name: parameter
        for name, parameter in stack.named_parameters()
        if name.endswith("bias")
    }
    assert biases, "probe is invalid: nothing in the stack carries a bias"
    for name, parameter in biases.items():
        if name.endswith("a_proj.bias"):
            assert torch.equal(parameter, torch.full_like(parameter, -3.0))
        elif name.endswith("b_proj.bias"):
            assert torch.equal(parameter, torch.zeros_like(parameter))
        else:  # pragma: no cover - the guard, not the case
            pytest.fail(f"unexpected bias {name}: the init policy needs revisiting")


def test_two_stacks_under_one_seed_are_identical():
    """The contract for reconstructing weights from a stored seed.

    Not "the same distribution" — bit-identical, because a consumer that stores
    an individual as its seed and rebuilds on demand needs the draw order to be
    part of the contract too.
    """
    a, b = build(seed=1234), build(seed=1234)
    for (name, left), (_, right) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.equal(left, right), name


def test_reset_parameters_is_re_callable():
    stack = build(seed=7)
    before = {k: v.clone() for k, v in stack.state_dict().items()}
    torch.manual_seed(99)
    stack.reset_parameters()
    assert any(not torch.equal(before[k], v) for k, v in stack.state_dict().items())


def test_the_depth_pass_runs_second_and_wins_over_zero_init():
    """Documented precedence, pinned so it cannot change silently.

    ``zero_init`` and the depth sweep want opposite things from the same tensor.
    The sweep runs last, so it wins. The alternative — skipping projections that
    are currently all-zero — is refused because it would make initialisation
    depend on parameter *values*, and the seed contract above needs it to depend
    only on the seed.
    """
    def factory(index: int) -> Block:
        local = UndertowAttention(
            UndertowConfig(d_model=D_MODEL, n_heads=4, window=8, zero_init=True)
        )
        return Block(D_MODEL, a_mixer(), norm_eps=1e-5, d_mlp=0, local=local)

    torch.manual_seed(0)
    stack = Stack(D_MODEL, 2, factory, norm_eps=1e-5)
    assert stack.blocks[0].local.o_proj.weight.abs().max() > 0


def test_residual_out_projections_reach_every_block():
    stack = build(n_layers=3, d_mlp=128, local=True)
    declared = {
        id(projection) for projection in stack.residual_out_projections()
    }
    swept = {
        id(module)
        for name, module in stack.named_modules()
        if name.endswith(("o_proj", "down_proj"))
    }
    assert declared and declared == swept


def test_residual_writes_is_reported_per_block_and_is_not_what_the_init_uses():
    """Two quantities that disagree, and the disagreement is `lumen#6` row 3."""
    stack = build(n_layers=3, d_mlp=128, local=True)
    assert stack.residual_writes() == (3, 3, 3)
    # The init divides by 2 per block regardless.  If that ever stops being
    # true, this test is where the change announces itself.
    assert_drawn_at(
        stack.residual_out_projections()[0].weight,
        0.02 / math.sqrt(2 * 3),
        "a residual write in a three-write block",
    )


# ── streaming ─────────────────────────────────────────────────────────────


def test_stack_state_is_frozen_and_holds_a_tuple():
    state = build(n_layers=2).init_state(1)
    assert isinstance(state.blocks, tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.blocks = ()  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [dict(), dict(d_mlp=128), dict(local=True), dict(d_mlp=128, local=True)],
    ids=["mixer", "mixer+mlp", "local+mixer", "local+mixer+mlp"],
)
def test_streaming_agrees_with_parallel(kwargs):
    """Prefill a prefix, step the rest, land where one pass lands.

    Nothing above the mixers tested this before. It is what catches a state
    threaded to the wrong block — outputs keep their shape and only the history
    is wrong.
    """
    stack = build(n_layers=3, **kwargs)
    x = torch.randn(2, 10, D_MODEL)

    whole = stack(x)
    split = 6
    prefix, state = stack(x[:, :split], return_state=True)
    outputs = [prefix]
    for t in range(split, x.shape[1]):
        y, state = stack.step(x[:, t : t + 1], state)
        outputs.append(y)

    # The final norm is applied per call, so the prefix rows are already
    # normalised and directly comparable position by position.
    assert (torch.cat(outputs, dim=1) - whole).abs().max() < EXACT


def test_a_batch_of_many_independent_streams():
    """``init_state(N)`` at N far above 1 — a primary path, not decode sugar.

    Checked against N separate streams of 1 rather than against itself, so it
    fails if the batch dimension is ever entangled across streams.
    """
    stack = build(n_layers=2)
    n = 32
    x = torch.randn(n, 4, D_MODEL)

    batched = stack(x)
    apart = torch.cat([stack(x[i : i + 1]) for i in range(n)], dim=0)
    assert (batched - apart).abs().max() < EXACT


def test_step_returns_a_successor_rather_than_mutating():
    stack = build(n_layers=2)
    x = torch.randn(2, 1, D_MODEL)
    state = stack.init_state(2)

    a, after_a = stack.step(x, state)
    b, _ = stack.step(x, state)

    assert torch.equal(a, b), "the shared prior state was mutated by the first step"
    assert after_a is not state


def test_step_refuses_more_than_one_position():
    stack = build(n_layers=2)
    with pytest.raises(ValueError, match="one position at a time"):
        stack.step(torch.randn(2, 3, D_MODEL), stack.init_state(2))


def test_a_state_from_the_wrong_stack_is_refused():
    """Loudly, rather than zipping short and silently skipping blocks."""
    shallow, deep = build(n_layers=2), build(n_layers=4)
    with pytest.raises(ValueError, match="block states"):
        deep(torch.randn(2, 4, D_MODEL), state=shallow.init_state(2))


def test_init_state_follows_the_module():
    stack = build(n_layers=2)
    state = stack.init_state(3)
    assert len(state.blocks) == 2
    assert all(isinstance(block, BlockState) for block in state.blocks)
    assert state.blocks[0].mixer.memory.shape[0] == 3
