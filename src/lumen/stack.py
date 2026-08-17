"""
A stack of residual blocks and a final norm — the trunk, and nothing above it.

`(B, T, d_model)` in, `(B, T, d_model)` out. No embedding, no LM head, no task
head: everything modality-specific is the consumer's, and keeping it out is what
makes a trained trunk transplantable between different bodies at all. That
transplant is a ``state_dict`` operation, which is why the checkpoint keys here
are inherited rather than chosen — see :mod:`lumen.block`.

Separate module from :mod:`lumen.block` on purpose. A caller may take
:class:`~lumen.block.Block` and not this, or neither; the mixers work standalone
and always did. This is the convenience for consumers who want the higher-level
object, and a consumer whose whole purpose is experimenting *at* block level is
expected to keep its own. What makes the two interoperate is the keys, not a
shared class.

**The stack owns initialisation, and that is the one thing it genuinely needs to
exist for.** A block cannot depth-scale itself — the scale depends on how many
blocks write to the stream — so this is the level where that policy can live at
all. It matters beyond convenience for any consumer that reconstructs weights
from a stored seed rather than loading them: there, the *draw order* is part of
the contract, not just the distribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable

import torch
import torch.nn as nn

from lumen.block import Block, BlockState
from lumen.nn import RMSNorm

__all__ = ["Stack", "StackState"]


#: The divisor's "two residual writes per block", inherited as a constant.
#:
#: A block's actual write count is 1, 2, 2 or 3 depending on whether it carries
#: a `local` sub-layer and an MLP, and a stack built from instances knows which
#: — :meth:`lumen.block.Block.residual_writes` reports it. Substituting the true
#: count changes every initial weight and nobody has compared the two, so the
#: constant stays and counting is an experiment: `latticedynamics/lumen#6`.
_WRITES_PER_BLOCK = 2


@dataclass(frozen=True)
class StackState:
    """One :class:`~lumen.block.BlockState` per block, in stack order.

    A tuple rather than a list, and frozen rather than mutable, because forking
    a stream must not leave two branches sharing anything. The lineage this was
    extracted from accumulated a bare ``list`` and returned it; that is harmless
    under a single training loop and a silent correctness bug anywhere a prefix
    is continued down two different futures.
    """

    blocks: tuple[BlockState, ...] = ()


class Stack(nn.Module):
    """``n_layers`` residual blocks and a final norm.

    Example::

        def layer(index: int) -> Block:
            return Block(512, GatedDeltaNet(config), norm_eps=1e-5, d_mlp=0)

        trunk = Stack(512, 8, layer, norm_eps=1e-5)
        y = trunk(x)                      # (B, T, d_model) -> (B, T, d_model)

    **Blocks come from a factory, not a config.** ``block(i)`` is called once per
    layer, so heterogeneity — this layer gets a mixer, that one gets a plain
    :class:`~lumen.nn.SwiGLU` where its mixer would be — is a two-line lambda
    rather than anything this class supports. A factory rather than a prebuilt
    sequence because ``n_layers`` has to exist *before* any block does: the
    depth-scaled init needs it, and under a sequence the caller would have to
    know it too, putting the same number in two places.

    Args:
        d_model:  Residual stream width.
        n_layers: How many blocks, and the depth the init scales by.
        block:    ``block(index) -> Block``. Called once per layer, in order.
        norm_eps: Epsilon for the final norm. Required, no default — see
                  :class:`~lumen.block.Block` and `latticedynamics/lumen#6`.
        init_std: Base standard deviation for every ``nn.Linear`` weight in the
                  trunk. Residual writes are then rescaled by
                  ``1/sqrt(2·n_layers)`` on top of it.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        block: Callable[[int], Block],
        *,
        norm_eps: float,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be at least 1, got {n_layers}")

        self.d_model = d_model
        self.init_std = init_std

        self.blocks = nn.ModuleList(block(index) for index in range(n_layers))
        self.norm_f = RMSNorm(d_model, norm_eps)

        self.reset_parameters()

    @property
    def n_layers(self) -> int:
        return len(self.blocks)

    # ── initialisation ────────────────────────────────────────────────────

    def reset_parameters(self) -> None:
        """Both passes: base, then the depth-scaled rescale of residual writes.

        Called from ``__init__``, so a `Stack` is fully initialised the moment it
        exists and a consumer that constructs its trunk *first* gets a stable
        prefix of the RNG draw sequence regardless of what is built around it.
        That ordering rule cannot be enforced from here, only stated; it is what
        makes "reconstruct this model from its seed" well defined without this
        class having to own a generator.

        **Biases are never touched.** The lineage decided this by reading values
        — it zeroed a bias only if the bias was already zero, which is a
        value-dependent branch that happens to preserve deliberate settings.
        Not touching them at all is equivalent for every layer in this package
        (the only biases are a gated delta net's ``a_proj``, deliberately −3.0,
        and ``b_proj``, deliberately zero) and it removes a branch whose
        correctness depended on "deliberate biases are nonzero" staying true.

        **The depth pass runs second and therefore wins**, including over a
        sub-layer that deliberately initialised its own output projection —
        ``UndertowConfig(zero_init=True)`` being the case that exists. That is
        the documented precedence rather than an accident: the alternative
        available shortcut, skipping projections that are currently all-zero,
        would make initialisation depend on parameter *values*, and the seed
        contract above needs it to depend only on the seed. A caller splicing a
        zero-initialised layer into a trained trunk is not constructing a fresh
        `Stack` and never meets this.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=self.init_std)

        depth_std = self.init_std / math.sqrt(_WRITES_PER_BLOCK * self.n_layers)
        for projection in self.residual_out_projections():
            nn.init.normal_(projection.weight, std=depth_std)

    def residual_out_projections(self) -> tuple[nn.Module, ...]:
        """Every projection in the trunk whose output reaches the stream.

        Asks its blocks, which ask their sub-layers. Nothing here matches a
        parameter name: a caller that swept ``named_parameters()`` for an
        ``o_proj.weight`` suffix would silently initialise one tensor fewer the
        day somebody renamed an attribute, and produce a model that trains and
        is wrong.

        Each returned module must expose ``.weight``, which is what "projection"
        means here.
        """
        writes: list[nn.Module] = []
        for block in self.blocks:
            writes.extend(block.residual_out_projections())
        return tuple(writes)

    def residual_writes(self) -> tuple[int, ...]:
        """Per-block count of additions to the residual stream.

        Reported, not used: :data:`_WRITES_PER_BLOCK` is what the init actually
        divides by. The two disagree for any block that is not exactly a mixer
        plus an MLP, and closing that gap is `latticedynamics/lumen#6`.
        """
        return tuple(block.residual_writes() for block in self.blocks)

    # ── forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        *,
        state: StackState | None = None,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, StackState]:
        """``(B, T, d_model)`` → ``(B, T, d_model)``, optionally continuing a stream."""
        prior = state.blocks if state is not None else (None,) * self.n_layers
        if len(prior) != self.n_layers:
            raise ValueError(
                f"state holds {len(prior)} block states but this stack has "
                f"{self.n_layers} blocks"
            )

        successors: list[BlockState] = []
        for block, block_state in zip(self.blocks, prior):
            if return_state:
                x, successor = block(x, state=block_state, return_state=True)
                successors.append(successor)
            else:
                x = block(x, state=block_state)

        x = self.norm_f(x)
        if not return_state:
            return x
        return x, StackState(blocks=tuple(successors))

    # ── streaming ─────────────────────────────────────────────────────────

    def init_state(
        self,
        batch: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> StackState:
        """An empty stream for `batch` independent sequences.

        ``batch`` is a real degree of freedom here, not a decode convenience: a
        consumer stepping many independent streams in lockstep uses this as its
        primary inference path, at sizes nothing else in the library exercises.
        """
        return StackState(
            blocks=tuple(
                block.init_state(batch, device, dtype) for block in self.blocks
            )
        )

    def step(
        self, x: torch.Tensor, state: StackState
    ) -> tuple[torch.Tensor, StackState]:
        """One position — ``(B, 1, d_model)`` → output and the successor state."""
        if x.shape[1] != 1:
            raise ValueError(
                f"step() consumes one position at a time, got {x.shape[1]}; "
                f"use forward(x, state=..., return_state=True) for a chunk"
            )

        successors: list[BlockState] = []
        for block, block_state in zip(self.blocks, state.blocks):
            x, successor = block.step(x, block_state)
            successors.append(successor)

        return self.norm_f(x), StackState(blocks=tuple(successors))

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_layers={self.n_layers}, "
            f"init_std={self.init_std}, residual_writes={self.residual_writes()}"
        )
