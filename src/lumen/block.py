"""
The residual block — where a result is actually read off.

Nothing in this module is novel, and that is exactly why it is here. A number
is never read off a mixer; it is read off a stack of these, and the block is
where norm placement, the residual arrangement and the depth-scaled init all
live. Every one of those sits between a shared mixer and every measured
quantity, so two projects can share `GatedDeltaNet` exactly, run the same
configuration, and still not be comparable.

**A block holds sub-layer instances, not a config.** It does not know how to
build a mixer and cannot be asked to. The cost is that a stack is not
reconstructible from a dataclass alone; the benefit is that heterogeneity —
this layer gets a mixer, that one gets a plain :class:`~lumen.nn.SwiGLU` where
its mixer would be — needs no support here at all. It is something the caller
already can do.

**Taking a `Block` does not commit a caller to taking a `Stack`, and taking
neither costs nothing.** The mixers work standalone and always did. This is a
convenience for consumers who want the higher-level object; a consumer whose
whole purpose is experimenting *at* block level is expected to keep its own,
and the thing that makes the two interoperate is the checkpoint keys below, not
a shared class.

**A sub-layer may be stateless.** Statefulness is read from the object — a
sub-layer is stateful if it has ``init_state`` — rather than declared, because
that is what lets a stateless :class:`~lumen.nn.SwiGLU` sit in the mixer slot
without a flag saying so.

**The attribute names are checkpoint keys.** ``local``, ``mixer``, ``mlp``,
``norm_local``, ``norm``, ``norm2`` are inherited from the lineage this was
extracted from, and the inconsistency among the norm names (one role-based, one
positional, one bare) is inherited deliberately: every tidying is a permanent
remap entry against archived runs, bought with nothing measurable.

``local`` in particular is a generic slot with a role-shaped name. It holds any
sub-layer with its own norm and residual, placed before the mixer. The name
records the arrangement that motivated it — one sub-layer resolves recent
positions exactly, the other carries the whole prefix through a fixed state, and
local detail is cheap to get right and expensive to store, so it is spent before
the state rather than after.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from lumen.nn import RMSNorm, SwiGLU

__all__ = ["Block", "BlockState"]


@dataclass(frozen=True)
class BlockState:
    """The streaming state of one block: one slot per stateful sub-layer.

    Frozen, and holding frozen sub-states, because forking a stream must not
    leave two branches sharing a buffer. That is a correctness property rather
    than tidiness for any caller that continues the same prefix down two
    different futures — and it is the reason this is a dataclass rather than the
    bare tuple the lineage threaded.

    ``None`` in a slot means the corresponding sub-layer is absent or stateless.
    It never means "a zeroed state": the mixers require a real state on ``step``
    and treat ``None`` as a missing argument, which is why :meth:`Block.init_state`
    exists rather than every caller knowing this block's shape.
    """

    local: Any = None
    mixer: Any = None


def _is_stateful(module: nn.Module | None) -> bool:
    """Read statefulness off the object rather than off a flag.

    A sub-layer that can build a state can carry one. This is duck typing on
    purpose: it is what lets a stateless :class:`~lumen.nn.SwiGLU` occupy the
    mixer slot without the block growing a branch for it.
    """
    return module is not None and hasattr(module, "init_state")


def _residual_writes(module: nn.Module) -> tuple[nn.Module, ...]:
    """Ask a sub-layer where it writes, and refuse to guess if it cannot say.

    Loud rather than quiet: a sub-layer with parameters that does not implement
    the contract would otherwise be silently skipped by a depth-scaled init,
    producing a model that trains and is wrong. There is no correct default —
    guessing an attribute name is exactly the fragile mechanism this contract
    replaces.
    """
    getter = getattr(module, "residual_out_projections", None)
    if getter is None:
        raise TypeError(
            f"{type(module).__name__} does not implement "
            "residual_out_projections(), so a caller cannot know which of its "
            "projections write to the residual stream. Implement it (return a "
            "tuple of the output projections) or keep this sub-layer out of a "
            "Block."
        )
    return tuple(getter())


class Block(nn.Module):
    """Pre-norm residual sub-layers over a ``(B, T, d_model)`` stream.

    Up to three, in order, each with its own norm and its own residual::

        x = x + local(norm_local(x))     # only if `local` was given
        x = x + mixer(norm(x))
        x = x + mlp(norm2(x))            # only if `d_mlp > 0`

    Example::

        mixer = GatedDeltaNet(GatedDeltaNetConfig(d_model=512, layout=layout))
        block = Block(512, mixer, norm_eps=1e-5, d_mlp=0)
        y = block(x)                      # (B, T, d_model) -> (B, T, d_model)

    Holds the same streaming contract the mixers do — ``forward(x, *, state=,
    return_state=)``, :meth:`init_state`, :meth:`step` — and calls its sub-layers
    in that form itself, which is why holding one requires no adapter.

    Args:
        d_model:  Residual stream width.
        mixer:    The main sub-layer. Any module taking and returning
                  ``(B, T, d_model)``; stateful ones additionally hold the house
                  streaming contract. A stateless module here is legal and is
                  how a layer trades its state for a plain feed-forward.
        norm_eps: Epsilon for all of this block's norms. **Required, no
                  default.** Two lineages disagree on the value for the mixers'
                  *head* norms, and the residual-stream norms here are a second,
                  independent knob that nobody has compared at all. Neither is
                  canonised — see `latticedynamics/lumen#6`.
        d_mlp:    Hidden width of the gated feed-forward sub-layer; ``0`` for
                  none. **Required, no default**, because the mixer's own output
                  path is already an up/gate/down arrangement, so a second one
                  is a claim about what the mixer is short of rather than a
                  neutral convention.
        local:    A sub-layer placed before the mixer, with its own norm and
                  residual. ``None`` for none.
    """

    def __init__(
        self,
        d_model: int,
        mixer: nn.Module,
        *,
        norm_eps: float,
        d_mlp: int,
        local: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if d_mlp < 0:
            raise ValueError(f"d_mlp must be non-negative, got {d_mlp}")

        self.d_model = d_model

        # Built in stream order, so the parameter registration order — and
        # therefore the RNG draw order under a seeded construction — follows the
        # order the block computes in. A caller reconstructing weights from a
        # stored seed depends on that being stable and obvious.
        if local is not None:
            self.norm_local = RMSNorm(d_model, norm_eps)
            self.local = local
        else:
            self.norm_local = None
            self.local = None

        self.norm = RMSNorm(d_model, norm_eps)
        self.mixer = mixer

        if d_mlp > 0:
            self.norm2 = RMSNorm(d_model, norm_eps)
            self.mlp = SwiGLU(d_model, d_mlp)
        else:
            self.norm2 = None
            self.mlp = None

    # ── the residual arrangement ──────────────────────────────────────────

    def _sub(
        self, module: nn.Module, x: torch.Tensor, state: Any, want_state: bool
    ) -> tuple[torch.Tensor, Any]:
        """One sub-layer, called in the house form, state threaded or not."""
        if not _is_stateful(module):
            return module(x), None
        if want_state:
            return module(x, state=state, return_state=True)
        return module(x, state=state), None

    def forward(
        self,
        x: torch.Tensor,
        *,
        state: BlockState | None = None,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, BlockState]:
        """``(B, T, d_model)`` → ``(B, T, d_model)``, optionally continuing a stream.

        Args:
            state: state to continue from. ``None`` starts each stateful
                sub-layer from its own beginning-of-sequence.
            return_state: also return the state after consuming ``x``, so a
                prefix can be consumed in one parallel pass and continued with
                :meth:`step`.
        """
        prior = state if state is not None else BlockState()

        s_local = None
        if self.local is not None:
            y, s_local = self._sub(
                self.local, self.norm_local(x), prior.local, return_state
            )
            x = x + y

        y, s_mixer = self._sub(self.mixer, self.norm(x), prior.mixer, return_state)
        x = x + y

        if self.mlp is not None:
            x = x + self.mlp(self.norm2(x))

        if not return_state:
            return x
        return x, BlockState(local=s_local, mixer=s_mixer)

    # ── streaming ─────────────────────────────────────────────────────────

    def init_state(
        self,
        batch: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> BlockState:
        """A zeroed state for this block, shaped like what :meth:`step` returns.

        The block owns this rather than the caller, so nothing outside has to
        know how many stateful sub-layers it happens to have. Device and dtype
        follow each sub-layer's own parameters unless overridden.
        """
        return BlockState(
            local=(
                self.local.init_state(batch, device, dtype)
                if _is_stateful(self.local)
                else None
            ),
            mixer=(
                self.mixer.init_state(batch, device, dtype)
                if _is_stateful(self.mixer)
                else None
            ),
        )

    def step(
        self, x: torch.Tensor, state: BlockState
    ) -> tuple[torch.Tensor, BlockState]:
        """One position — ``(B, 1, d_model)`` → output and the successor state."""
        if x.shape[1] != 1:
            raise ValueError(
                f"step() consumes one position at a time, got {x.shape[1]}; "
                f"use forward(x, state=..., return_state=True) for a chunk"
            )

        s_local = None
        if self.local is not None:
            if _is_stateful(self.local):
                y, s_local = self.local.step(self.norm_local(x), state.local)
            else:
                y = self.local(self.norm_local(x))
            x = x + y

        if _is_stateful(self.mixer):
            y, s_mixer = self.mixer.step(self.norm(x), state.mixer)
        else:
            y, s_mixer = self.mixer(self.norm(x)), None
        x = x + y

        if self.mlp is not None:
            x = x + self.mlp(self.norm2(x))

        return x, BlockState(local=s_local, mixer=s_mixer)

    # ── structure ─────────────────────────────────────────────────────────

    def residual_out_projections(self) -> tuple[nn.Module, ...]:
        """Every projection in this block whose output reaches the stream.

        The block answers for its sub-layers rather than making a caller walk
        into them, so the contract composes: a stack asks its blocks, a block
        asks its sub-layers, and a custom block with an arrangement nobody here
        anticipated overrides one method instead of being un-scalable.

        Raises:
            TypeError: if a sub-layer cannot say. Refusing is deliberate — a
                silently skipped write is a model that trains and is wrong.
        """
        writes: list[nn.Module] = []
        for module in (self.local, self.mixer, self.mlp):
            if module is not None:
                writes.extend(_residual_writes(module))
        return tuple(writes)

    def residual_writes(self) -> int:
        """How many times this block adds to the residual stream.

        One per sub-layer present, so 1 to 3 — and *not* the same quantity as
        ``len(residual_out_projections())``, which counts projections and would
        agree only by accident.

        Depth-scaled initialisation conventionally divides by ``sqrt(2·n_layers)``
        where the 2 stands for "two residual writes per block". That is exactly
        right for a block with a mixer and an MLP and wrong for the other three
        arrangements. Reported rather than used: substituting the true count
        changes every initial weight, and no comparison has been run —
        `latticedynamics/lumen#6`.
        """
        return sum(
            module is not None for module in (self.local, self.mixer, self.mlp)
        )

    def extra_repr(self) -> str:
        parts = [f"d_model={self.d_model}"]
        if self.local is not None:
            parts.append("local=True")
        if self.mlp is not None:
            parts.append(f"d_mlp={self.mlp.up.out_features}")
        parts.append(f"residual_writes={self.residual_writes()}")
        return ", ".join(parts)
