"""
Registering the state containers so ``torch`` transforms can walk them.

The mixers' contract is ``forward`` / ``init_state`` / ``step``, and **the state
object is half of it** — two of those three methods exist only to produce and
consume one.  A caller can hold either mixer without knowing which, which is the
property this library exists to protect, right up until it hands a state to
:func:`torch.func.vmap`, :func:`~torch.func.functional_call` or anything that
flattens across a :func:`torch.compile` boundary.  There the abstraction stops,
because a pytree that has never been declared is a leaf, and a leaf that is not
a tensor is rejected.

Nothing about these four classes is hostile to registration.  They are frozen
dataclasses holding tensors, tuples of tensors, ``None``, and one ``int``.  The
declaration is the whole fix.  See `latticedynamics/lumen#12`.

**This is additive and numerically inert.**  No tensor changes value, no
checkpoint key moves, no existing call path behaves differently.  What is
mutated lives in a registry inside ``torch``, not in anything here.

Why registration happens at import
----------------------------------

Because the failure it prevents is silent and remote from its cause.  A consumer
who forgot an explicit opt-in does not get a message about the opt-in; it gets an
``in_dim=0`` rejection from inside a recurrence, naming a container it did not
know was involved.  The cost of the alternative is real and worth stating rather
than waving: ``import lumen`` mutates a ``torch`` registry for the whole process.
That is a larger commitment than this package usually makes, and it is made here
because the states are Lumen's own classes — no other library can be holding a
registration for them to collide with.

Why three registrations are public and one is not
-------------------------------------------------

:func:`torch.export.register_dataclass` handles the case that looks hardest.
``GatedDeltaNetState.conv`` is ``None`` on a layer with no convolution and
``BlockState.local`` is ``None`` on a block with no local sub-layer; it drops
absent fields from the children and records their absence in the context, which
is exactly what a hand-rolled flatten gets wrong on the first attempt.

It cannot express the case that looks easy.  ``UndertowState.seen`` is present,
permanent, and not a tensor, so it becomes a child and ``vmap`` refuses it for
the same reason it refuses ``None``.  ``seen`` is not data: it drives Python
control flow — the valid-position count, the front-padding width, the slice
bounds — and it describes the tree rather than living in it.  That is what a
treespec context is for, and ``register_dataclass`` has no way to put a field
there.

``drop_field_names`` is the public-looking escape and is a worse trap than the
one it solves: flattening succeeds and unflattening raises, because a dropped
field is reconstructed from nothing.  **The repair that suggests itself next is
the one to refuse loudest** — giving ``seen`` a default makes every round-tripped
state rebuild as ``seen=0``, and a window state that has forgotten its count
masks the wrong slots and returns numbers instead of an error.

So one class is registered against ``torch.utils._pytree``, which is private.
The degradation discipline covers it on the same terms as everything else here:
attempt it, skip it if the surface moved, and change nothing that works today.
A skipped registration costs the ``torch.func`` path, which is unavailable
anyway on a version that lacks the API.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

import torch

from lumen.block import BlockState
from lumen.gdn import GatedDeltaNetState
from lumen.stack import StackState
from lumen.undertow import UndertowState

__all__ = ["REGISTERED", "register_state_pytrees"]


# Registered classes, so a second call is a no-op rather than an error.  The two
# torch entry points disagree about repetition -- `register_dataclass` overwrites
# with a UserWarning, `register_pytree_node` raises -- and neither is a thing a
# caller should have to know.  Tracking here makes both idempotent and quiet.
_REGISTERED: set[type] = set()


def _flatten_undertow(
    state: UndertowState,
) -> tuple[list[torch.Tensor], int]:
    """Two tensors as children, the count as context."""
    return [state.keys, state.values], state.seen


def _unflatten_undertow(
    children: Iterable[torch.Tensor], seen: int
) -> UndertowState:
    keys, values = children
    return UndertowState(keys=keys, values=values, seen=seen)


def _register_dataclass(cls: type) -> None:
    from torch.export import register_dataclass

    register_dataclass(cls)


def _register_undertow(cls: type) -> None:
    """Register with paths if this ``torch`` has them, without if it does not.

    The keyed flatten is what makes ``tree_map_with_path`` treat all four states
    alike; omitting it leaves the path-carrying traversals working on the three
    dataclass-registered states and failing on this one, which is the asymmetry
    between mixers this module exists to remove.

    It is still worth falling back rather than failing.  Paths are the smaller
    half of what registration buys -- ``vmap`` and ``tree_map`` never ask for
    them -- so a ``torch`` without the keyed variant should cost the paths, not
    the whole state.  Resolved eagerly rather than inside the callback, so an
    absent name is a registration that degrades here instead of a traversal that
    raises much later.
    """
    from torch.utils._pytree import register_pytree_node

    try:
        from torch.utils._pytree import GetAttrKey
    except ImportError:
        register_pytree_node(cls, _flatten_undertow, _unflatten_undertow)
        return

    def flatten_with_keys(
        state: UndertowState,
    ) -> tuple[list[tuple[Any, torch.Tensor]], int]:
        return [
            (GetAttrKey("keys"), state.keys),
            (GetAttrKey("values"), state.values),
        ], state.seen

    try:
        register_pytree_node(
            cls,
            _flatten_undertow,
            _unflatten_undertow,
            flatten_with_keys_fn=flatten_with_keys,
        )
    except TypeError:
        # The keyword is not accepted on this version, so nothing was registered
        # -- binding fails before any work.  Retrying without it is safe.
        register_pytree_node(cls, _flatten_undertow, _unflatten_undertow)


#: Each state and the entry point that can express its shape.  Enumerated in one
#: place rather than declared beside each class: the set is no longer a uniform
#: loop, and the asymmetry is the thing most worth keeping visible.
_STATES: tuple[tuple[type, Callable[[type], None]], ...] = (
    (GatedDeltaNetState, _register_dataclass),
    (BlockState, _register_dataclass),
    (StackState, _register_dataclass),
    (UndertowState, _register_undertow),
)


def register_state_pytrees() -> tuple[str, ...]:
    """Declare the state containers as pytree nodes.  Idempotent.

    Returns:
        The names of every state that is registered once this call returns,
        in declaration order.  A short tuple is a partial registration and a
        working library: the states behave exactly as they did before, and only
        the ``torch.func`` path on the missing ones is unavailable.

    Failures are captured rather than raised, matching :mod:`lumen.probe` and
    for the same reason -- one absent API in ``torch`` should be a shorter tuple
    here, not an ``ImportError`` out of ``import lumen``.
    """
    for cls, register in _STATES:
        if cls in _REGISTERED:
            continue
        try:
            register(cls)
        except Exception:  # noqa: BLE001 -- an absent API is data, not a crash
            continue
        _REGISTERED.add(cls)

    return tuple(cls.__name__ for cls, _ in _STATES if cls in _REGISTERED)


#: What registration achieved on this ``torch``, evaluated once at import.
#:
#: Four names is the whole set.  Anything shorter names the states a
#: ``torch.func`` transform will still refuse, which is the only question a
#: consumer needs answered and is cheaper to read than to rediscover from a
#: ``vmap`` error.
REGISTERED: tuple[str, ...] = register_state_pytrees()
