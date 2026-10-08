"""Sparse Delta Memory — a gated delta rule over a large, sparsely addressed table.

Per head, a table of `N` slots.  Each position writes `W` of them and reads `R`,
chosen by product keys, so per-token arithmetic is `O((W+R)·d_v)` whatever
`N` is, and a slot nobody writes to is frozen — decay included.  State size is
decoupled from parameter count and from compute — except that ``step`` copies
the table per stream unless its state is donated; see
:class:`SparseDeltaMemoryState`.

Public surface is the layer, its config and its state::

    from lumen.sdm import SparseDeltaMemory, SparseDeltaMemoryConfig

    config = SparseDeltaMemoryConfig(
        d_model=512, n_heads=2, n_slots=128**2, initial_memory="learned"
    )
    mixer = SparseDeltaMemory(config)

``initial_memory`` has no default: ``"zero"`` and ``"learned"`` are different
models, and choosing between them is the caller's decision.

The kernels in :mod:`lumen.sdm.reference` are not exported.  Reuse is by
subclassing — override ``_address``, ``_features``, ``_scan`` or ``_out`` — and
tests import the kernels by path.
"""

from __future__ import annotations

from lumen.sdm.layer import (
    SparseDeltaMemory,
    SparseDeltaMemoryConfig,
    SparseDeltaMemoryState,
)

__all__ = [
    "SparseDeltaMemory",
    "SparseDeltaMemoryConfig",
    "SparseDeltaMemoryState",
]
