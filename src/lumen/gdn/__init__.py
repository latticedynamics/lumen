"""Gated DeltaNet — a fixed-size associative memory with a delta-rule write.

Public surface is the layer, its config, and the head layout::

    from lumen.gdn import GatedDeltaNet, GatedDeltaNetConfig, HeadLayout

    config = GatedDeltaNetConfig(
        d_model=512, layout=HeadLayout.shared_key(8), expand_k=2.0
    )
    mixer = GatedDeltaNet(config)

The kernels in :mod:`lumen.gdn.reference` are not exported.  Reuse is by
subclassing — override ``_features``, ``_scan`` or ``_out`` — and tests import
the kernels by path.
"""

from __future__ import annotations

from lumen.gdn.layer import (
    GatedDeltaNet,
    GatedDeltaNetConfig,
    GatedDeltaNetState,
    ShortConv,
)
from lumen.gdn.layout import HeadLayout

__all__ = [
    "GatedDeltaNet",
    "GatedDeltaNetConfig",
    "GatedDeltaNetState",
    "HeadLayout",
    "ShortConv",
]
