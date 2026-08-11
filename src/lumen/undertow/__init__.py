"""Undertow — fixed-window causal attention, no positional encoding.

Public surface is the layer and its config::

    from lumen import UndertowAttention, UndertowConfig

    attn = UndertowAttention(UndertowConfig(
        d_model=384, n_heads=8, window=32, plateau=24,
    ))
    y = attn(x)

The kernels in :mod:`lumen.undertow.reference` are deliberately not exported.
They are seams for subclasses and the substrate the test suite checks against,
not an API to build on — reuse here happens by subclassing the layer, which is
the pattern that has already survived a real downstream architecture.
"""

from __future__ import annotations

from lumen.undertow.layer import UndertowAttention, UndertowConfig, UndertowState

__all__ = ["UndertowAttention", "UndertowConfig", "UndertowState"]
