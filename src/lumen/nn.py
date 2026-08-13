"""
Small shared building blocks, in the sense of *already rebuilt more than once*.

Nothing here is novel.  That is the point: a consolidation library earns its
keep first on the components nobody would write a design record about, because
those are the ones each project quietly re-types with a slightly different cast
policy and a slightly different epsilon, and then compares numbers across.

**Why RMS normalisation is a function and a module rather than just a module.**
The layers in this package hold their normalisation weight as a bare
``nn.Parameter`` named ``head_norm``, not as a submodule.  That is not an
oversight and it cannot be tidied away: a submodule prefixes its parameters, so
``self.head_norm = RMSNorm(d)`` silently renames the checkpoint key from
``head_norm`` to ``head_norm.weight`` and every archived state dict stops
loading.  A cosmetic refactor that invalidates checkpoints is not cosmetic.

So the *arithmetic* is what gets shared -- :func:`rms_norm` -- and the module is
a thin convenience over it for consumers whose norm really is a standalone
layer, such as a pre-norm residual block.  One implementation, two shapes of
caller, and no checkpoint touched.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["RMSNorm", "rms_norm"]


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Root-mean-square normalise `x` over its last dimension, then scale.

    ``x * rsqrt(mean(x²) + eps) * weight``, with no mean subtraction -- which is
    the whole difference from layer normalisation, and the reason this is
    scale-invariant but not shift-invariant.  Callers rely on that: an RMS norm
    satisfies ``rms_norm(c·v) = sign(c)·rms_norm(v)`` for scalar `c`, which is
    what lets a scalar factor move across it to a cheaper position.

    `weight` broadcasts against the trailing dimensions, so a per-head layer
    holding one `(d_head,)` vector shared across heads and a plain `(d_model,)`
    block norm are the same call.

    **This function has no cast policy on purpose.**  It computes in whatever
    dtype it is handed.  The reduction genuinely wants fp32 -- a mean of squares
    underflows in fp16 at realistic activation scales -- but *where* the cast
    happens is a decision belonging to the caller, because in this package's
    mixers that cast is shared with the surrounding output path rather than
    owned by the norm.  :class:`RMSNorm` applies the usual policy for callers
    that have no opinion.

    Args:
        x:      Input; normalised over ``dim=-1``.
        weight: Elementwise scale, broadcast against `x`.  Cast to `x`'s dtype.
        eps:    Added inside the square root, before the reciprocal.

    Returns:
        A tensor shaped like `x`, in `x`'s dtype.
    """
    normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return normed * weight.to(normed.dtype)


class RMSNorm(nn.Module):
    """RMS normalisation as a standalone layer, computed in fp32.

    The fp32 promotion is the difference between this and calling
    :func:`rms_norm` directly, and it is here because a standalone norm is
    usually sitting on a residual stream that may be fp16 or bf16, where
    ``mean(x²)`` is the exact operation that underflows.  Input dtype is
    restored on the way out, so the layer is transparent to autocast.

    Args:
        dim: Width of the normalised (last) dimension.
        eps: Added inside the square root.  Note the default matches this
             package's gated delta net rather than its local attention, which
             uses ``1e-6`` -- the two lineages disagreed and no comparison has
             been run, so neither value is canonised here.  Pass it explicitly.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        return rms_norm(x.float(), self.weight, self.eps).to(dtype)

    def extra_repr(self) -> str:
        return f"dim={self.weight.numel()}, eps={self.eps}"
