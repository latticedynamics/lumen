"""Gated DeltaNet on `flash-linear-attention`'s Triton chunk kernels.

Not Lumen's kernels.  This module is a **shape and convention adapter** around
someone else's, and the name says so: the config value is ``backend="fla"``
rather than ``"triton"``.  Two reasons, and the second is the one that bites
later.

* Undertow's ``backend="triton"`` means *Lumen's own* Triton kernels, in
  :mod:`lumen.undertow.triton_kernels`.  If this were also ``"triton"`` then one
  word would name Lumen code in one component and a pinned third-party
  dependency in another, and an archived config would not say which ran.
* It leaves ``"triton"`` free here for a kernel Lumen actually writes.

Why this exists at all, in one line: on an A100 fla's kernels beat the fp32
reference by 1.33x in fp32 and 3.21x under bf16 autocast, forward *and*
backward, at matched parameter count.  The measurement lives in the design
record; what lives here is the reason it is *allowed* to be used.

**It computes the same function.**  Verified against Lumen's fp64 sequential
oracle, forward and backward, on the diagonal layout:

    lumen chunk  vs oracle   2.07e-7      fla chunk vs oracle   2.24e-7
    gradients, fla vs lumen: dq 2.4e-7  dk 2.8e-7  dv 2.6e-7
                             dbeta 2.9e-7  dg 5.3e-7

fla lands the same distance from Lumen's own oracle as Lumen's chunkwise path
does.  That is `docs/design/GATED_DELTANET.md` §3.7's *"one algorithm reached by
two routes"* — the case that earns the tight tolerance — rather than two
different algorithms for one function.  So this is a faster path, not a
different model, and `tests/test_gdn_backend.py` is the standing proof.

Two conventions that were checked rather than assumed, because both fail
silently:

* **Pass `beta` post-sigmoid, exactly as the reference takes it.**  fla accepts
  `beta` in `[0, 2]` with no flags; `allow_neg_eigval` governs whether the
  *kernel* applies its own `2·sigmoid`, not whether reflections are permitted.
  Reaching for that flag would require handing over the pre-activation instead,
  and passing the post-sigmoid value to it computes `2·sigmoid(2·sigmoid(x))` —
  right shape, no error, wrong model.
* **`scale=1.0`.**  fla scales `q` inside the kernel by default; the reference
  takes `q` already unit-norm and applies nothing.

What is deliberately *not* supported is in :func:`unsupported`.
"""

from __future__ import annotations

import torch

from lumen.gdn.layout import HeadLayout

try:  # pragma: no cover - trivially environment-dependent
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    HAS_FLA = True
except Exception:  # pragma: no cover - a broken install is the same as none
    HAS_FLA = False

__all__ = ["HAS_FLA", "chunk_gated_delta_fla", "unsupported", "usable"]


# fp64 is not in this list on purpose.  These kernels are compiled for fp32 and
# below, so an fp64 caller takes the reference path -- which is the right answer
# for an oracle run, not a limitation of it.
_DTYPES = (torch.float32, torch.bfloat16, torch.float16)


def unsupported(layout: HeadLayout, decay: str) -> str | None:
    """Why this config cannot use fla, or ``None`` if it can.

    Returns a *reason* rather than a bool so the construction error can say
    which part of the config is the problem.  Checked against the config alone,
    with no device access, so it stays testable on a CPU-only box.
    """
    if layout != HeadLayout.diagonal(layout.n_heads):
        return (
            f"fla implements one key per head; this layout is "
            f"{layout.describe()}. Only HeadLayout.diagonal is supported."
        )
    # On a diagonal layout G_k == H and m == 1, so "one rate per key group" and
    # "one rate per state" are the same object -- which is why `state` is here
    # and `state_gated` is not.  `state_gated` widens `a_proj` to H independent
    # signals, and that is a different model, not a different arrangement of
    # the same one.
    if decay not in ("key_group", "state"):
        return (
            f"decay={decay!r} gives each state its own modulation, which fla's "
            f"kernel does not express. Use 'key_group' or the reference backend."
        )
    # `centre` is deliberately absent: it is applied in `_features`, before the
    # kernel sees anything, so it is orthogonal to which backend runs.
    return None


def usable(tensor: torch.Tensor) -> bool:
    """Can the fla path run on this tensor?  CUDA-only, and not fp64.

    A CPU tensor falling back is a device gap rather than a missing capability,
    which is why it is a fallback here and an error at construction.
    """
    return HAS_FLA and tensor.is_cuda and tensor.dtype in _DTYPES


def chunk_gated_delta_fla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    log_alpha: torch.Tensor,
    state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop-in for :func:`lumen.gdn.reference.chunk_gated_delta`, minus the chunk size.

    Same shapes in and out as the reference — `(B, G_k, m, T, d)` — so the
    layer's `_scan` can pick between them without either side knowing.  The
    `m` axis is 1 throughout, guaranteed by :func:`unsupported` refusing every
    layout but the diagonal one.

    ``chunk_size`` is absent from the signature rather than ignored in the body.
    fla chunks internally and Lumen's value cannot reach it, and a parameter
    accepted-and-dropped is how a config field goes quietly inert.
    """
    batch, n_key_groups, per_group, seq_len, d_k = q.shape

    def flatten(x: torch.Tensor) -> torch.Tensor:
        """`(B, G_k, 1, T, D)` → `(B, T, H, D)`, the layout fla expects."""
        return x.squeeze(2).permute(0, 2, 1, 3).contiguous()

    initial = None if state is None else state.squeeze(2).contiguous()

    out, final = chunk_gated_delta_rule(
        q=flatten(q),
        k=flatten(k),
        v=flatten(v),
        # `log_alpha` is `(B, G_k, 1, T)`; fla wants `(B, T, H)`.
        g=log_alpha.squeeze(2).permute(0, 2, 1).contiguous(),
        # Post-sigmoid, exactly as the reference takes it.  See the module
        # docstring -- this is the convention that fails silently if reversed.
        beta=beta.squeeze(2).permute(0, 2, 1).contiguous(),
        # The reference takes `q` unit-norm and applies no scale of its own.
        scale=1.0,
        initial_state=initial,
        output_final_state=True,
    )

    out = out.permute(0, 2, 1, 3).unsqueeze(2)
    return out, final.unsqueeze(2)
