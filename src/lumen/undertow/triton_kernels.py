"""Triton kernels for Undertow's windowed attention — optional, opt-in, measured.

Importing this module never fails.  ``HAS_TRITON`` reports whether the kernels
are usable; everything degrades to :mod:`lumen.undertow.reference` when it is
False.  Triton is a runtime capability here, not a dependency.

What the kernels are for
------------------------
The windowed forward is a gather-and-reduce: each query touches `W` keys that
sit at a fixed offset from it.  Written in torch it becomes `W` separate kernel
launches, each streaming its slice of K or V from HBM.  Fusing the offset loop
into one kernel keeps Q (or the attention weights) in SRAM across all `W`
iterations and pays the memory traffic once.

That is a *bandwidth* win, not a compute win — which matters, because it is
also the reason the win does not depend on tensor cores.  Kernels tuned for
tensor-core GEMMs tend to lose to plain torch on hardware that has none; this
one is not in that category.

The backward reuses the forward kernels
---------------------------------------
Two of the four gradients are the forward kernels wearing different arguments::

    dQ[t]         = Σ_w  g[t,w] · Kext[t+w]     — the aggregate kernel
    dWeights[t,w] = ⟨ g[t] , Vext[t+w] ⟩         — the scores kernel, scale 1

Only ``dKext`` and ``dVext`` have the other shape, where each key accumulates
from the `W` future queries that read it, and those stay in torch.  Half the
backward is covered by kernels that had to exist anyway.  This matters more
than it sounds: the backward is the larger term in training, and an accelerated
forward bolted to a slow backward is how a kernel wins every microbenchmark and
loses end-to-end.

Indexing
--------
These kernels operate on the **extended** key/value array — `W-1` slots of
history (real, or zeros at the start of a sequence) followed by the chunk — so
query ``t`` reads extended index ``t + w`` with no bounds check in the inner
loop.  Building the extension is a cat or a pad in torch, it is differentiable,
and it makes the chunked-prefill path fall out for free rather than needing a
second kernel.  Slots that are structurally absent are masked to ``-inf`` after
the kernel, where the decay profile is also applied.
"""

from __future__ import annotations

import torch

try:  # pragma: no cover - trivially environment-dependent
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover
    HAS_TRITON = False


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


if HAS_TRITON:

    @triton.jit
    def _scores_kernel(
        Q, KEXT, OUT,
        sq_bh, sq_t, sq_d,
        sk_bh, sk_t, sk_d,
        so_bh, so_t, so_w,
        T,
        scale,
        D: tl.constexpr,
        W: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """``out[t, w] = ⟨Q[t], KEXT[t+w]⟩ / scale``.

        Q is loaded once into SRAM and reused across all `W` offsets.  Only `W`
        and the block sizes are ``constexpr``: `T` deliberately is not, because
        making the sequence length part of the specialisation key means a fresh
        compile for every new length, and cold compiles are not free.
        """
        pid_bh = tl.program_id(0)
        pid_t = tl.program_id(1)

        t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        t_mask = t_offs < T
        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < D

        q = tl.load(
            Q + pid_bh * sq_bh + t_offs[:, None] * sq_t + d_offs[None, :] * sq_d,
            mask=t_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        for w in range(W):
            # t + w is always inside the extended array: max is (T-1)+(W-1),
            # and the array holds T+W-1 rows.  No validity branch needed.
            k = tl.load(
                KEXT
                + pid_bh * sk_bh
                + (t_offs + w)[:, None] * sk_t
                + d_offs[None, :] * sk_d,
                mask=t_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            tl.store(
                OUT + pid_bh * so_bh + t_offs * so_t + w * so_w,
                tl.sum(q * k, axis=1) / scale,
                mask=t_mask,
            )

    @triton.jit
    def _aggregate_kernel(
        WT, VEXT, OUT,
        sw_bh, sw_t, sw_w,
        sv_bh, sv_t, sv_d,
        so_bh, so_t, so_d,
        T,
        D: tl.constexpr,
        W: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """``out[t, d] = Σ_w WT[t, w] · VEXT[t+w, d]``, accumulated in fp32."""
        pid_bh = tl.program_id(0)
        pid_t = tl.program_id(1)

        t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        t_mask = t_offs < T
        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < D

        acc = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)

        for w in range(W):
            weight = tl.load(
                WT + pid_bh * sw_bh + t_offs * sw_t + w * sw_w,
                mask=t_mask,
                other=0.0,
            ).to(tl.float32)
            v = tl.load(
                VEXT
                + pid_bh * sv_bh
                + (t_offs + w)[:, None] * sv_t
                + d_offs[None, :] * sv_d,
                mask=t_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += weight[:, None] * v

        tl.store(
            OUT + pid_bh * so_bh + t_offs[:, None] * so_t + d_offs[None, :] * so_d,
            acc,
            mask=t_mask[:, None] & d_mask[None, :],
        )


# Fixed rather than autotuned.  The probe on the development bench reports an
# autotune tax of roughly a minute per kernel per (shape, dtype) key, against
# 48 KB of shared memory that most upstream config sweeps cannot fit anyway.
# Paying that to rediscover a block size on every new shape is a bad trade.
_BLOCK_T = 32


def _scores_fwd(
    q: torch.Tensor, k_ext: torch.Tensor, window: int, scale: float
) -> torch.Tensor:
    batch, heads, seq_len, dim = q.shape
    q_flat = q.reshape(batch * heads, seq_len, dim).contiguous()
    k_flat = k_ext.reshape(batch * heads, k_ext.shape[2], dim).contiguous()
    out = torch.empty(
        batch * heads, seq_len, window, device=q.device, dtype=torch.float32
    )

    grid = (batch * heads, triton.cdiv(seq_len, _BLOCK_T))
    _scores_kernel[grid](
        q_flat, k_flat, out,
        *q_flat.stride(), *k_flat.stride(), *out.stride(),
        seq_len,
        scale,
        D=dim,
        W=window,
        BLOCK_T=_BLOCK_T,
        BLOCK_D=_next_pow2(dim),
    )
    return out.reshape(batch, heads, seq_len, window)


def _aggregate_fwd(
    weights: torch.Tensor, v_ext: torch.Tensor, window: int
) -> torch.Tensor:
    batch, heads, seq_len, _ = weights.shape
    dim = v_ext.shape[-1]
    w_flat = weights.reshape(batch * heads, seq_len, window).contiguous()
    v_flat = v_ext.reshape(batch * heads, v_ext.shape[2], dim).contiguous()
    out = torch.empty(
        batch * heads, seq_len, dim, device=v_ext.device, dtype=torch.float32
    )

    grid = (batch * heads, triton.cdiv(seq_len, _BLOCK_T))
    _aggregate_kernel[grid](
        w_flat, v_flat, out,
        *w_flat.stride(), *v_flat.stride(), *out.stride(),
        seq_len,
        D=dim,
        W=window,
        BLOCK_T=_BLOCK_T,
        BLOCK_D=_next_pow2(dim),
    )
    return out.reshape(batch, heads, seq_len, dim)


def _scatter_back(
    source: torch.Tensor, per_offset: torch.Tensor, window: int, extended_len: int
) -> torch.Tensor:
    """Accumulate `(B,H,T,·)` contributions back onto extended positions ``t+w``.

    The transpose of the forward gather: forward reads ``ext[t+w]`` for each
    ``(t, w)``, so the gradient writes back to the same place.  Torch rather
    than Triton — scattered accumulation is the shape the forward kernels do
    not have, and slice-accumulation vectorises well enough that a kernel here
    would be optimising the smaller half.
    """
    batch, heads, seq_len, _ = source.shape
    out = source.new_zeros(batch, heads, extended_len, source.shape[-1])
    for w in range(window):
        out[:, :, w:w + seq_len, :] += per_offset[..., w].unsqueeze(-1) * source
    return out


class _Scores(torch.autograd.Function):
    """Triton forward for windowed QKᵀ; backward reuses the aggregate kernel."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx, q: torch.Tensor, k_ext: torch.Tensor, window: int, scale: float
    ) -> torch.Tensor:
        ctx.save_for_backward(q, k_ext)
        ctx.window = window
        ctx.scale = scale
        return _scores_fwd(q, k_ext, window, scale)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):  # type: ignore[override]
        q, k_ext = ctx.saved_tensors
        window, scale = ctx.window, ctx.scale
        grad = grad.float().contiguous()

        # dQ[t] = Σ_w g[t,w]·Kext[t+w] — the aggregate kernel exactly.
        dq = _aggregate_fwd(grad, k_ext.float(), window) / scale
        dk_ext = _scatter_back(q.float(), grad, window, k_ext.shape[2]) / scale

        return dq.to(q.dtype), dk_ext.to(k_ext.dtype), None, None


class _Aggregate(torch.autograd.Function):
    """Triton forward for the windowed sum; backward reuses the scores kernel."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx, weights: torch.Tensor, v_ext: torch.Tensor, window: int
    ) -> torch.Tensor:
        ctx.save_for_backward(weights, v_ext)
        ctx.window = window
        return _aggregate_fwd(weights, v_ext, window)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):  # type: ignore[override]
        weights, v_ext = ctx.saved_tensors
        window = ctx.window
        grad = grad.float().contiguous()

        # dWeights[t,w] = ⟨g[t], Vext[t+w]⟩ — the scores kernel at scale 1.
        d_weights = _scores_fwd(grad, v_ext.float(), window, 1.0)
        dv_ext = _scatter_back(grad, weights.float(), window, v_ext.shape[2])

        return d_weights.to(weights.dtype), dv_ext.to(v_ext.dtype), None


def windowed_scores(
    q: torch.Tensor, k_ext: torch.Tensor, window: int, scale: float
) -> torch.Tensor:
    """`(B,H,T,D)` × `(B,H,T+W-1,D)` → `(B,H,T,W)`, autograd-safe."""
    if torch.is_grad_enabled() and (q.requires_grad or k_ext.requires_grad):
        return _Scores.apply(q, k_ext, window, scale)
    return _scores_fwd(q, k_ext, window, scale)


def windowed_aggregate(
    weights: torch.Tensor, v_ext: torch.Tensor, window: int
) -> torch.Tensor:
    """`(B,H,T,W)` × `(B,H,T+W-1,D)` → `(B,H,T,D)`, autograd-safe."""
    if torch.is_grad_enabled() and (weights.requires_grad or v_ext.requires_grad):
        return _Aggregate.apply(weights, v_ext, window)
    return _aggregate_fwd(weights, v_ext, window)


def usable(tensor: torch.Tensor) -> bool:
    """Can the Triton path run on this tensor?  CUDA-only and fp32.

    The dtype check used to be unnecessary rather than absent: the layer cast
    everything with ``.float()`` before dispatching, so nothing else could
    arrive.  That cast was a silent demotion for fp64 and has been replaced by a
    promotion, which means the guarantee has to be stated where it is relied on
    instead of inherited from a cast three call levels away.  An fp64 caller now
    takes the reference path, which is the correct answer rather than a
    limitation -- these kernels are compiled for fp32.
    """
    return HAS_TRITON and tensor.is_cuda and tensor.dtype is torch.float32
