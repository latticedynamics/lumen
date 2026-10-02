"""Triton kernels for Sparse Delta Memory — optional, opt-in, measured.

Importing this module never fails.  ``HAS_TRITON`` reports whether the kernels
are usable; everything degrades to :mod:`lumen.sdm.reference` when it is False.
Triton is a runtime capability here, not a dependency.

What the kernels are for
------------------------
The reference splits a chunk into what needs the table and what does not
(:func:`~lumen.sdm.reference._chunk_terms` / :func:`~lumen.sdm.reference._chunk_apply`),
batches the second over a group of chunks, and walks the first one chunk at a
time.  On a long sequence both halves are expensive in ways that have nothing to
do with the recurrence: the terms make a dozen passes over dense `(C, K, C)`
arrays that are almost entirely zero, and the walk is a dozen small launches per
chunk, mostly waiting on the host.

Both halves run here as kernels, one group of chunks per autograd node:

* **The terms** -- one program per chunk.  A chunk's writes are sorted by
  `slot·C + position` (in torch, integers only), so the writes that share a
  slot form one contiguous *segment*, in position order.  Each entry walks its
  own segment -- the partners it actually has -- and never the `C` cells per
  entry of the dense layout.  `A` and `QK` are built a row at a time.
* **The walk** -- one launch per group.  A program owns one stream's table and
  a block of its value columns and loops over the group's chunks in order,
  with a barrier between a chunk's reads of the table and its writes.  Columns
  never mix, so the programs never talk.

The carry is one number per write
---------------------------------
The reference's carry is a `(C, W, C)` array: what position `s`'s delta adds
to each write entry's slot by the chunk's end.  But every write entry of one
slot carries the *same* row, and that row is nonzero only at the slot's own
writes.  So it is a single coefficient per write,
`c_m = k_m · exp(G_end − G_m)`, and the new row of a slot is

    M_C[n] = exp(G_end) · M₀[n] + Σ_{m writes n} c_m · δ_{position(m)}

summed along the slot's segment.  The dense array, the largest thing a
training step kept, is never formed; neither is the matmul that applied it.
The same reading turns its transpose in the backward into a gather: the
gradient reaching `δ_s` from the new rows is `Σ_w c[s,w] · g[slot(s,w)]`.

Deterministic, with no atomics
------------------------------
Every reduction here has one owner and a fixed order.  The terms sum each
pair's two ends from the two entries' own segments -- sharing a slot is
symmetric.  The walk's backward hands each slot a chunk touched to one owner
-- its first writer, or for a slot only read, its first reader -- which sums
everything that reached the slot in segment order.  So the backward is
reproducible run to run, which the reference's own backward on CUDA is not
unless deterministic algorithms are on.  (It also never needs an atomic, and
Triton 3.5's atomics do not assemble below sm_70.)

Numerics
--------
The same arithmetic as the reference in fp32, in different summation orders:
round-off apart, not bit-identical.  The fp64 oracle is what both answer to.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch.autograd.function import once_differentiable

from lumen.gdn.reference import inv_unit
from lumen.sdm import reference as ref

try:  # pragma: no cover - trivially environment-dependent
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover
    HAS_TRITON = False


def _next_pow2(n: int) -> int:
    return 1 << (max(n, 1) - 1).bit_length()


if HAS_TRITON:

    # ── the table-free terms ──────────────────────────────────────────────

    @triton.jit
    def _terms_fwd_kernel(
        WS, KV, LD, QV,
        WORDER, WLO, WHI, RLO, RHI,
        GW, GR, GEND,
        A, QK, WWEIGHT, RWEIGHT, DECAY, COEF, DEST,
        C: tl.constexpr, W: tl.constexpr, R: tl.constexpr,
        BC: tl.constexpr, BW: tl.constexpr, BR: tl.constexpr,
    ):
        """:func:`~lumen.sdm.reference._chunk_terms` for one chunk, less the solve.

        Flat per-chunk layouts: write entries `e = t·W + w`, read entries
        `f = t·R + r`.  ``WORDER`` is the chunk's write entries sorted by
        `slot·C + t`; ``WLO``/``WHI`` (``RLO``/``RHI`` for reads) bound the
        segment of writes to each entry's slot.  Returns `A` rather than the
        transform -- `(I + diag(β) A)⁻¹` is one batched library call -- and
        the carry as one coefficient per write (see the module docstring).
        """
        pc = tl.program_id(0).to(tl.int64)
        cw = pc * C * W
        cr = pc * C * R
        wv = tl.arange(0, BW)
        w_ok = wv < W
        rv = tl.arange(0, BR)
        r_ok = rv < R
        cols = tl.arange(0, BC)
        c_ok = cols < C

        # ── pass 1: cumulative log-decays, summed member by member ─────────
        # Members arrive in position order and stop counting past `t`, so two
        # entries that share a history add the same terms in the same order
        # and agree bit for bit -- the property the reference keeps for the
        # same reason: a read with nothing written between it and its partner
        # must see an exponent of exactly 0.
        for t in range(C):
            e = t * W + wv
            lo = tl.load(WLO + cw + e, mask=w_ok, other=0)
            n = tl.load(WHI + cw + e, mask=w_ok, other=0) - lo
            g = tl.zeros((BW,), dtype=tl.float32)
            g_end = tl.zeros((BW,), dtype=tl.float32)
            for j in range(tl.max(n, axis=0)):
                valid = j < n
                m = tl.load(WORDER + cw + lo + j, mask=valid, other=0)
                log_decay = tl.load(LD + cw + m, mask=valid, other=0.0)
                g_end += tl.where(valid, log_decay, 0.0)
                g += tl.where(valid & (m // W <= t), log_decay, 0.0)
            tl.store(GW + cw + e, g, mask=w_ok)
            tl.store(GEND + cw + e, g_end, mask=w_ok)
            k = tl.load(KV + cw + e, mask=w_ok, other=0.0)
            tl.store(WWEIGHT + cw + e, k * tl.exp(g), mask=w_ok)
            tl.store(DECAY + cw + e, tl.exp(g_end), mask=w_ok)
            # Both exponents are <= 0: G_end sums every write G sums, and more.
            tl.store(COEF + cw + e, k * tl.exp(g_end - g), mask=w_ok)
            # A slot's first writer in the chunk is the head of its segment.
            head = tl.load(WORDER + cw + lo, mask=w_ok, other=-1)
            slot = tl.load(WS + cw + e, mask=w_ok, other=-1)
            tl.store(DEST + cw + e, tl.where(head == e, slot, -1), mask=w_ok)

            f = t * R + rv
            lo_r = tl.load(RLO + cr + f, mask=r_ok, other=0)
            n_r = tl.load(RHI + cr + f, mask=r_ok, other=0) - lo_r
            g_r = tl.zeros((BR,), dtype=tl.float32)
            for j in range(tl.max(n_r, axis=0)):
                valid = j < n_r
                m = tl.load(WORDER + cw + lo_r + j, mask=valid, other=0)
                log_decay = tl.load(LD + cw + m, mask=valid, other=0.0)
                g_r += tl.where(valid & (m // W <= t), log_decay, 0.0)
            tl.store(GR + cr + f, g_r, mask=r_ok)
            q = tl.load(QV + cr + f, mask=r_ok, other=0.0)
            tl.store(RWEIGHT + cr + f, q * tl.exp(g_r), mask=r_ok)

        # Pass 2 reads other positions' G.
        tl.debug_barrier()

        # ── pass 2: A and QK, a row at a time ──────────────────────────────
        for t in range(C):
            e = t * W + wv
            lo = tl.load(WLO + cw + e, mask=w_ok, other=0)
            n = tl.load(WHI + cw + e, mask=w_ok, other=0) - lo
            k = tl.load(KV + cw + e, mask=w_ok, other=0.0)
            g = tl.load(GW + cw + e, mask=w_ok, other=0.0)
            a_row = tl.zeros((BC,), dtype=tl.float32)
            for j in range(tl.max(n, axis=0)):
                valid = j < n
                m = tl.load(WORDER + cw + lo + j, mask=valid, other=0)
                s = m // W
                pair = valid & (s < t)
                k_m = tl.load(KV + cw + m, mask=pair, other=0.0)
                g_m = tl.load(GW + cw + m, mask=pair, other=0.0)
                # Every factor in (0, 1] by construction, as the reference forms it.
                a = tl.where(pair, k * k_m * tl.exp(tl.where(pair, g - g_m, 0.0)), 0.0)
                hit = (s[:, None] == cols[None, :]) & pair[:, None]
                a_row += tl.sum(tl.where(hit, a[:, None], 0.0), axis=0)
            tl.store(A + (pc * C + t) * C + cols, a_row, mask=c_ok)

            f = t * R + rv
            lo_r = tl.load(RLO + cr + f, mask=r_ok, other=0)
            n_r = tl.load(RHI + cr + f, mask=r_ok, other=0) - lo_r
            q = tl.load(QV + cr + f, mask=r_ok, other=0.0)
            g_r = tl.load(GR + cr + f, mask=r_ok, other=0.0)
            qk_row = tl.zeros((BC,), dtype=tl.float32)
            for j in range(tl.max(n_r, axis=0)):
                valid = j < n_r
                m = tl.load(WORDER + cw + lo_r + j, mask=valid, other=0)
                s = m // W
                pair = valid & (s <= t)
                k_m = tl.load(KV + cw + m, mask=pair, other=0.0)
                g_m = tl.load(GW + cw + m, mask=pair, other=0.0)
                qk = tl.where(pair, q * k_m * tl.exp(tl.where(pair, g_r - g_m, 0.0)), 0.0)
                hit = (s[:, None] == cols[None, :]) & pair[:, None]
                qk_row += tl.sum(tl.where(hit, qk[:, None], 0.0), axis=0)
            tl.store(QK + (pc * C + t) * C + cols, qk_row, mask=c_ok)

    @triton.jit
    def _terms_bwd_kernel(
        KV, QV, WORDER, WLO, WHI, RLO, RHI, RORDER, WRLO, WRHI,
        GW, GR, GEND,
        DA, DQK, DWW, DRW, DDE, DCOEF,
        SG, SE, SGR,
        DK, DL, DQ,
        C: tl.constexpr, W: tl.constexpr, R: tl.constexpr,
        BW: tl.constexpr, BR: tl.constexpr,
    ):
        """The terms' backward: ``DK``, ``DL``, ``DQ`` from the gradients of every output.

        Every pair has two ends.  An entry's own end is summed over its own
        segment; the far end is summed by the partner over *its* segment,
        which holds the same pairs.  A write's reads come from ``RORDER``, the
        chunk's reads sorted the same way, bounded by ``WRLO``/``WRHI``.
        ``SG``, ``SE``, ``SGR`` hold the gradients of the cumulative log-decays,
        which the second pass hands back to every write they summed.
        """
        pc = tl.program_id(0).to(tl.int64)
        cw = pc * C * W
        cr = pc * C * R
        cc = pc * C * C
        wv = tl.arange(0, BW)
        w_ok = wv < W
        rv = tl.arange(0, BR)
        r_ok = rv < R

        # ── pass 1: k, q and the gradients of every cumulative decay ──────
        for t in range(C):
            e = t * W + wv
            lo = tl.load(WLO + cw + e, mask=w_ok, other=0)
            n = tl.load(WHI + cw + e, mask=w_ok, other=0) - lo
            k = tl.load(KV + cw + e, mask=w_ok, other=0.0)
            g = tl.load(GW + cw + e, mask=w_ok, other=0.0)
            g_end = tl.load(GEND + cw + e, mask=w_ok, other=0.0)
            d_k = tl.zeros((BW,), dtype=tl.float32)
            d_g = tl.zeros((BW,), dtype=tl.float32)
            d_end = tl.zeros((BW,), dtype=tl.float32)
            for j in range(tl.max(n, axis=0)):
                valid = j < n
                m = tl.load(WORDER + cw + lo + j, mask=valid, other=0)
                s = m // W
                k_m = tl.load(KV + cw + m, mask=valid, other=0.0)
                g_m = tl.load(GW + cw + m, mask=valid, other=0.0)
                # A[t, s], s < t: this entry is the later end
                late = valid & (s < t)
                ratio = tl.exp(tl.where(late, g - g_m, 0.0))
                d_a = tl.load(DA + cc + t * C + s, mask=late, other=0.0)
                d_k += tl.where(late, d_a * k_m * ratio, 0.0)
                d_g += tl.where(late, d_a * (k * k_m * ratio), 0.0)
                # A[s, t], s > t: this entry is the earlier end
                early = valid & (s > t)
                ratio = tl.exp(tl.where(early, g_m - g, 0.0))
                d_a = tl.load(DA + cc + s * C + t, mask=early, other=0.0)
                d_k += tl.where(early, d_a * k_m * ratio, 0.0)
                d_g -= tl.where(early, d_a * (k_m * k * ratio), 0.0)
            # QK[t_f, t], t_f >= t: this entry as the partner of later reads
            lo2 = tl.load(WRLO + cw + e, mask=w_ok, other=0)
            n2 = tl.load(WRHI + cw + e, mask=w_ok, other=0) - lo2
            for j in range(tl.max(n2, axis=0)):
                valid = j < n2
                f = tl.load(RORDER + cr + lo2 + j, mask=valid, other=0)
                t_f = f // R
                read = valid & (t_f >= t)
                q_f = tl.load(QV + cr + f, mask=read, other=0.0)
                g_f = tl.load(GR + cr + f, mask=read, other=0.0)
                ratio = tl.exp(tl.where(read, g_f - g, 0.0))
                d_qk = tl.load(DQK + cc + t_f * C + t, mask=read, other=0.0)
                d_k += tl.where(read, d_qk * q_f * ratio, 0.0)
                d_g -= tl.where(read, d_qk * (q_f * k * ratio), 0.0)
            # The per-entry outputs: weight, decay to the end, carry coefficient.
            decay = tl.exp(g)
            d_ww = tl.load(DWW + cw + e, mask=w_ok, other=0.0)
            d_k += d_ww * decay
            d_g += d_ww * (k * decay)
            d_de = tl.load(DDE + cw + e, mask=w_ok, other=0.0)
            d_end += d_de * tl.exp(g_end)
            to_end = tl.exp(g_end - g)
            d_coef = tl.load(DCOEF + cw + e, mask=w_ok, other=0.0)
            d_k += d_coef * to_end
            d_end += d_coef * (k * to_end)
            d_g -= d_coef * (k * to_end)
            tl.store(DK + cw + e, d_k, mask=w_ok)
            tl.store(SG + cw + e, d_g, mask=w_ok)
            tl.store(SE + cw + e, d_end, mask=w_ok)

            f = t * R + rv
            lo_r = tl.load(RLO + cr + f, mask=r_ok, other=0)
            n_r = tl.load(RHI + cr + f, mask=r_ok, other=0) - lo_r
            q = tl.load(QV + cr + f, mask=r_ok, other=0.0)
            g_r = tl.load(GR + cr + f, mask=r_ok, other=0.0)
            d_q = tl.zeros((BR,), dtype=tl.float32)
            d_gr = tl.zeros((BR,), dtype=tl.float32)
            for j in range(tl.max(n_r, axis=0)):
                valid = j < n_r
                m = tl.load(WORDER + cw + lo_r + j, mask=valid, other=0)
                s = m // W
                pair = valid & (s <= t)
                k_m = tl.load(KV + cw + m, mask=pair, other=0.0)
                g_m = tl.load(GW + cw + m, mask=pair, other=0.0)
                ratio = tl.exp(tl.where(pair, g_r - g_m, 0.0))
                d_qk = tl.load(DQK + cc + t * C + s, mask=pair, other=0.0)
                d_q += tl.where(pair, d_qk * k_m * ratio, 0.0)
                d_gr += tl.where(pair, d_qk * (q * k_m * ratio), 0.0)
            decay_r = tl.exp(g_r)
            d_rw = tl.load(DRW + cr + f, mask=r_ok, other=0.0)
            d_q += d_rw * decay_r
            d_gr += d_rw * (q * decay_r)
            tl.store(DQ + cr + f, d_q, mask=r_ok)
            tl.store(SGR + cr + f, d_gr, mask=r_ok)

        # Pass 2 reads other positions' decay gradients.
        tl.debug_barrier()

        # ── pass 2: each write's log-decay, from every sum it entered ──────
        for t in range(C):
            m = t * W + wv
            lo = tl.load(WLO + cw + m, mask=w_ok, other=0)
            n = tl.load(WHI + cw + m, mask=w_ok, other=0) - lo
            d_l = tl.zeros((BW,), dtype=tl.float32)
            for j in range(tl.max(n, axis=0)):
                valid = j < n
                e = tl.load(WORDER + cw + lo + j, mask=valid, other=0)
                # G(e) summed this write if e is at or after it; G_end(e) did regardless.
                d_l += tl.load(SG + cw + e, mask=valid & (e // W >= t), other=0.0)
                d_l += tl.load(SE + cw + e, mask=valid, other=0.0)
            lo2 = tl.load(WRLO + cw + m, mask=w_ok, other=0)
            n2 = tl.load(WRHI + cw + m, mask=w_ok, other=0) - lo2
            for j in range(tl.max(n2, axis=0)):
                valid = j < n2
                f = tl.load(RORDER + cr + lo2 + j, mask=valid, other=0)
                d_l += tl.load(SGR + cr + f, mask=valid & (f // R >= t), other=0.0)
            tl.store(DL + cw + m, d_l, mask=w_ok)

    # ── the walk over the table ───────────────────────────────────────────

    @triton.jit
    def _mm(a, b, DOT: tl.constexpr):
        """`a @ b` for a `(C, C)` and a `(C, BD)` tile.

        ``tl.dot`` (plain FMAs on a card without tensor cores) where every side
        is at least 16; a broadcast-and-reduce below that, where ``tl.dot``
        does not compile.  The broadcast holds `C·C·BD` products at once, which
        is cheap at 8 columns and a register hog at 32 -- so the switch is a
        cost choice as much as a legality one.
        """
        if DOT:
            return tl.dot(a, b, input_precision="ieee")
        return tl.sum(a[:, :, None] * b[None, :, :], axis=1)

    @triton.jit
    def _mm_t(a, b, DOT: tl.constexpr):
        """`aᵀ @ b`, the same way."""
        if DOT:
            return tl.dot(tl.trans(a), b, input_precision="ieee")
        return tl.sum(a[:, :, None] * b[:, None, :], axis=0)


    @triton.jit
    def _walk_fwd_kernel(
        TABLE, W_IDX, R_IDX, DEST, WORDER, WLO, WHI,
        V, BETA, WW, RW, X, Q, DE, COEF,
        OUT, SAVE_W, SAVE_R, DELTA, ERR,
        n_chunks, L,
        D: tl.constexpr, W: tl.constexpr, R: tl.constexpr, C: tl.constexpr,
        BC: tl.constexpr, BW: tl.constexpr, BR: tl.constexpr, BD: tl.constexpr,
        DOT: tl.constexpr, SAVE: tl.constexpr,
    ):
        """One stream, one block of value columns, every chunk of a group in order.

        Flat layouts, one row of the batch per stream: positions ``(P, L)``,
        entries ``(P, L, W|R)``, the solve and the read-out ``(P, L, C)``, the
        segments per chunk.  ``TABLE`` is ``(P·N, D)``, written in place.
        ``DELTA`` is written whether or not ``SAVE``: the new rows gather it.
        """
        p = tl.program_id(0).to(tl.int64)
        d = tl.program_id(1) * BD + tl.arange(0, BD)
        d_ok = d < D
        t = tl.arange(0, BC)
        t_ok = t < C
        td_ok = t_ok[:, None] & d_ok[None, :]
        cc_ok = t_ok[:, None] & t_ok[None, :]
        t3 = t[:, None, None]
        d3 = d[None, None, :]
        d3_ok = d_ok[None, None, :]
        w3 = tl.arange(0, BW)[None, :, None]
        r3 = tl.arange(0, BR)[None, :, None]

        for c in range(n_chunks):
            base = p * L + c * C
            pos = base + t

            # ── the rows as the chunk enters: retrieve and read back ─────────
            ret = tl.zeros((BC, BD), dtype=tl.float32)
            for w0 in range(0, W, BW):
                ok = (t3 < C) & (w0 + w3 < W)
                entry = (base + t3) * W + w0 + w3
                idx = tl.load(W_IDX + entry, mask=ok, other=0)
                weight = tl.load(WW + entry, mask=ok, other=0.0)
                rows = tl.load(
                    TABLE + idx * D + d3, mask=ok & d3_ok, other=0.0, cache_modifier=".cg"
                )
                if SAVE:
                    tl.store(SAVE_W + entry * D + d3, rows, mask=ok & d3_ok)
                ret += tl.sum(weight * rows, axis=1)
            read_back = tl.zeros((BC, BD), dtype=tl.float32)
            for r0 in range(0, R, BR):
                ok = (t3 < C) & (r0 + r3 < R)
                entry = (base + t3) * R + r0 + r3
                idx = tl.load(R_IDX + entry, mask=ok, other=0)
                weight = tl.load(RW + entry, mask=ok, other=0.0)
                rows = tl.load(
                    TABLE + idx * D + d3, mask=ok & d3_ok, other=0.0, cache_modifier=".cg"
                )
                if SAVE:
                    tl.store(SAVE_R + entry * D + d3, rows, mask=ok & d3_ok)
                read_back += tl.sum(weight * rows, axis=1)

            # ── the solve and the read-out ─────────────────────────────────
            v = tl.load(V + pos[:, None] * D + d[None, :], mask=td_ok, other=0.0)
            beta = tl.load(BETA + pos, mask=t_ok, other=0.0)
            err = v - ret
            u = beta[:, None] * err
            transform = tl.load(X + pos[:, None] * C + t[None, :], mask=cc_ok, other=0.0)
            delta = _mm(transform, u, DOT)
            qk = tl.load(Q + pos[:, None] * C + t[None, :], mask=cc_ok, other=0.0)
            out = read_back + _mm(qk, delta, DOT)
            tl.store(OUT + pos[:, None] * D + d[None, :], out, mask=td_ok)
            tl.store(DELTA + pos[:, None] * D + d[None, :], delta, mask=td_ok)
            if SAVE:
                tl.store(ERR + pos[:, None] * D + d[None, :], err, mask=td_ok)

            # Every read of the table lands before any write, and every δ of
            # the chunk before the new rows gather them.
            tl.debug_barrier()

            # ── the new rows, one per slot, placed by its first writer ─────
            for w0 in range(0, W, BW):
                ok = (t3 < C) & (w0 + w3 < W)
                entry = (base + t3) * W + w0 + w3
                dest = tl.load(DEST + entry, mask=ok, other=-1)
                first = ok & (dest >= 0)
                decay = tl.load(DE + entry, mask=first, other=0.0)
                if SAVE:
                    rows = tl.load(SAVE_W + entry * D + d3, mask=first & d3_ok, other=0.0)
                else:
                    # Safe to gather again: only a slot's first writer stores,
                    # each slot has one, so no earlier block wrote this row.
                    idx = tl.load(W_IDX + entry, mask=first, other=0)
                    rows = tl.load(
                        TABLE + idx * D + d3, mask=first & d3_ok, other=0.0, cache_modifier=".cg"
                    )
                # The head of a slot's segment is its first writer itself, at
                # this very position: its share is this chunk's δ, already here.
                coef = tl.load(COEF + entry, mask=first, other=0.0)
                new_rows = decay * rows + coef * delta[:, None, :]
                lo = tl.load(WLO + entry, mask=first, other=0)
                n = tl.load(WHI + entry, mask=first, other=0) - lo
                for j in range(1, tl.max(n)):
                    valid = first & (j < n)
                    m = tl.load(WORDER + base * W + lo + j, mask=valid, other=0)
                    coef = tl.load(COEF + base * W + m, mask=valid, other=0.0)
                    moved = tl.load(
                        DELTA + (base + m // W) * D + d3, mask=valid & d3_ok, other=0.0
                    )
                    new_rows += coef * moved
                tl.store(TABLE + dest * D + d3, new_rows, mask=first & d3_ok)

            # And every write lands before the next chunk reads.
            tl.debug_barrier()

    @triton.jit
    def _read_sums_kernel(
        RW, G_OUT, RORDER, RLO, RHI, RRLO, RRHI,
        RSUM, ROWNER,
        D: tl.constexpr, R: tl.constexpr, C: tl.constexpr,
        BC: tl.constexpr, BR: tl.constexpr, BD: tl.constexpr,
    ):
        """What a chunk's reads hand back to each slot they read, summed per slot.

        Depends on nothing the backward walk carries -- only the output's
        gradient -- so it runs before the walk, with the whole card, one
        program per chunk and block of columns.  A read segment's sum lands at
        its head, but only where the walk will look for it: a slot read more
        than once, or read and written (its first writer gathers the sum).
        Most slots are read once and not written, and their one term is the
        walk's to form from registers.  ``ROWNER`` marks the heads whose slot
        no write names, which own it in the walk: 1 for a lone read, 2 for a
        segment whose sum is here.
        """
        pc = tl.program_id(0).to(tl.int64)
        d = tl.program_id(1) * BD + tl.arange(0, BD)
        base = pc * C
        t = tl.arange(0, BC)
        t3 = t[:, None, None]
        d3 = d[None, None, :]
        d3_ok = d3 < D
        r3 = tl.arange(0, BR)[None, :, None]
        g_out = tl.load(
            G_OUT + (base + t[:, None]) * D + d[None, :],
            mask=(t < C)[:, None] & (d < D)[None, :],
            other=0.0,
        )
        for r0 in range(0, R, BR):
            ok = (t3 < C) & (r0 + r3 < R)
            entry = (base + t3) * R + r0 + r3
            lo = tl.load(RRLO + entry, mask=ok, other=0)
            head = ok & (tl.load(RORDER + base * R + lo, mask=ok, other=-1) == entry - base * R)
            n = tl.load(RRHI + entry, mask=head, other=0) - tl.where(head, lo, 0)
            written = tl.load(RHI + entry, mask=head, other=0) != tl.load(
                RLO + entry, mask=head, other=0
            )
            summed = head & ((n > 1) | written)
            # The head reads at this position, so its share needs no gather.
            owed = tl.load(RW + entry, mask=summed, other=0.0) * g_out[:, None, :]
            for j in range(1, tl.max(n)):
                valid = summed & (j < n)
                f = tl.load(RORDER + base * R + lo + j, mask=valid, other=0)
                weight = tl.load(RW + base * R + f, mask=valid, other=0.0)
                g_out_f = tl.load(
                    G_OUT + (base + f // R) * D + d3, mask=valid & d3_ok, other=0.0
                )
                owed += weight * g_out_f
            tl.store(RSUM + entry * D + d3, owed, mask=summed & d3_ok)
            if tl.program_id(1) == 0:
                owner = head & (written == 0)
                tl.store(ROWNER + entry, tl.where(owner, tl.where(n > 1, 2, 1), 0).to(tl.int8), mask=ok)

    @triton.jit
    def _walk_bwd_kernel(
        GT, W_IDX, R_IDX, DEST,
        WORDER, WLO, WHI, RORDER, WRLO, WRHI, RSUM, ROWNER,
        BETA, WW, RW, X, Q, DE, COEF, G_OUT,
        GSLOT, GDELTA, GU,
        n_chunks, L,
        D: tl.constexpr, W: tl.constexpr, R: tl.constexpr, C: tl.constexpr,
        BC: tl.constexpr, BW: tl.constexpr, BR: tl.constexpr, BD: tl.constexpr,
        DOT: tl.constexpr,
    ):
        """The table gradient, carried back through a group's chunks in reverse.

        ``GT`` is the gradient of the table as the group exits, `(P·N, D)`,
        and leaves as the gradient of the table as it entered.  Per chunk:
        gather each write's slot gradient (``GSLOT``, kept for the terms), the
        gradient of every δ from it, then hand each slot the chunk touched to
        one owner, which sums everything that reached its old row in segment
        order -- a store, not an atomic.
        """
        p = tl.program_id(0).to(tl.int64)
        d = tl.program_id(1) * BD + tl.arange(0, BD)
        d_ok = d < D
        t = tl.arange(0, BC)
        t_ok = t < C
        td_ok = t_ok[:, None] & d_ok[None, :]
        cc_ok = t_ok[:, None] & t_ok[None, :]
        t3 = t[:, None, None]
        d3 = d[None, None, :]
        d3_ok = d_ok[None, None, :]
        w3 = tl.arange(0, BW)[None, :, None]
        r3 = tl.arange(0, BR)[None, :, None]

        for i in range(n_chunks):
            c = n_chunks - 1 - i
            base = p * L + c * C
            pos = base + t

            g_out = tl.load(G_OUT + pos[:, None] * D + d[None, :], mask=td_ok, other=0.0)
            beta = tl.load(BETA + pos, mask=t_ok, other=0.0)
            qk = tl.load(Q + pos[:, None] * C + t[None, :], mask=cc_ok, other=0.0)
            # gδ[s] = Σ_t QK[t, s] · g_out[t]
            g_delta = _mm_t(qk, g_out, DOT)

            # ── the new rows: each write's slot gradient, and δ's share ────
            for w0 in range(0, W, BW):
                ok = (t3 < C) & (w0 + w3 < W)
                entry = (base + t3) * W + w0 + w3
                idx = tl.load(W_IDX + entry, mask=ok, other=0)
                g_slot = tl.load(
                    GT + idx * D + d3, mask=ok & d3_ok, other=0.0, cache_modifier=".cg"
                )
                tl.store(GSLOT + entry * D + d3, g_slot, mask=ok & d3_ok)
                coef = tl.load(COEF + entry, mask=ok, other=0.0)
                g_delta += tl.sum(coef * g_slot, axis=1)

            transform = tl.load(X + pos[:, None] * C + t[None, :], mask=cc_ok, other=0.0)
            # gu[s] = Σ_t X[t, s] · gδ[t]
            g_u = _mm_t(transform, g_delta, DOT)
            tl.store(GDELTA + pos[:, None] * D + d[None, :], g_delta, mask=td_ok)
            tl.store(GU + pos[:, None] * D + d[None, :], g_u, mask=td_ok)

            # Every slot gradient is read before any is replaced, and gu is
            # stored before the owners gather it.
            tl.debug_barrier()

            # ── written slots: the first writer owns the old row ───────────
            for w0 in range(0, W, BW):
                ok = (t3 < C) & (w0 + w3 < W)
                entry = (base + t3) * W + w0 + w3
                dest = tl.load(DEST + entry, mask=ok, other=-1)
                first = ok & (dest >= 0)
                decay = tl.load(DE + entry, mask=first, other=0.0)
                # What the new row passed back through the decay ...
                owed = decay * tl.load(GSLOT + entry * D + d3, mask=first & d3_ok, other=0.0)
                # ... what every write's retrieval drew from it, starting with
                # the owner's own, at this position ...
                weight = tl.load(WW + entry, mask=first, other=0.0)
                owed -= (weight * beta[:, None, None]) * g_u[:, None, :]
                lo = tl.load(WLO + entry, mask=first, other=0)
                n = tl.load(WHI + entry, mask=first, other=0) - lo
                for j in range(1, tl.max(n)):
                    valid = first & (j < n)
                    m = tl.load(WORDER + base * W + lo + j, mask=valid, other=0)
                    s = m // W
                    weight = tl.load(WW + base * W + m, mask=valid, other=0.0)
                    beta_m = tl.load(BETA + base + s, mask=valid, other=0.0)
                    g_u_m = tl.load(GU + (base + s) * D + d3, mask=valid & d3_ok, other=0.0)
                    owed -= (weight * beta_m) * g_u_m
                # ... and every read of it, summed before the walk began.
                lo = tl.load(WRLO + entry, mask=first, other=0)
                read = first & (tl.load(WRHI + entry, mask=first, other=0) > lo)
                head = tl.load(RORDER + base * R + lo, mask=read, other=0)
                owed += tl.load(RSUM + (base * R + head) * D + d3, mask=read & d3_ok, other=0.0)
                # A replaced row's gradient is exactly this: nothing after the
                # chunk reached the old value except through the new row.
                tl.store(GT + dest * D + d3, owed, mask=first & d3_ok)

            # ── slots only read: the first reader owns them ────────────────
            for r0 in range(0, R, BR):
                ok = (t3 < C) & (r0 + r3 < R)
                entry = (base + t3) * R + r0 + r3
                kind = tl.load(ROWNER + entry, mask=ok, other=0)
                owner = ok & (kind != 0)
                keep = owner & d3_ok
                idx = tl.load(R_IDX + entry, mask=owner, other=0)
                # A lone read's share is formed here; a segment's was summed.
                lone = tl.load(RW + entry, mask=owner & (kind == 1), other=0.0)
                owed = lone * g_out[:, None, :] + tl.load(
                    RSUM + entry * D + d3, mask=keep & (kind == 2), other=0.0
                )
                held = tl.load(GT + idx * D + d3, mask=keep, other=0.0, cache_modifier=".cg")
                tl.store(GT + idx * D + d3, held + owed, mask=keep)

            # Every slot is settled before the previous chunk reads.
            tl.debug_barrier()


    @triton.jit
    def _mm_nt(a, b, DOT: tl.constexpr):
        """`a @ bᵀ` for two `(C, BD)` tiles, the same way as :func:`_mm`."""
        if DOT:
            return tl.dot(a, tl.trans(b), input_precision="ieee")
        return tl.sum(a[:, None, :] * b[None, :, :], axis=2)

    @triton.jit
    def _walk_grads_kernel(
        SAVE_W, SAVE_R, GSLOT, ERR, DELTA, G_OUT, GDELTA, GU, BETA, DEST,
        D_X, D_QK, D_BETA, D_V, D_WW, D_RW, D_DE, D_COEF,
        D: tl.constexpr, W: tl.constexpr, R: tl.constexpr, C: tl.constexpr,
        BC: tl.constexpr, BW: tl.constexpr, BR: tl.constexpr, BD: tl.constexpr,
        DOT: tl.constexpr,
    ):
        """Everything in a chunk's backward that reduces over the value width.

        One program per chunk, after the walk has passed: nothing here is
        sequential, so it runs with the whole card, and each sum over the
        width is done by one program in a fixed order.
        """
        pc = tl.program_id(0).to(tl.int64)
        base = pc * C
        t = tl.arange(0, BC)
        t_ok = t < C
        pos = base + t
        cc_ok = t_ok[:, None] & t_ok[None, :]
        beta = tl.load(BETA + pos, mask=t_ok, other=0.0)

        # ── the solve's and the read-out's matrices, β, and v ─────────────
        d_x = tl.zeros((BC, BC), dtype=tl.float32)
        d_qk = tl.zeros((BC, BC), dtype=tl.float32)
        d_beta = tl.zeros((BC,), dtype=tl.float32)
        for d0 in range(0, D, BD):
            d = d0 + tl.arange(0, BD)
            at = pos[:, None] * D + d[None, :]
            ok = t_ok[:, None] & (d < D)[None, :]
            err = tl.load(ERR + at, mask=ok, other=0.0)
            g_u = tl.load(GU + at, mask=ok, other=0.0)
            g_delta = tl.load(GDELTA + at, mask=ok, other=0.0)
            g_out = tl.load(G_OUT + at, mask=ok, other=0.0)
            delta = tl.load(DELTA + at, mask=ok, other=0.0)
            d_x += _mm_nt(g_delta, beta[:, None] * err, DOT)
            d_qk += _mm_nt(g_out, delta, DOT)
            d_beta += tl.sum(g_u * err, axis=1)
            tl.store(D_V + at, beta[:, None] * g_u, mask=ok)
        tl.store(D_X + (base + t[:, None]) * C + t[None, :], d_x, mask=cc_ok)
        tl.store(D_QK + (base + t[:, None]) * C + t[None, :], d_qk, mask=cc_ok)
        tl.store(D_BETA + pos, d_beta, mask=t_ok)

        # ── per write: its weight, its decay to the end, its coefficient ───
        t3 = t[:, None, None]
        w3 = tl.arange(0, BW)[None, :, None]
        for w0 in range(0, W, BW):
            ok_e = (t3 < C) & (w0 + w3 < W)
            entry = (base + t3) * W + w0 + w3
            d_ww = tl.zeros((BC, BW), dtype=tl.float32)
            d_de = tl.zeros((BC, BW), dtype=tl.float32)
            d_coef = tl.zeros((BC, BW), dtype=tl.float32)
            for d0 in range(0, D, BD):
                d = d0 + tl.arange(0, BD)
                d3 = d[None, None, :]
                ok = ok_e & (d3 < D)
                rows = tl.load(SAVE_W + entry * D + d3, mask=ok, other=0.0)
                g_slot = tl.load(GSLOT + entry * D + d3, mask=ok, other=0.0)
                at = pos[:, None] * D + d[None, :]
                ok2 = t_ok[:, None] & (d < D)[None, :]
                g_ret = -beta[:, None] * tl.load(GU + at, mask=ok2, other=0.0)
                delta = tl.load(DELTA + at, mask=ok2, other=0.0)
                d_ww += tl.sum(g_ret[:, None, :] * rows, axis=2)
                d_de += tl.sum(g_slot * rows, axis=2)
                d_coef += tl.sum(g_slot * delta[:, None, :], axis=2)
            flat = (base + t[:, None]) * W + w0 + tl.arange(0, BW)[None, :]
            ok_f = t_ok[:, None] & (w0 + tl.arange(0, BW) < W)[None, :]
            # Only a slot's first writer's decay reached anything.
            first = tl.load(DEST + flat, mask=ok_f, other=-1) >= 0
            tl.store(D_WW + flat, d_ww, mask=ok_f)
            tl.store(D_DE + flat, tl.where(first, d_de, 0.0), mask=ok_f)
            tl.store(D_COEF + flat, d_coef, mask=ok_f)

        # ── per read: its weight ───────────────────────────────────────────
        r3 = tl.arange(0, BR)[None, :, None]
        for r0 in range(0, R, BR):
            ok_e = (t3 < C) & (r0 + r3 < R)
            entry = (base + t3) * R + r0 + r3
            d_rw = tl.zeros((BC, BR), dtype=tl.float32)
            for d0 in range(0, D, BD):
                d = d0 + tl.arange(0, BD)
                d3 = d[None, None, :]
                rows = tl.load(SAVE_R + entry * D + d3, mask=ok_e & (d3 < D), other=0.0)
                at = pos[:, None] * D + d[None, :]
                g_out = tl.load(G_OUT + at, mask=t_ok[:, None] & (d < D)[None, :], other=0.0)
                d_rw += tl.sum(g_out[:, None, :] * rows, axis=2)
            flat = (base + t[:, None]) * R + r0 + tl.arange(0, BR)[None, :]
            ok_f = t_ok[:, None] & (r0 + tl.arange(0, BR) < R)[None, :]
            tl.store(D_RW + flat, d_rw, mask=ok_f)


# ── launch configuration ──────────────────────────────────────────────────


def _walk_launch(
    rows: int, chunk: int, n_writes: int, n_reads: int, d_v: int, device: torch.device
) -> tuple[dict[str, int], tuple[int, int], int]:
    """Tiles, grid and warps for one walk launch.  Fixed, not autotuned.

    The walk is a chain of dependent loads per program, and two things decide
    how long it is.  **Narrow column blocks gather a whole chunk's rows in one
    round** -- 8 columns is one 32-byte sector of a row, the unit the memory
    system moves anyway -- so they shorten the chain.  But each program holds
    a chunk's worth of tiles and fills its multiprocessor's registers, and a
    persistent kernel whose grid does not fit the card at once runs in waves,
    each paying the whole chain.  So: the narrowest block whose grid fits one
    wave, one program per multiprocessor.  Measured on one Pascal card, where
    that rule picked the best of the column blocks tried at every shape.

    The walks' loops over entry blocks are deliberately *not* unrolled.
    Unrolling was meant to put every block's loads in flight at once; what it
    did was keep every block's tiles live, at 255 registers and spilling, and
    the rolled loop -- one block's tiles at a time, 128 to 168 registers --
    was 9-22% faster at every shape measured.
    """
    block_c = _next_pow2(chunk)
    slots = torch.cuda.get_device_properties(device).multi_processor_count
    # Past 32 positions the broadcast products hold C·C·BD values at once,
    # which no column block fits, so the block must be wide enough for tl.dot.
    narrowest = (8, 16) if block_c <= 32 else (16,)
    block_d = 32
    for width in narrowest:
        if rows * triton.cdiv(d_v, width) <= slots:
            block_d = width
            break
    # A (C, BW, BD) tile reduced across warps is staged whole in shared
    # memory: 32 KB at 8192 floats, inside the 48 KB a block gets on the bench
    # card; twice that does not launch there.
    entries = max(1, 8192 // (block_c * block_d))
    blocks = dict(
        BC=block_c,
        BW=min(_next_pow2(n_writes), entries),
        BR=min(_next_pow2(n_reads), entries),
        BD=block_d,
        DOT=block_c >= 16 and block_d >= 16,
    )
    return blocks, (rows, triton.cdiv(d_v, block_d)), 8


def _terms_blocks(chunk: int, n_writes: int, n_reads: int) -> dict[str, int]:
    """Tiles and warps for the terms kernels: one position's entries per step."""
    block_w, block_r = _next_pow2(n_writes), _next_pow2(n_reads)
    warps = max(1, min(4, max(block_w, block_r) // 32))
    return dict(BW=block_w, BR=block_r, num_warps=warps)


# ── the sort ──────────────────────────────────────────────────────────────


def _segments(
    w_idx: torch.Tensor, r_idx: torch.Tensor, n_slots: int, reads_too: bool
) -> tuple[torch.Tensor, ...]:
    """Every entry's segments, by one sort of the writes (and one of the reads).

    `(P', C, W|R)` slots, each below ``n_slots`` → int32, flat per chunk:

    * ``worder`` `(P', C·W)` -- write entries sorted by `slot·C + position`,
      unique because a position names a slot at most once;
    * ``wlo, whi`` -- each write's segment of same-slot writes in it;
    * ``rlo, rhi`` -- each read's segment of writes to its slot;

    and with ``reads_too`` -- the backward's, not the forward's:

    * ``rorder`` `(P', C·R)` -- read entries sorted the same way (a slot read
      twice at one position ties; the sort is stable, so ties keep entry order);
    * ``wrlo, wrhi`` -- each write's segment of reads of its slot;
    * ``rrlo, rrhi`` -- each read's segment of reads of its slot.

    The keys are int32 whenever `n_slots·C` fits, which halves what the sort
    and the searches move.  Integers only, no gradient.
    """
    rows, chunk, n_writes = w_idx.shape
    n_reads = r_idx.shape[-1]
    if n_slots * chunk < 2**31:
        w_idx, r_idx = w_idx.to(torch.int32), r_idx.to(torch.int32)
    position = torch.arange(chunk, device=w_idx.device, dtype=w_idx.dtype).view(1, chunk, 1)
    w_base = (w_idx * chunk).reshape(rows, chunk * n_writes)
    r_base = (r_idx * chunk).reshape(rows, chunk * n_reads)
    w_keys, w_order = torch.sort((w_idx * chunk + position).reshape(rows, -1), dim=-1)

    def search(keys: torch.Tensor, base: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.searchsorted(keys, base, out_int32=True),
            torch.searchsorted(keys, base + chunk, out_int32=True),
        )

    found = (w_order.to(torch.int32), *search(w_keys, w_base), *search(w_keys, r_base))
    if not reads_too:
        return found
    r_keys, r_order = torch.sort(
        (r_idx * chunk + position).reshape(rows, -1), dim=-1, stable=True
    )
    return (
        *found,
        r_order.to(torch.int32),
        *search(r_keys, w_base),
        *search(r_keys, r_base),
    )


# ── one group of chunks: terms, solve, walk ───────────────────────────────


def _prepare_group(
    w_idx: torch.Tensor,
    r_idx: torch.Tensor,
    w_val: torch.Tensor,
    w_logd: torch.Tensor,
    r_val: torch.Tensor,
    beta: torch.Tensor,
    chunk: int,
    n_slots: int,
) -> dict[str, torch.Tensor]:
    """A group's table-free half: the sort, the terms, the solve.

    `(P, L, …)` contiguous inputs, `L = n·C`.  Depends on nothing the walk
    writes, so the driver runs it one group ahead, on a second stream, while
    the walk of the group before holds the first.
    """
    rows, length, n_writes = w_idx.shape
    n_reads = r_idx.shape[-1]
    n = length // chunk
    fold = (rows * n, chunk)
    w_order, w_lo, w_hi, r_lo, r_hi = _segments(
        w_idx.view(*fold, n_writes), r_idx.view(*fold, n_reads), n_slots, reads_too=False
    )
    new = w_val.new_empty
    g_w, g_end, g_r = new(w_val.shape), new(w_val.shape), new(r_val.shape)
    a, qk = new(rows * n, chunk, chunk), new(rows * n, chunk, chunk)
    w_weight, r_weight = new(w_val.shape), new(r_val.shape)
    decay_end, coef = new(w_val.shape), new(w_val.shape)
    dest = torch.empty_like(w_idx)
    _terms_fwd_kernel[(rows * n,)](
        w_idx, w_val, w_logd, r_val,
        w_order, w_lo, w_hi, r_lo, r_hi,
        g_w, g_r, g_end,
        a, qk, w_weight, r_weight, decay_end, coef, dest,
        C=chunk, W=n_writes, R=n_reads, BC=_next_pow2(chunk),
        **_terms_blocks(chunk, n_writes, n_reads),
    )
    # contiguous: a batched triangular solve may hand back column-major
    # storage, and a kernel reads raw strides.
    transform = inv_unit(beta.view(*fold).unsqueeze(-1) * a).contiguous()
    return dict(
        a=a, transform=transform, qk=qk, dest=dest, coef=coef,
        w_weight=w_weight, r_weight=r_weight, decay_end=decay_end,
        g_w=g_w, g_r=g_r, g_end=g_end, w_order=w_order, w_lo=w_lo, w_hi=w_hi,
    )


def _walk_group(
    table: torch.Tensor,
    w_idx: torch.Tensor,
    r_idx: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    prepared: dict[str, torch.Tensor],
    chunk: int,
    save: bool,
) -> dict[str, torch.Tensor]:
    """A group's walk; ``table`` `(P·N, D)` is written in place.

    Returns ``out`` and -- with ``save`` -- the rows and vectors the backward
    needs.
    """
    rows, length, n_writes = w_idx.shape
    n_reads = r_idx.shape[-1]
    d_v = table.shape[-1]
    out = v.new_empty(rows, length, d_v)
    delta = v.new_empty(rows, length, d_v)
    if save:
        save_w = v.new_empty(rows, length, n_writes, d_v)
        save_r = v.new_empty(rows, length, n_reads, d_v)
        err = v.new_empty(rows, length, d_v)
    else:
        save_w = save_r = err = out  # never touched: SAVE is a constexpr
    blocks, grid, warps = _walk_launch(rows, chunk, n_writes, n_reads, d_v, v.device)
    _walk_fwd_kernel[grid](
        table, w_idx, r_idx, prepared["dest"],
        prepared["w_order"], prepared["w_lo"], prepared["w_hi"],
        v, beta, prepared["w_weight"], prepared["r_weight"], prepared["transform"],
        prepared["qk"], prepared["decay_end"], prepared["coef"],
        out, save_w, save_r, delta, err,
        length // chunk, length,
        D=d_v, W=n_writes, R=n_reads, C=chunk, SAVE=save, num_warps=warps,
        **blocks,
    )
    return dict(out=out, save_w=save_w, save_r=save_r, delta=delta, err=err)


def _forward_group(
    table: torch.Tensor,
    w_idx: torch.Tensor,
    r_idx: torch.Tensor,
    w_val: torch.Tensor,
    w_logd: torch.Tensor,
    r_val: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    chunk: int,
    save: bool,
) -> dict[str, torch.Tensor]:
    """Both halves of a group on the current stream -- for callers and tests
    that want one group in isolation.  The driver pipelines them instead."""
    prepared = _prepare_group(w_idx, r_idx, w_val, w_logd, r_val, beta, chunk, table.shape[0])
    walked = _walk_group(table, w_idx, r_idx, v, beta, prepared, chunk, save)
    return {**prepared, **walked} if save else {"out": walked["out"]}


_SIDE_STREAMS: dict[int, torch.cuda.Stream] = {}


def _side_stream(device: torch.device) -> torch.cuda.Stream:
    """One second stream per device, for the work that does not wait on the walk."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _SIDE_STREAMS:
        _SIDE_STREAMS[index] = torch.cuda.Stream(device=index)
    return _SIDE_STREAMS[index]


def _cross(tensors: Iterable[object], stream: torch.cuda.Stream) -> None:
    """Tell the allocator these tensors are also used on ``stream``.

    Without it, a tensor freed on the stream that made it can be handed out
    again while ``stream`` is still reading it.
    """
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
            tensor.record_stream(stream)


_SAVED = (
    "a", "transform", "qk", "dest", "coef", "w_weight", "r_weight", "decay_end",
    "g_w", "g_r", "g_end", "save_w", "save_r", "delta", "err",
)


class _TritonGroup(torch.autograd.Function):
    """One group of chunks -- terms, solve and walk -- as one autograd node.

    Plugs into :class:`~lumen.sdm.reference._Arena`'s token chain the way the
    reference's own gather and write do: the previous token in, the next token
    out, the table gradient handed down the chain by reference.  Its inputs are
    the layer's own (weights, log-decays, values, write strengths), so nothing
    of the dense pairwise layout ever enters autograd.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        token: torch.Tensor,
        arena: ref._Arena,
        w_idx: torch.Tensor,
        r_idx: torch.Tensor,
        w_val: torch.Tensor,
        w_logd: torch.Tensor,
        r_val: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        chunk: int,
        prepared: dict[str, torch.Tensor],
        last: bool,
    ):
        """``prepared`` is the group's table-free half, made ahead by the driver
        (outside autograd, which never sees it); ``last`` marks the group whose
        backward runs last, which waits for the second stream."""
        inputs = (w_idx, r_idx, w_val, w_logd, r_val, v, beta)
        walked = _walk_group(arena.table, w_idx, r_idx, v, beta, prepared, chunk, save=True)
        found = {**prepared, **walked}
        ctx.save_for_backward(*inputs, *(found[name] for name in _SAVED))
        ctx.table_shape = arena.table.shape
        ctx.chunk = chunk
        ctx.last = last
        ctx.set_materialize_grads(False)
        return walked["out"], ref._token(arena.table)

    @staticmethod
    @once_differentiable
    def backward(ctx, g_out, grad):  # type: ignore[override]
        w_idx, r_idx, w_val, w_logd, r_val, v, beta, *rest = ctx.saved_tensors
        saved = dict(zip(_SAVED, rest))
        chunk = ctx.chunk
        rows, length, n_writes = w_idx.shape
        n_reads = r_idx.shape[-1]
        d_v_width = v.shape[-1]
        n = length // chunk
        fold = (rows * n, chunk)
        if grad is None:
            grad = v.new_zeros(ctx.table_shape)
        g_out = v.new_zeros(v.shape) if g_out is None else g_out.contiguous()
        segments = _segments(
            w_idx.view(*fold, n_writes), r_idx.view(*fold, n_reads), ctx.table_shape[0],
            reads_too=True,
        )

        w_order, w_lo, w_hi, r_lo, r_hi, r_order, wr_lo, wr_hi, rr_lo, rr_hi = segments

        # ── what the reads hand back: no walk needed, so all at once ───────
        read_sums = v.new_empty(rows, length, n_reads, d_v_width)
        read_owner = torch.empty(rows, length, n_reads, device=v.device, dtype=torch.int8)
        # Parallel, not a chain: wide column blocks, so each program's index
        # loads serve more columns.
        block_c, block_d = _next_pow2(chunk), min(32, _next_pow2(d_v_width))
        _read_sums_kernel[(rows * n, triton.cdiv(d_v_width, block_d))](
            saved["r_weight"], g_out, r_order, r_lo, r_hi, rr_lo, rr_hi,
            read_sums, read_owner,
            D=d_v_width, R=n_reads, C=chunk,
            BC=block_c, BR=min(_next_pow2(n_reads), max(1, 8192 // (block_c * block_d))),
            BD=block_d, num_warps=4,
        )

        # ── the walk, in reverse: only the table gradient is sequential ────
        g_slot = v.new_empty(rows, length, n_writes, d_v_width)
        g_delta = torch.empty_like(v)
        g_u = torch.empty_like(v)
        blocks, grid, warps = _walk_launch(rows, chunk, n_writes, n_reads, d_v_width, v.device)
        _walk_bwd_kernel[grid](
            grad, w_idx, r_idx, saved["dest"],
            w_order, w_lo, w_hi, r_order, wr_lo, wr_hi, read_sums, read_owner,
            beta, saved["w_weight"], saved["r_weight"], saved["transform"], saved["qk"],
            saved["decay_end"], saved["coef"], g_out,
            g_slot, g_delta, g_u,
            n, length,
            D=d_v_width, W=n_writes, R=n_reads, C=chunk, num_warps=warps, **blocks,
        )
        del read_sums  # the backward's largest transient, used by the walk alone

        # ── everything after the walk waits on this group's walk only ─────
        # So it runs on the second stream, beside the next group's walk, and
        # the stream the caller sees waits for it once, after the last group.
        main = torch.cuda.current_stream()
        side = _side_stream(v.device)
        side.wait_stream(main)
        _cross((*ctx.saved_tensors, g_out, g_slot, g_delta, g_u, *segments), side)
        with torch.cuda.stream(side):
            d_k, d_l, d_q, d_v, d_beta = _after_walk(
                ctx, saved, w_val, r_val, v, beta, g_out, g_slot, g_delta, g_u, segments,
            )
        _cross((d_k, d_l, d_q, d_v, d_beta), main)
        if ctx.last:
            main.wait_stream(side)
        return (
            grad, None, None, None,
            d_k, d_l, d_q, d_v, d_beta.view(beta.shape),
            None, None, None,
        )


def _after_walk(
    ctx: Any,
    saved: dict[str, torch.Tensor],
    w_val: torch.Tensor,
    r_val: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_out: torch.Tensor,
    g_slot: torch.Tensor,
    g_delta: torch.Tensor,
    g_u: torch.Tensor,
    segments: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    """A group's backward once its walk has passed: the width reductions, the
    solve's backward and the terms' backward.  Runs on whichever stream is
    current; nothing in it depends on another group."""
    chunk = ctx.chunk
    rows, length, n_writes = g_slot.shape[:3]
    n_reads = r_val.shape[-1]
    d_v_width = v.shape[-1]
    n = length // chunk
    fold = (rows * n, chunk)
    w_order, w_lo, w_hi, r_lo, r_hi, r_order, wr_lo, wr_hi, _, _ = segments
    new = v.new_empty
    d_transform, d_qk = new(rows * n, chunk, chunk), new(rows * n, chunk, chunk)
    d_beta, d_v = new(beta.shape), new(v.shape)
    d_w_weight, d_decay, d_coef = new(w_val.shape), new(w_val.shape), new(w_val.shape)
    d_r_weight = new(r_val.shape)
    # One program per chunk, looping over the width: 32 columns and an
    # 8192-float tile were the fastest of those tried at every shape.
    block_c = _next_pow2(chunk)
    block_d = 32
    entries = max(1, 8192 // (block_c * block_d))
    _walk_grads_kernel[(rows * n,)](
        saved["save_w"], saved["save_r"], g_slot, saved["err"], saved["delta"],
        g_out, g_delta, g_u, beta, saved["dest"],
        d_transform, d_qk, d_beta, d_v, d_w_weight, d_r_weight, d_decay, d_coef,
        D=d_v_width, W=n_writes, R=n_reads, C=chunk,
        BC=block_c, BW=min(_next_pow2(n_writes), entries),
        BR=min(_next_pow2(n_reads), entries), BD=block_d,
        DOT=block_c >= 16, num_warps=4,
    )
    beta_c = beta.view(*fold)
    transform, a = saved["transform"], saved["a"]

    # The solve: X = (I + B)⁻¹ with B = diag(β) A strictly lower, so
    # dB = −Xᵀ dX Xᵀ on the strict lower triangle -- what autograd through
    # the unitriangular solve gives the reference.
    transform_t = transform.transpose(-1, -2)
    d_b = torch.tril(-(transform_t @ d_transform @ transform_t), diagonal=-1)
    d_beta = d_beta.view(*fold) + (d_b * a).sum(-1)
    d_a = beta_c.unsqueeze(-1) * d_b

    # ── the terms ─────────────────────────────────────────────────────
    scratch_g, scratch_end = torch.empty_like(w_val), torch.empty_like(w_val)
    scratch_gr = torch.empty_like(r_val)
    d_k, d_l, d_q = torch.empty_like(w_val), torch.empty_like(w_val), torch.empty_like(r_val)
    _terms_bwd_kernel[(rows * n,)](
        w_val, r_val, w_order, w_lo, w_hi, r_lo, r_hi, r_order, wr_lo, wr_hi,
        saved["g_w"], saved["g_r"], saved["g_end"],
        *(x.contiguous() for x in (d_a, d_qk, d_w_weight, d_r_weight, d_decay, d_coef)),
        scratch_g, scratch_end, scratch_gr,
        d_k, d_l, d_q,
        C=chunk, W=n_writes, R=n_reads,
        **_terms_blocks(chunk, n_writes, n_reads),
    )
    return d_k, d_l, d_q, d_v, d_beta


# ── the driver ────────────────────────────────────────────────────────────


#: The longest chunk the kernels take.  The walk holds a chunk's `(C, C)`
#: solve as one tile, and ``tl.dot`` stages it in shared memory: 16 KB at 64,
#: 64 KB at 128, past the 48 KB a block gets on the bench card.  Longer chunks
#: were also the slowest measured there, so they run the reference.
MAX_CHUNK = 64


def usable(tensor: torch.Tensor) -> bool:
    """Can the Triton path run on this tensor?  CUDA-only and fp32, as Undertow's."""
    return HAS_TRITON and tensor.is_cuda and tensor.dtype is torch.float32


def chunk_sparse_delta(
    memory: torch.Tensor,
    write_idx: torch.Tensor,
    write_val: torch.Tensor,
    log_decay: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    read_idx: torch.Tensor,
    read_val: torch.Tensor,
    chunk_size: int,
    check_writes: bool = True,
    group: int | None = None,
    recompute_pairwise: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`lumen.sdm.reference.chunk_sparse_delta` as kernels.

    Same signature less ``in_place`` (the table is always held in place here),
    same shapes, same preconditions.  ``recompute_pairwise`` is accepted and
    has nothing to do: no pairwise array is kept, or formed.

    Under ``torch.vmap`` **without gradients** the kernels still run: the
    mapped axis becomes more streams (see :func:`_forward_op`).  With
    gradients under a transform, on a CPU tensor, in any dtype but fp32, or for
    a chunk longer than ``MAX_CHUNK``, the reference runs instead -- rather
    than a refusal.
    """
    args = (memory, write_idx, write_val, log_decay, v, beta, read_idx, read_val)
    floats = (memory, write_val, log_decay, v, beta, read_val)
    runs_here = usable(v) and usable(memory) and chunk_size <= MAX_CHUNK
    if runs_here and ref._transforms_active():
        wants_grad = torch.is_grad_enabled() and any(x.requires_grad for x in floats)
        if not wants_grad:
            if check_writes:
                ref._check_distinct_writes(write_idx)
            return _forward_op(*args, chunk_size, 0 if group is None else group)
        runs_here = False
    if not runs_here:
        return ref.chunk_sparse_delta(
            *args, chunk_size=chunk_size, check_writes=check_writes, group=group,
            recompute_pairwise=recompute_pairwise,
        )
    return _drive(*args, chunk_size=chunk_size, check_writes=check_writes, group=group)


def _drive(
    memory: torch.Tensor,
    write_idx: torch.Tensor,
    write_val: torch.Tensor,
    log_decay: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    read_idx: torch.Tensor,
    read_val: torch.Tensor,
    chunk_size: int,
    check_writes: bool,
    group: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The kernels' driver: flatten, pad, group, walk.  Plain tensors only."""
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    if group is not None and group < 1:
        raise ValueError(f"group must be >= 1, got {group}")
    ref._check_shapes(memory, write_idx, write_val, log_decay, v, beta, read_idx, read_val)
    if check_writes:
        ref._check_distinct_writes(write_idx)

    *lead, seq_len, n_writes = write_idx.shape
    n_reads = read_idx.shape[-1]
    n_slots, d_v = memory.shape[-2:]
    rows = math.prod(lead)
    device = write_idx.device
    n_real = rows * n_slots

    offset = (torch.arange(rows, device=device) * n_slots).view(rows, 1, 1)
    write_idx = write_idx.reshape(rows, seq_len, n_writes) + offset
    read_idx = read_idx.reshape(rows, seq_len, n_reads) + offset
    write_val = write_val.reshape(rows, seq_len, n_writes)
    log_decay = log_decay.reshape(rows, seq_len, n_writes)
    read_val = read_val.reshape(rows, seq_len, n_reads)
    v = v.reshape(rows, seq_len, d_v)
    beta = beta.reshape(rows, seq_len)

    # Padding exactly as the reference pads -- see there for why it is exact.
    remainder = seq_len % chunk_size
    if remainder:
        pad = chunk_size - remainder
        filler = torch.arange(n_writes, device=device) + offset
        write_idx = torch.cat([write_idx, filler.expand(rows, pad, n_writes)], dim=1)
        read_idx = torch.cat([read_idx, offset.expand(rows, pad, n_reads)], dim=1)
        write_val, log_decay, read_val, v = (
            torch.cat([x, x.new_zeros(rows, pad, x.shape[-1])], dim=1)
            for x in (write_val, log_decay, read_val, v)
        )
        beta = torch.cat([beta, beta.new_zeros(rows, pad)], dim=1)
    if group is None:
        # The reference's budget, kept: what a group holds is still sized by
        # it -- the saved rows now, rather than the pairwise arrays.
        group = ref._group_size(rows, chunk_size, max(n_writes, n_reads))

    def groups(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return torch.split(x, group * chunk_size, dim=1)

    entering = memory.expand(*lead, n_slots, d_v)
    differentiable = torch.is_grad_enabled() and any(
        x.requires_grad for x in (memory, write_val, log_decay, v, beta, read_val)
    )
    arena = ref._Arena(entering, 0) if differentiable else None
    table = arena.table if arena is not None else entering.reshape(n_real, d_v).clone()

    parts = [
        tuple(x.contiguous() for x in group_parts)
        for group_parts in zip(
            *(groups(x) for x in (write_idx, read_idx, write_val, log_decay, read_val, v, beta))
        )
    ]
    # The table-free half of group g+1 is made on a second stream while the
    # walk of group g holds the first: it depends on nothing the walk writes.
    main = torch.cuda.current_stream()
    side = _side_stream(device)
    side.wait_stream(main)

    def prepare(group_parts: tuple[torch.Tensor, ...]) -> tuple[dict[str, torch.Tensor], torch.cuda.Event]:
        w_idx, r_idx, w_val, w_logd, r_val, _, beta = group_parts
        with torch.cuda.stream(side), torch.no_grad():
            prepared = _prepare_group(w_idx, r_idx, w_val, w_logd, r_val, beta, chunk_size, n_real)
            ready = torch.cuda.Event()
            ready.record(side)
        _cross(group_parts, side)
        return prepared, ready

    outputs = []
    pending = prepare(parts[0])
    for index, group_parts in enumerate(parts):
        prepared, ready = pending
        if index + 1 < len(parts):
            pending = prepare(parts[index + 1])
        main.wait_event(ready)
        _cross(prepared.values(), main)
        w_idx, r_idx, w_val, w_logd, r_val, v_group, beta_group = group_parts
        if arena is not None:
            out, arena.token = _TritonGroup.apply(
                arena.token, arena, *group_parts, chunk_size, prepared, index == 0
            )
        else:
            out = _walk_group(
                table, w_idx, r_idx, v_group, beta_group, prepared, chunk_size, save=False
            )["out"]
        outputs.append(out)

    padded_len = write_idx.shape[1]
    out = torch.cat(outputs, dim=1).reshape(rows, padded_len, d_v)[:, :seq_len]
    final = arena.result(n_real) if arena is not None else table
    return out.reshape(*lead, seq_len, d_v), final.reshape(*lead, n_slots, d_v)


# ── vmap without gradients: the mapped axis is more streams ──────────────
#
# A ``torch.func`` transform cannot see into a Triton launch, so the forward is
# a custom op with its own batching rule.  The rule is the kernels' native
# shape: any leading axes are streams, so the mapped axis moves to the front
# and the whole call runs once.  Forward only -- with gradients under a
# transform the reference's functional holder runs, which every transform
# accepts.


@torch.library.custom_op("lumen::sdm_triton_forward", mutates_args=())
def _forward_op(
    memory: torch.Tensor,
    write_idx: torch.Tensor,
    write_val: torch.Tensor,
    log_decay: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    read_idx: torch.Tensor,
    read_val: torch.Tensor,
    chunk_size: int,
    group: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        return _drive(
            memory, write_idx, write_val, log_decay, v, beta, read_idx, read_val,
            chunk_size=chunk_size, check_writes=False, group=group or None,
        )


@_forward_op.register_fake
def _(
    memory: torch.Tensor,
    write_idx: torch.Tensor,
    write_val: torch.Tensor,
    log_decay: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    read_idx: torch.Tensor,
    read_val: torch.Tensor,
    chunk_size: int,
    group: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    lead = write_idx.shape[:-2]
    n_slots, d_v = memory.shape[-2:]
    return v.new_empty(*lead, write_idx.shape[-2], d_v), memory.new_empty(*lead, n_slots, d_v)


@_forward_op.register_vmap
def _(
    info: Any,
    in_dims: tuple[int | None, ...],
    memory: torch.Tensor,
    write_idx: torch.Tensor,
    write_val: torch.Tensor,
    log_decay: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    read_idx: torch.Tensor,
    read_val: torch.Tensor,
    chunk_size: int,
    group: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[int, int]]:
    size = info.batch_size

    def front(x: torch.Tensor, dim: int | None) -> torch.Tensor:
        return x.movedim(dim, 0) if dim is not None else x.expand(size, *x.shape)

    tensors = [
        front(x, dim)
        for x, dim in zip(
            (memory, write_idx, write_val, log_decay, v, beta, read_idx, read_val), in_dims[:8]
        )
    ]
    # A shared table has fewer leading axes than the streams; its own mapped
    # axis now leads, and the axes it broadcasts over go in after it.
    while tensors[0].dim() < tensors[1].dim():
        tensors[0] = tensors[0].unsqueeze(1)
    return _forward_op(*tensors, chunk_size, group), (0, 0)
