"""Triton kernels for Sparse Delta Memory's sequential loop — optional, opt-in, measured.

Importing this module never fails.  ``HAS_TRITON`` reports whether the kernels
are usable; everything degrades to :mod:`lumen.sdm.reference` when it is False.
Triton is a runtime capability here, not a dependency.

What the kernels are for
------------------------
The reference splits a chunk into what needs the table and what does not
(:func:`~lumen.sdm.reference._chunk_terms` / :func:`~lumen.sdm.reference._chunk_apply`),
batches the second over a group of chunks, and walks the first one chunk at a
time.  That walk is a dozen small torch ops per chunk -- gathers, three batched
products, a write -- and on a long sequence it is most of a training step, spent
mostly launching kernels rather than running them.

Here the walk is one kernel launch per group.  Each program owns one stream's
table (a row of the flattened batch) and a block of its value columns, and loops
over the group's chunks in order: gather the chunk's rows, retrieve, solve, read
out, write the new rows back -- with a barrier between the reads and the writes
of a chunk and another before the next chunk reads.  **Columns of the table never
mix** -- every operation in the recurrence is per value column -- so programs
that share a stream but not a column block never need to talk to each other.

The backward is the same walk in reverse, and only the part of it that is
sequential: the table's gradient.  Everything else the backward needs --
gradients for the chunk's terms, all reductions over the value width -- is a
batched product over the group once the walk has passed, as cuBLAS likes it.

Numerics
--------
The same arithmetic as :func:`~lumen.sdm.reference._chunk_apply` in fp32, with
different summation orders: round-off apart, not bit-identical.  The fp64 oracle
is what both answer to.  The backward accumulates the table gradient with
atomics where a chunk touches one slot from several entries, so like the
reference's own backward on CUDA it is not bit-deterministic run to run.
"""

from __future__ import annotations

import math

import torch
from torch.autograd.function import once_differentiable
from torch.utils.checkpoint import checkpoint

from lumen.gdn.reference import inv_unit
from lumen.sdm import reference as ref

#: Which code computes the table-free terms: ``"triton"`` (the kernels below)
#: or ``"torch"`` (the reference's own, so the walk can be measured alone).
TERMS = "triton"

try:  # pragma: no cover - trivially environment-dependent
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover
    HAS_TRITON = False


def _next_pow2(n: int) -> int:
    return 1 << (max(n, 1) - 1).bit_length()


if HAS_TRITON:

    @triton.jit
    def _red_add(ptr, val, mask):
        """``*ptr += val`` where ``mask``, atomically: a predicated PTX ``red``.

        Not ``tl.atomic_add``: Triton 3.5 always gives its atomics a memory-order
        qualifier (``.relaxed``, ``.acq_rel``, …), which ptxas refuses below
        sm_70 -- every ``sem`` and ``scope`` fails to assemble on Pascal.  A bare
        ``red.global.add.f32`` has been there since sm_20 and is all a scatter-add
        needs; ordering against the next chunk is the barrier's job.
        """
        tl.inline_asm_elementwise(
            asm="""{
            .reg .pred p;
            setp.ne.b32 p, $3, 0;
            @p red.global.add.f32 [$1], $2;
            mov.b32 $0, 0;
            }""",
            constraints="=r,l,f,r",
            args=[ptr.to(tl.int64, bitcast=True), val, mask.to(tl.int32)],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )

    @triton.jit
    def _scan_fwd_kernel(
        TABLE, W_IDX, R_IDX, DEST,
        V, BETA, WW, RW, X, Q, DE, K,
        OUT, SAVE_W, SAVE_R, DELTA, ERR,
        n_chunks, L,
        D: tl.constexpr, W: tl.constexpr, R: tl.constexpr, C: tl.constexpr,
        BC: tl.constexpr, BW: tl.constexpr, BR: tl.constexpr, BD: tl.constexpr,
        SAVE: tl.constexpr,
    ):
        """One stream, one block of value columns, every chunk of a group in order.

        Flat layouts, one row of the batch per stream: positions ``(P, L)``,
        entries ``(P, L, W|R)``, the carry ``(P, L, W, C)``, the solve and the
        read-out ``(P, L, C)``.  ``TABLE`` is ``(P·N, D)`` and written in place.
        """
        p = tl.program_id(0)
        d = tl.program_id(1) * BD + tl.arange(0, BD)
        d_ok = d < D
        t = tl.arange(0, BC)
        t_ok = t < C
        cc_ok = t_ok[:, None] & t_ok[None, :]
        td_ok = t_ok[:, None] & d_ok[None, :]
        # (position, entry, column) for the gathers ...
        t3 = t[:, None, None]
        d3 = d[None, None, :]
        d3_ok = d_ok[None, None, :]
        w3 = tl.arange(0, BW)[None, :, None]
        r3 = tl.arange(0, BR)[None, :, None]
        # ... and (position·entry, ·) for the carry product, m = t·BW + w.
        m = tl.arange(0, BC * BW)
        mt = m // BW
        mw = m % BW

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
            delta = tl.dot(transform, u, input_precision="ieee")
            qk = tl.load(Q + pos[:, None] * C + t[None, :], mask=cc_ok, other=0.0)
            out = read_back + tl.dot(qk, delta, input_precision="ieee")
            tl.store(OUT + pos[:, None] * D + d[None, :], out, mask=td_ok)
            if SAVE:
                tl.store(DELTA + pos[:, None] * D + d[None, :], delta, mask=td_ok)
                tl.store(ERR + pos[:, None] * D + d[None, :], err, mask=td_ok)

            # Every read this chunk makes of the table lands before any write.
            tl.debug_barrier()

            # ── the new rows, placed by each slot's first writer ───────────
            for w0 in range(0, W, BW):
                ok = (mt < C) & (w0 + mw < W)
                entry = (base + mt) * W + w0 + mw
                dest = tl.load(DEST + entry, mask=ok, other=-1)
                carry = tl.load(
                    K + entry[:, None] * C + t[None, :],
                    mask=ok[:, None] & t_ok[None, :],
                    other=0.0,
                )
                moved = tl.dot(carry, delta, input_precision="ieee")
                decay = tl.load(DE + entry, mask=ok, other=0.0)
                if SAVE:
                    rows = tl.load(
                        SAVE_W + entry[:, None] * D + d[None, :],
                        mask=ok[:, None] & d_ok[None, :],
                        other=0.0,
                    )
                else:
                    # Safe to gather again: only a slot's first writer stores,
                    # each slot has one, so no earlier block wrote this row.
                    idx = tl.load(W_IDX + entry, mask=ok, other=0)
                    rows = tl.load(
                        TABLE + idx[:, None] * D + d[None, :],
                        mask=ok[:, None] & d_ok[None, :],
                        other=0.0,
                        cache_modifier=".cg",
                    )
                new_rows = decay[:, None] * rows + moved
                writes = ok & (dest >= 0)
                tl.store(
                    TABLE + dest[:, None] * D + d[None, :],
                    new_rows,
                    mask=writes[:, None] & d_ok[None, :],
                )

            # And every write lands before the next chunk reads.
            tl.debug_barrier()

    @triton.jit
    def _scan_bwd_kernel(
        GT, W_IDX, R_IDX, DEST,
        BETA, WW, RW, X, Q, DE, K, G_OUT,
        GNR, GDELTA, GU,
        n_chunks, L,
        D: tl.constexpr, W: tl.constexpr, R: tl.constexpr, C: tl.constexpr,
        BC: tl.constexpr, BW: tl.constexpr, BR: tl.constexpr, BD: tl.constexpr,
    ):
        """The table gradient, carried back through a group's chunks in reverse.

        ``GT`` is the gradient of the table as the group exits, `(P·N, D)`, and
        leaves as the gradient of the table as it entered.  Per chunk, the
        gradient of a replaced row is read at its destination and the row is
        handed back -- its old value reached the new row through the decay --
        then every row the chunk gathered gets its share added in.
        """
        p = tl.program_id(0)
        d = tl.program_id(1) * BD + tl.arange(0, BD)
        d_ok = d < D
        t = tl.arange(0, BC)
        t_ok = t < C
        cc_ok = t_ok[:, None] & t_ok[None, :]
        td_ok = t_ok[:, None] & d_ok[None, :]
        t3 = t[:, None, None]
        d3 = d[None, None, :]
        d3_ok = d_ok[None, None, :]
        w3 = tl.arange(0, BW)[None, :, None]
        r3 = tl.arange(0, BR)[None, :, None]
        m = tl.arange(0, BC * BW)
        mt = m // BW
        mw = m % BW

        for i in range(n_chunks):
            c = n_chunks - 1 - i
            base = p * L + c * C
            pos = base + t

            g_out = tl.load(G_OUT + pos[:, None] * D + d[None, :], mask=td_ok, other=0.0)
            # Qᵀ and Xᵀ by addressing: element [i, j] is Q[j, i].
            qk_t = tl.load(Q + (base + t[None, :]) * C + t[:, None], mask=cc_ok, other=0.0)
            g_delta = tl.dot(qk_t, g_out, input_precision="ieee")

            # ── the writes: gradient of each new row, then the row handed back
            for w0 in range(0, W, BW):
                ok = (mt < C) & (w0 + mw < W)
                entry = (base + mt) * W + w0 + mw
                dest = tl.load(DEST + entry, mask=ok, other=-1)
                writes = ok & (dest >= 0)
                g_new = tl.load(
                    GT + dest[:, None] * D + d[None, :],
                    mask=writes[:, None] & d_ok[None, :],
                    other=0.0,
                    cache_modifier=".cg",
                )
                tl.store(
                    GNR + entry[:, None] * D + d[None, :],
                    g_new,
                    mask=ok[:, None] & d_ok[None, :],
                )
                decay = tl.load(DE + entry, mask=ok, other=0.0)
                # The replaced row's gradient is what its decayed self carried
                # into the new row -- a store, not an add: nothing else in this
                # chunk has touched the destination yet.
                tl.store(
                    GT + dest[:, None] * D + d[None, :],
                    decay[:, None] * g_new,
                    mask=writes[:, None] & d_ok[None, :],
                )
                carry_t = tl.load(
                    K + entry[None, :] * C + t[:, None],
                    mask=t_ok[:, None] & ok[None, :],
                    other=0.0,
                )
                g_delta += tl.dot(carry_t, g_new, input_precision="ieee")

            transform_t = tl.load(X + (base + t[None, :]) * C + t[:, None], mask=cc_ok, other=0.0)
            g_u = tl.dot(transform_t, g_delta, input_precision="ieee")
            beta = tl.load(BETA + pos, mask=t_ok, other=0.0)
            g_ret = -beta[:, None] * g_u
            tl.store(GDELTA + pos[:, None] * D + d[None, :], g_delta, mask=td_ok)
            tl.store(GU + pos[:, None] * D + d[None, :], g_u, mask=td_ok)

            # Every destination is read and handed back before any gather's
            # gradient is added: one may land on the other.
            tl.debug_barrier()

            # ── the gathers: each entry's row gets its share, added ─────────
            g_ret3 = g_ret[:, None, :]
            for w0 in range(0, W, BW):
                ok = (t3 < C) & (w0 + w3 < W)
                entry = (base + t3) * W + w0 + w3
                idx = tl.load(W_IDX + entry, mask=ok, other=0)
                weight = tl.load(WW + entry, mask=ok, other=0.0)
                _red_add(GT + idx * D + d3, weight * g_ret3, ok & d3_ok)
            g_out3 = g_out[:, None, :]
            for r0 in range(0, R, BR):
                ok = (t3 < C) & (r0 + r3 < R)
                entry = (base + t3) * R + r0 + r3
                idx = tl.load(R_IDX + entry, mask=ok, other=0)
                weight = tl.load(RW + entry, mask=ok, other=0.0)
                _red_add(GT + idx * D + d3, weight * g_out3, ok & d3_ok)

            tl.debug_barrier()


if HAS_TRITON:

    # ── the table-free terms ──────────────────────────────────────────────
    #
    # One program per chunk of one stream.  The reference finds every entry's
    # partner at every position -- a dense `(C, K, C)` array -- and makes a
    # dozen passes over arrays of that shape in memory.  Here a chunk's writes
    # are sorted once by `slot·C + position` (in torch, integers only), so the
    # writes that share a slot are one contiguous *segment*, in position order.
    # Each entry then walks its own segment: the partners it actually has, one
    # or two at the access statistics product keys produce, up to `C` in the
    # worst case -- never the `C` cells per entry of the dense layout.
    #
    # Everything a pair contributes is formed in registers.  `A` and `QK` are
    # built a row at a time -- one position's entries, their partners' columns
    # selected by a one-hot -- and the carry a block of rows at a time, so no
    # scatter-add exists anywhere and the result is deterministic.  The backward
    # is the same walk: a pair's far end is handled from the far entry's own
    # segment (sharing a slot is symmetric), and a write's reads from a second
    # sort, of the chunk's reads.

    @triton.jit
    def _terms_fwd_kernel(
        WS, KV, LD, QV,
        WORDER, WLO, WHI, RLO, RHI,
        GW, GR, GEND,
        A, QK, CARRY, WWEIGHT, RWEIGHT, DECAY, DEST,
        C: tl.constexpr, W: tl.constexpr, R: tl.constexpr,
        BC: tl.constexpr, BW: tl.constexpr, BR: tl.constexpr,
    ):
        """:func:`~lumen.sdm.reference._chunk_terms` for one chunk, less the solve.

        Flat per-chunk layouts: write entries `e = t·W + w`, read entries
        `f = t·R + r`.  ``WORDER`` is the chunk's write entries sorted by
        `slot·C + t`; ``WLO``/``WHI`` (and ``RLO``/``RHI`` for reads) bound the
        segment of writes to each entry's slot.  Returns `A` rather than the
        transform: `(I + diag(β) A)⁻¹` is one batched library call, and its
        backward is autograd's.
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
        # Members arrive in position order and stop counting past `t` -- so two
        # entries that share a history add the same terms in the same order and
        # agree bit for bit, the property the reference keeps for the same
        # reason: a read with nothing written between it and its partner must
        # see an exponent of exactly 0.
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

        # ── pass 2: A, QK and the carry, a row of positions at a time ──────
        for t in range(C):
            e = t * W + wv
            lo = tl.load(WLO + cw + e, mask=w_ok, other=0)
            n = tl.load(WHI + cw + e, mask=w_ok, other=0) - lo
            k = tl.load(KV + cw + e, mask=w_ok, other=0.0)
            g = tl.load(GW + cw + e, mask=w_ok, other=0.0)
            g_end = tl.load(GEND + cw + e, mask=w_ok, other=0.0)
            a_row = tl.zeros((BC,), dtype=tl.float32)
            carry = tl.zeros((BW, BC), dtype=tl.float32)
            for j in range(tl.max(n, axis=0)):
                valid = j < n
                m = tl.load(WORDER + cw + lo + j, mask=valid, other=0)
                s = m // W
                k_m = tl.load(KV + cw + m, mask=valid, other=0.0)
                g_m = tl.load(GW + cw + m, mask=valid, other=0.0)
                hit = (s[:, None] == cols[None, :]) & valid[:, None]
                # A[t, s] = Σ_w k · k_m · exp(G − G_m), s < t; every factor in
                # (0, 1] by construction, as the reference forms it.
                pair = valid & (s < t)
                a = tl.where(pair, k * k_m * tl.exp(tl.where(pair, g - g_m, 0.0)), 0.0)
                a_row += tl.sum(tl.where(hit, a[:, None], 0.0), axis=0)
                carried = tl.where(valid, k_m * tl.exp(tl.where(valid, g_end - g_m, 0.0)), 0.0)
                carry += tl.where(hit, carried[:, None], 0.0)
            tl.store(A + (pc * C + t) * C + cols, a_row, mask=c_ok)
            tl.store(
                CARRY + (cw + e[:, None]) * C + cols[None, :],
                carry,
                mask=w_ok[:, None] & c_ok[None, :],
            )

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
                k_m = tl.load(KV + cw + m, mask=valid, other=0.0)
                g_m = tl.load(GW + cw + m, mask=valid, other=0.0)
                pair = valid & (s <= t)
                qk = tl.where(pair, q * k_m * tl.exp(tl.where(pair, g_r - g_m, 0.0)), 0.0)
                hit = (s[:, None] == cols[None, :]) & pair[:, None]
                qk_row += tl.sum(tl.where(hit, qk[:, None], 0.0), axis=0)
            tl.store(QK + (pc * C + t) * C + cols, qk_row, mask=c_ok)

    @triton.jit
    def _terms_bwd_kernel(
        KV, QV, WORDER, WLO, WHI, RLO, RHI, RORDER, WRLO, WRHI,
        GW, GR, GEND,
        DA, DQK, DCARRY, DWW, DRW, DDE,
        SG, SE, SGR,
        DK, DL, DQ,
        C: tl.constexpr, W: tl.constexpr, R: tl.constexpr,
        BW: tl.constexpr, BR: tl.constexpr,
    ):
        """The terms' backward: ``DK``, ``DL``, ``DQ`` from the gradients of every output.

        Every pair has two ends.  An entry's own end is summed over its own
        segment; the far end -- the partner's share of the same pair -- is
        summed by the partner over *its* segment, which holds the same pairs
        because sharing a slot is symmetric.  A write's reads come from
        ``RORDER``, the chunk's reads sorted the same way, bounded by
        ``WRLO``/``WRHI``.  ``SG``, ``SE``, ``SGR`` are scratch for the
        gradients of the cumulative log-decays, which the second pass hands
        back to every write they summed.  No entry is written by two lanes.
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
            # carry[m, t] = k · exp(G_end − G) for every m sharing the slot
            to_end = tl.exp(g_end - g)
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
                # carry[e, s] = k_m · exp(G_end − G_m): this entry's own row
                from_m = tl.exp(tl.where(valid, g_end - g_m, 0.0))
                d_c = tl.load(DCARRY + (cw + e) * C + s, mask=valid, other=0.0)
                d_end += tl.where(valid, d_c * (k_m * from_m), 0.0)
                # carry[m, t]: this entry as the partner m's row carries
                d_c = tl.load(DCARRY + (cw + m) * C + t, mask=valid, other=0.0)
                d_k += tl.where(valid, d_c * to_end, 0.0)
                d_g -= tl.where(valid, d_c * (k * to_end), 0.0)
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
            decay = tl.exp(g)
            d_ww = tl.load(DWW + cw + e, mask=w_ok, other=0.0)
            d_k += d_ww * decay
            d_g += d_ww * (k * decay)
            d_de = tl.load(DDE + cw + e, mask=w_ok, other=0.0)
            d_end += d_de * tl.exp(g_end)
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
                d_l += tl.where(valid & (e // W >= t), tl.load(SG + cw + e, mask=valid, other=0.0), 0.0)
                d_l += tl.load(SE + cw + e, mask=valid, other=0.0)
            lo2 = tl.load(WRLO + cw + m, mask=w_ok, other=0)
            n2 = tl.load(WRHI + cw + m, mask=w_ok, other=0) - lo2
            for j in range(tl.max(n2, axis=0)):
                valid = j < n2
                f = tl.load(RORDER + cr + lo2 + j, mask=valid, other=0)
                read = valid & (f // R >= t)
                d_l += tl.load(SGR + cr + f, mask=read, other=0.0)
            tl.store(DL + cw + m, d_l, mask=w_ok)


# ── launch configuration ──────────────────────────────────────────────────


def _blocks(chunk: int, n_writes: int, n_reads: int, d_v: int) -> dict[str, int]:
    """Fixed tile sizes for the walk, not autotuned (see Undertow's kernels for why).

    Measured on one Pascal card: a 64-column block spills and runs ten times
    slower; 32 columns at 8 warps is best wherever there are a few streams'
    worth of programs.  ``_walk_launch`` narrows to 16 when there are not.
    """
    block_c = max(16, _next_pow2(chunk))  # tl.dot wants every side >= 16
    block_d = max(16, min(32, _next_pow2(d_v)))
    # The carry product stages a (C·BW, C) and a (C·BW, BD) operand in shared
    # memory; at 128 rows that is 32 KB, inside the 48 KB a block gets on the
    # bench card.  256 rows needs 64 KB and does not launch there.
    block_w = max(1, min(_next_pow2(n_writes), 128 // block_c))
    block_r = max(1, min(_next_pow2(n_reads), 128 // block_c))
    return dict(BC=block_c, BW=block_w, BR=block_r, BD=block_d)


def _walk_launch(rows: int, chunk: int, n_writes: int, n_reads: int, d_v: int) -> tuple[dict[str, int], tuple[int, int], int]:
    """Blocks, grid and warps for one walk launch.

    The walk is a chain of dependent loads per program, so it is latency that
    bounds it, and only more programs hide latency.  With few streams -- one
    sequence, two heads -- a narrower column block buys programs.
    """
    blocks = _blocks(chunk, n_writes, n_reads, d_v)
    warps = 8
    if rows * triton.cdiv(d_v, blocks["BD"]) < 16 and blocks["BD"] > 16:
        blocks["BD"], warps = 16, 4
    # The backward's atomics meet `(C, BW, BD)` and `(C, BR, BD)` tiles; no
    # replicas, for the reason `_terms_blocks` gives.
    smallest = blocks["BC"] * min(blocks["BW"], blocks["BR"]) * blocks["BD"]
    warps = max(1, min(warps, smallest // 32))
    return blocks, (rows, triton.cdiv(d_v, blocks["BD"])), warps


def _terms_blocks(chunk: int, n_writes: int, n_reads: int) -> dict[str, int]:
    """Tiles and warps for the terms kernels: one position's entries per step."""
    block_w, block_r = _next_pow2(n_writes), _next_pow2(n_reads)
    warps = max(1, min(4, max(block_w, block_r) // 32))
    return dict(BW=block_w, BR=block_r, num_warps=warps)


def _segments(
    w_idx: torch.Tensor, r_idx: torch.Tensor, reads_too: bool
) -> tuple[torch.Tensor, ...]:
    """Each entry's segment of same-slot writes, by one sort per chunk.

    `(P', C, W|R)` slots → ``worder`` `(P', C·W)`, the write entries sorted by
    `slot·C + position` (unique: a position names a slot at most once), and
    each write's and read's ``[lo, hi)`` into it.  With ``reads_too``, the same
    for the reads -- ``rorder`` and each write's ``[lo, hi)`` into it -- which
    the backward needs and the forward does not.  Integers only, no gradient.
    """
    rows, chunk, n_writes = w_idx.shape
    n_reads = r_idx.shape[-1]
    position = torch.arange(chunk, device=w_idx.device).view(1, chunk, 1)
    w_base = (w_idx * chunk).reshape(rows, chunk * n_writes)
    r_base = (r_idx * chunk).reshape(rows, chunk * n_reads)
    w_keys, w_order = torch.sort((w_idx * chunk + position).reshape(rows, -1), dim=-1)
    found = [
        w_order,
        torch.searchsorted(w_keys, w_base),
        torch.searchsorted(w_keys, w_base + chunk),
        torch.searchsorted(w_keys, r_base),
        torch.searchsorted(w_keys, r_base + chunk),
    ]
    if reads_too:
        r_keys, r_order = torch.sort((r_idx * chunk + position).reshape(rows, -1), dim=-1)
        found += [
            r_order,
            torch.searchsorted(r_keys, w_base),
            torch.searchsorted(r_keys, w_base + chunk),
        ]
    return tuple(x.to(torch.int32) for x in found)


class _TritonTerms(torch.autograd.Function):
    """The table-free terms of `P'` chunks, forward and backward in Triton.

    Saves its inputs and the three cumulative log-decays -- `O(C·(W+R))` per
    chunk -- and none of the pairwise arrays, which never exist; so
    ``recompute_pairwise`` has nothing left to recompute on this path.  The
    backward sorts again rather than keep the forward's segments.
    """

    @staticmethod
    def forward(ctx, w_idx, w_val, w_logd, r_idx, r_val):  # type: ignore[override]
        w_idx, w_val, w_logd, r_idx, r_val = (
            x.contiguous() for x in (w_idx, w_val, w_logd, r_idx, r_val)
        )
        rows, chunk, n_writes = w_idx.shape
        n_reads = r_idx.shape[-1]
        w_order, w_lo, w_hi, r_lo, r_hi = _segments(w_idx, r_idx, reads_too=False)
        new = w_val.new_empty
        g_w, g_end, g_r = new(w_val.shape), new(w_val.shape), new(r_val.shape)
        a, qk = new(rows, chunk, chunk), new(rows, chunk, chunk)
        carry = new(rows, chunk, n_writes, chunk)
        w_weight, decay_end, r_weight = new(w_val.shape), new(w_val.shape), new(r_val.shape)
        dest = torch.empty_like(w_idx)
        _terms_fwd_kernel[(rows,)](
            w_idx, w_val, w_logd, r_val,
            w_order, w_lo, w_hi, r_lo, r_hi,
            g_w, g_r, g_end,
            a, qk, carry, w_weight, r_weight, decay_end, dest,
            C=chunk, W=n_writes, R=n_reads, BC=_next_pow2(chunk),
            **_terms_blocks(chunk, n_writes, n_reads),
        )
        ctx.save_for_backward(w_idx, w_val, r_idx, r_val, g_w, g_r, g_end)
        ctx.mark_non_differentiable(dest)
        ctx.set_materialize_grads(False)
        return a, qk, carry, w_weight, r_weight, decay_end, dest

    @staticmethod
    @once_differentiable
    def backward(ctx, d_a, d_qk, d_carry, d_ww, d_rw, d_de, d_dest):  # type: ignore[override]
        w_idx, w_val, r_idx, r_val, g_w, g_r, g_end = ctx.saved_tensors
        rows, chunk, n_writes = w_idx.shape
        n_reads = r_idx.shape[-1]

        def given(grad: torch.Tensor | None, shape: tuple[int, ...]) -> torch.Tensor:
            return w_val.new_zeros(shape) if grad is None else grad.contiguous()

        d_a = given(d_a, (rows, chunk, chunk))
        d_qk = given(d_qk, (rows, chunk, chunk))
        d_carry = given(d_carry, (rows, chunk, n_writes, chunk))
        d_ww = given(d_ww, w_val.shape)
        d_rw = given(d_rw, r_val.shape)
        d_de = given(d_de, w_val.shape)
        segments = _segments(w_idx, r_idx, reads_too=True)
        scratch_g, scratch_end = torch.empty_like(w_val), torch.empty_like(w_val)
        scratch_gr = torch.empty_like(r_val)
        d_k, d_l, d_q = torch.empty_like(w_val), torch.empty_like(w_val), torch.empty_like(r_val)
        _terms_bwd_kernel[(rows,)](
            w_val, r_val, *segments,
            g_w, g_r, g_end,
            d_a, d_qk, d_carry, d_ww, d_rw, d_de,
            scratch_g, scratch_end, scratch_gr,
            d_k, d_l, d_q,
            C=chunk, W=n_writes, R=n_reads,
            **_terms_blocks(chunk, n_writes, n_reads),
        )
        return None, d_k, d_l, None, d_q


def _group_terms(
    w_idx: torch.Tensor,
    w_val: torch.Tensor,
    w_logd: torch.Tensor,
    beta: torch.Tensor,
    r_idx: torch.Tensor,
    r_val: torch.Tensor,
    chunk: int,
) -> tuple[torch.Tensor, ...]:
    """:func:`lumen.sdm.reference._group_terms`, Triton terms, the reference's solve.

    `(P, n·C, …)` inputs → the terms as `(P, n, C, …)`, and each write entry's
    destination: its slot if it is its slot's first writer in the chunk, else
    ``-1`` -- the walk simply does not store those.
    """
    rows, length, _ = w_idx.shape
    n = length // chunk

    def fold(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(rows * n, chunk, *x.shape[2:])

    def unfold(x: torch.Tensor) -> torch.Tensor:
        return x.view(rows, n, *x.shape[1:])

    a, qk, carry, w_weight, r_weight, decay_end, dest = _TritonTerms.apply(
        fold(w_idx), fold(w_val), fold(w_logd), fold(r_idx), fold(r_val)
    )
    transform = inv_unit(fold(beta).unsqueeze(-1) * a)
    return tuple(unfold(x) for x in (w_weight, r_weight, transform, qk, decay_end, carry, dest))


def _launch_fwd(
    table: torch.Tensor,
    w_idx: torch.Tensor,
    r_idx: torch.Tensor,
    dest: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    terms: tuple[torch.Tensor, ...],
    chunk: int,
    save: bool,
) -> tuple[torch.Tensor, ...]:
    """Run one group's walk.  Returns ``out`` and, with ``save``, what the backward needs."""
    w_weight, r_weight, transform, qk, decay_end, carry = terms
    rows, length, n_writes = w_idx.shape
    n_reads = r_idx.shape[-1]
    d_v = table.shape[-1]
    out = v.new_empty(rows, length, d_v)
    if save:
        save_w = v.new_empty(rows, length, n_writes, d_v)
        save_r = v.new_empty(rows, length, n_reads, d_v)
        delta = v.new_empty(rows, length, d_v)
        err = v.new_empty(rows, length, d_v)
    else:
        save_w = save_r = delta = err = out  # never touched: SAVE is a constexpr
    blocks, grid, warps = _walk_launch(rows, chunk, n_writes, n_reads, d_v)
    _scan_fwd_kernel[grid](
        table, w_idx, r_idx, dest,
        v, beta, w_weight, r_weight, transform, qk, decay_end, carry,
        out, save_w, save_r, delta, err,
        length // chunk, length,
        D=d_v, W=n_writes, R=n_reads, C=chunk, SAVE=save,
        num_warps=warps,
        **blocks,
    )
    return out, save_w, save_r, delta, err


class _TritonGroup(torch.autograd.Function):
    """A group's walk over the arena's table, in place, with a hand-carried gradient.

    Plugs into :class:`~lumen.sdm.reference._Arena`'s token chain the way the
    reference's own gather and write do: the previous token in, the next token
    out, the table gradient handed down the chain by reference.  The chunk's
    terms and inputs are ordinary autograd values, so everything upstream of
    them -- the pairwise terms, the solve, the projections -- backpropagates as
    it always has.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        token: torch.Tensor,
        arena: ref._Arena,
        w_idx: torch.Tensor,
        r_idx: torch.Tensor,
        dest: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        w_weight: torch.Tensor,
        r_weight: torch.Tensor,
        transform: torch.Tensor,
        qk: torch.Tensor,
        decay_end: torch.Tensor,
        carry: torch.Tensor,
        chunk: int,
    ):
        table = arena.table
        terms = tuple(
            t.contiguous() for t in (w_weight, r_weight, transform, qk, decay_end, carry)
        )
        v, beta = v.contiguous(), beta.contiguous()
        out, save_w, save_r, delta, err = _launch_fwd(
            table, w_idx, r_idx, dest, v, beta, terms, chunk, save=True
        )
        ctx.save_for_backward(w_idx, r_idx, dest, beta, *terms, save_w, save_r, delta, err)
        ctx.table_shape = table.shape
        ctx.chunk = chunk
        ctx.set_materialize_grads(False)
        return out, ref._token(table)

    @staticmethod
    @once_differentiable
    def backward(ctx, g_out, grad):  # type: ignore[override]
        (
            w_idx, r_idx, dest, beta,
            w_weight, r_weight, transform, qk, decay_end, carry,
            save_w, save_r, delta, err,
        ) = ctx.saved_tensors
        chunk = ctx.chunk
        rows, length, n_writes = w_idx.shape
        n_reads = r_idx.shape[-1]
        d_v = delta.shape[-1]
        n = length // chunk
        if grad is None:
            grad = delta.new_zeros(ctx.table_shape)
        g_out = delta.new_zeros(delta.shape) if g_out is None else g_out.contiguous()

        g_new = delta.new_empty(rows, length, n_writes, d_v)
        g_delta = torch.empty_like(delta)
        g_u = torch.empty_like(delta)
        blocks, grid, warps = _walk_launch(rows, chunk, n_writes, n_reads, d_v)
        _scan_bwd_kernel[grid](
            grad, w_idx, r_idx, dest,
            beta, w_weight, r_weight, transform, qk, decay_end, carry, g_out,
            g_new, g_delta, g_u,
            n, length,
            D=d_v, W=n_writes, R=n_reads, C=chunk,
            num_warps=warps,
            **blocks,
        )

        # Everything else is a reduction over the value width: batched, now
        # that the sequential part has passed.  `(P, n, C, …)` throughout.
        def chunked(x: torch.Tensor) -> torch.Tensor:
            return x.view(rows, n, chunk, *x.shape[2:])

        g_out_c, delta_c, err_c, g_delta_c, g_u_c = map(
            chunked, (g_out, delta, err, g_delta, g_u)
        )
        beta_c = chunked(beta)
        u_c = beta_c.unsqueeze(-1) * err_c
        g_ret_c = -beta_c.unsqueeze(-1) * g_u_c
        save_w_c, save_r_c, g_new_c = map(chunked, (save_w, save_r, g_new))

        g_transform = g_delta_c @ u_c.transpose(-1, -2)
        g_qk = g_out_c @ delta_c.transpose(-1, -2)
        g_beta = (g_u_c * err_c).sum(-1).view(rows, length)
        g_v = (beta_c.unsqueeze(-1) * g_u_c).view(rows, length, d_v)
        g_w_weight = torch.einsum("pncd,pncwd->pncw", g_ret_c, save_w_c)
        g_r_weight = torch.einsum("pncd,pncrd->pncr", g_out_c, save_r_c)
        g_decay = (g_new_c * save_w_c).sum(-1)
        g_carry = torch.einsum("pncwd,pnsd->pncws", g_new_c, delta_c)

        return (
            grad, None, None, None, None,
            g_v, g_beta,
            g_w_weight.view(w_weight.shape),
            g_r_weight.view(r_weight.shape),
            g_transform.view(transform.shape),
            g_qk.view(qk.shape),
            g_decay.view(decay_end.shape),
            g_carry.view(carry.shape),
            None,
        )


# ── the driver ────────────────────────────────────────────────────────────


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
    """:func:`lumen.sdm.reference.chunk_sparse_delta`, with the walk in Triton.

    Same signature less ``in_place`` (the table is always held in place here),
    same shapes, same preconditions, the same terms computed by the same code.
    Falls back to the reference -- which is what runs under a ``torch.func``
    transform, on a CPU tensor, or in any dtype but fp32 -- rather than refusing.
    """
    if (
        not usable(v)
        or not usable(memory)
        or ref._transforms_active()
    ):
        return ref.chunk_sparse_delta(
            memory, write_idx, write_val, log_decay, v, beta, read_idx, read_val,
            chunk_size=chunk_size, check_writes=check_writes, group=group,
            recompute_pairwise=recompute_pairwise,
        )
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    if group is not None and group < 1:
        raise ValueError(f"group must be >= 1, got {group}")
    recompute_pairwise = recompute_pairwise and torch.is_grad_enabled()
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
        group = ref._group_size(rows, chunk_size, max(n_writes, n_reads))

    # Non-first writers get destination -1: nothing past the table to hold them,
    # because the kernel simply does not store them.
    not_first = torch.full((rows, chunk_size, n_writes), -1, device=device, dtype=torch.long)

    def groups(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return torch.split(x, group * chunk_size, dim=1)

    entering = memory.expand(*lead, n_slots, d_v)
    differentiable = torch.is_grad_enabled() and any(
        x.requires_grad for x in (memory, write_val, log_decay, v, beta, read_val)
    )
    arena = ref._Arena(entering, 0) if differentiable else None
    table = arena.table if arena is not None else entering.reshape(n_real, d_v).clone()

    outputs = []
    for w_idx, w_val, w_logd, v_group, beta_group, r_idx, r_val in zip(
        *(
            groups(x)
            for x in (write_idx, write_val, log_decay, v, beta, read_idx, read_val)
        )
    ):
        if TERMS == "triton":
            terms = _group_terms(w_idx, w_val, w_logd, beta_group, r_idx, r_val, chunk_size)
        else:
            inputs = (w_idx, w_val, w_logd, beta_group, r_idx, r_val, not_first)
            if recompute_pairwise:
                terms = checkpoint(
                    ref._group_terms, *inputs, use_reentrant=False, preserve_rng_state=False
                )
            else:
                terms = ref._group_terms(*inputs)
        *float_terms, dest = terms
        w_idx_c, r_idx_c, dest = w_idx.contiguous(), r_idx.contiguous(), dest.contiguous()
        if arena is not None:
            out, arena.token = _TritonGroup.apply(
                arena.token, arena, w_idx_c, r_idx_c, dest, v_group, beta_group,
                *float_terms, chunk_size,
            )
        else:
            out = _launch_fwd(
                table, w_idx_c, r_idx_c, dest, v_group.contiguous(), beta_group.contiguous(),
                tuple(t.contiguous() for t in float_terms), chunk_size, save=False,
            )[0]
        outputs.append(out)

    padded_len = write_idx.shape[1]
    out = torch.cat(outputs, dim=1).reshape(rows, padded_len, d_v)[:, :seq_len]
    final = arena.result(n_real) if arena is not None else table
    return out.reshape(*lead, seq_len, d_v), final.reshape(*lead, n_slots, d_v)
