"""Sparse Delta Memory reference kernels — fp32, torch-only, correct before fast.

Per head, a table `M ∈ ℝ^(N × d_v)` — `N` slots, one value vector each — read
as `y_t = M_tᵀ q_t` and updated by::

    M_t  =  (I − β_t k_t k_tᵀ) Λ_t M_{t−1}  +  β_t k_t v_tᵀ

`k_t` is nonzero only on the `W` slots position `t` writes and `q_t` only on the
`R` slots it reads.  `Λ_t` is **diagonal**, and it is the whole difference from
Gated DeltaNet: it decays the slots `t` writes and leaves every other slot
exactly as it was.  Forgetting is triggered by writes, not by time passing — a
slot nobody writes to is frozen, decay included.

The decay comes first and the delta rule reads the decayed memory::

    δ_t = β_t (v_t − k_tᵀ Λ_t M_{t−1}),      M_t = Λ_t M_{t−1} + k_t δ_tᵀ

When `Λ_t` is uniform over the write set the two factors commute and the order
is immaterial.  When it is not — a decay weighted by the write weights — it is
not, and this is the order.  The kernels take a log-decay **per write entry**,
so they cannot tell the two arrangements apart and do not need to.

Two paths live here and they compute the same recurrence, as in
:mod:`lumen.gdn.reference`:

* the **sequential** path, one position at a time — the decode step and the
  *oracle*.  The recurrence written in the most obvious way.
* the **chunkwise** path, which is what a layer runs, and which is required to
  reproduce the oracle to round-off in fp64.

Why the decay stays inside the sum
----------------------------------
Within a chunk, entering state `M₀`, write `G_{t,n} = Σ_{r≤t} log λ_{r,n}` for
the cumulative log-decay of slot `n`, inclusive of `t`.  Unrolling::

    δ_t    = β_t ( v_t − r_t − Σ_{s<t} A[t,s] δ_s )
    y_t    = u_t + Σ_{s≤t} QK[t,s] δ_s
    M_C[n] = exp(G_{C,n}) M₀[n] + Σ_s exp(G_{C,n} − G_{s,n}) k_{s,n} δ_s

    A[t,s]  = Σ_n k_{t,n} k_{s,n} exp(G_{t,n} − G_{s,n})      s <  t
    QK[t,s] = Σ_n q_{t,n} k_{s,n} exp(G_{t,n} − G_{s,n})      s <= t
    r_t     = Σ_n k_{t,n} exp(G_{t,n}) M₀[n]
    u_t     = Σ_n q_{t,n} exp(G_{t,n}) M₀[n]

So `(I + diag(β) A) δ = diag(β)(V − r)` — Gated DeltaNet's UT/WY solve, with
the decay **inside the sum over slots** instead of factoring out as a scalar.
That is why GDN's homomorphism trick (fold one `D[t,s] = γ_t/γ_s` into the
solve) has no analogue here: there is no single `D`.  Every ratio is formed per
slot and per pair, directly, as `exp(G_t − G_s)` with a non-positive exponent,
so every factor lies in `(0, 1]` by construction.

The per-channel factorisation — `(k ⊙ e^{G})(k ⊙ e^{−G})ᵀ`, one matmul —
materialises `e^{−G}`, the `1/γ` that ``docs/design/GATED_DELTANET.md`` §3.8
removed, and it arrives faster here: each write adds `|log α|` to one slot at
once, and a slot wiped a few times inside one chunk is past fp32's `exp` range.
Refused, for the same reason.

The partner layout
------------------
Every quantity above is nonzero only on slots the chunk touches.  `A[t,s]`
needs the slots two writes share; `QK[t,s]` a read at `t` and a write at `s`.
Because a position writes each slot **at most once**, an entry `(t, w)` has at
most one partner at any other position `s` — so the pairwise sums fit a dense
`(C, W, C)` array with one cell per (entry, partner position), filled by
gathers and reduced by plain sums.  Nothing in the forward is a scatter-add
into a repeated destination, so the forward is deterministic on every device.
(A gather's backward is a scatter-add; that is inherent to sparse addressing,
and deterministic under ``torch.use_deterministic_algorithms``.)

Finding the partner is a dense slot compare — `O(C²·W·(W+R))` booleans per
chunk, transient.  What autograd keeps is `O(C²·(W+R))` per chunk, which is
linear in `C` over a sequence: the chunk size is a memory dial here as well as
a speed one.  Most cells are empty.  A merge that enumerates only real pairs is
a faster path, and it earns its place by measurement.

Shape convention
----------------
Any leading axes ``…`` (batch, head)::

    memory     (…, N, d_v)
    write_idx  (…, T, W)   int64, distinct within each position
    write_val  (…, T, W)   `k_t` on its support
    log_decay  (…, T, W)   `log λ` per write entry, `<= 0`
    v          (…, T, d_v)
    beta       (…, T)
    read_idx   (…, T, R)   int64
    read_val   (…, T, R)   `q_t` on its support

`memory` may have fewer leading axes than the rest and is broadcast against
them — a learned initial table `(H, N, d_v)` serves a batch of streams.
"""

from __future__ import annotations

import math

import torch

from lumen.gdn.reference import inv_unit


# ── preconditions ─────────────────────────────────────────────────────────


def _check_distinct_writes(write_idx: torch.Tensor) -> None:
    """Raise if any position names the same slot twice in its write set.

    Product-key addressing guarantees distinct indices — the `W` selected slots
    are distinct `(i, j)` pairs — and both paths depend on it: the oracle's
    replacing scatter would keep one of two writes, and the chunkwise path's
    partner layout assumes one partner per position.  Either would return a
    plausible tensor of the right shape, so this is checked rather than assumed.
    """
    ordered = write_idx.sort(dim=-1).values
    if bool((ordered[..., 1:] == ordered[..., :-1]).any()):
        raise ValueError(
            "a position writes the same slot twice; the kernels require each "
            "position to name each slot at most once, which product-key "
            "addressing guarantees -- check how the write indices were built"
        )


def _check_shapes(
    memory: torch.Tensor,
    write_idx: torch.Tensor,
    write_val: torch.Tensor,
    log_decay: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    read_idx: torch.Tensor,
    read_val: torch.Tensor,
) -> None:
    if write_val.shape != write_idx.shape or log_decay.shape != write_idx.shape:
        raise ValueError(
            f"write_idx, write_val and log_decay must share a shape, got "
            f"{tuple(write_idx.shape)}, {tuple(write_val.shape)}, "
            f"{tuple(log_decay.shape)}"
        )
    if read_val.shape != read_idx.shape:
        raise ValueError(
            f"read_idx and read_val must share a shape, got "
            f"{tuple(read_idx.shape)} and {tuple(read_val.shape)}"
        )
    positions = write_idx.shape[:-1]
    if read_idx.shape[:-1] != positions or beta.shape != positions:
        raise ValueError(
            f"reads, writes and beta must cover the same positions, got "
            f"{tuple(write_idx.shape[:-1])}, {tuple(read_idx.shape[:-1])} and "
            f"{tuple(beta.shape)}"
        )
    if v.shape[:-1] != positions or v.shape[-1] != memory.shape[-1]:
        raise ValueError(
            f"v must be (…, T, d_v) with d_v = {memory.shape[-1]}, got "
            f"{tuple(v.shape)}"
        )
    if write_idx.shape[-1] > memory.shape[-2]:
        raise ValueError(
            f"{write_idx.shape[-1]} writes per position cannot be distinct in "
            f"a table of {memory.shape[-2]} slots"
        )


# ── one position ──────────────────────────────────────────────────────────


def _rows(table: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """`table (…, N, d_v)`, `index (…, K)` → the indexed rows, `(…, K, d_v)`."""
    return torch.gather(
        table, -2, index.unsqueeze(-1).expand(*index.shape, table.shape[-1])
    )


def read_sparse_delta(
    memory: torch.Tensor, read_idx: torch.Tensor, read_val: torch.Tensor
) -> torch.Tensor:
    """One position, read-only: `y = Mᵀ q`.  The table is not touched.

    `memory (…, N, d_v)`, `read_idx / read_val (…, R)` → `(…, d_v)`.  This is
    the readout of :func:`recurrent_sparse_delta`, which calls it — so the
    decode step's read and a read on its own are one expression.
    """
    return (read_val.unsqueeze(-1) * _rows(memory, read_idx)).sum(-2)


def recurrent_sparse_delta(
    memory: torch.Tensor,
    write_idx: torch.Tensor,
    write_val: torch.Tensor,
    log_decay: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    read_idx: torch.Tensor,
    read_val: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One position of the recurrence — the decode step, and the oracle's body.

    The module's shapes with the time axis removed: `memory (…, N, d_v)`,
    `write_* (…, W)`, `v (…, d_v)`, `beta (…)`, `read_* (…, R)`.  Returns the
    output `(…, d_v)` and the successor table.

    Only the `W` written rows are touched; every other row of the returned
    table is the incoming row, bit for bit.  Write indices must be distinct —
    this path does not check, because it runs once per decoded token and the
    layer's product keys guarantee it; :func:`sequential_sparse_delta` checks
    once for a whole sequence.

    Written to be read, not to be fast.  Everything in
    :func:`chunk_sparse_delta` is answerable to this function.
    """
    rows = _rows(memory, write_idx)
    # Decay first: the delta rule reads the memory as it stands AFTER this
    # position's forgetting, so a wiped slot contributes nothing to the value
    # being corrected.
    decayed = rows * log_decay.exp().unsqueeze(-1)
    retrieved = (write_val.unsqueeze(-1) * decayed).sum(-2)
    delta = beta.unsqueeze(-1) * (v - retrieved)
    written = decayed + write_val.unsqueeze(-1) * delta.unsqueeze(-2)
    # A replacing scatter, not an additive one: the written rows ARE the new
    # rows, so nothing is formed as `old + (new - old)` and rounded twice.
    index = write_idx.unsqueeze(-1).expand_as(written)
    memory = memory.scatter(-2, index, written)
    return read_sparse_delta(memory, read_idx, read_val), memory


def sequential_sparse_delta(
    memory: torch.Tensor,
    write_idx: torch.Tensor,
    write_val: torch.Tensor,
    log_decay: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    read_idx: torch.Tensor,
    read_val: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The whole sequence, one position at a time — the correctness oracle.

    Same signature as :func:`chunk_sparse_delta` minus ``chunk_size``.  `O(T)`
    Python iterations and not meant for training; it exists so the chunkwise
    path has something to be checked against.

    Returns:
        `(…, T, d_v)` outputs and the `(…, N, d_v)` final table.
    """
    _check_shapes(memory, write_idx, write_val, log_decay, v, beta, read_idx, read_val)
    _check_distinct_writes(write_idx)

    outputs = []
    for t in range(write_idx.shape[-2]):
        out, memory = recurrent_sparse_delta(
            memory,
            write_idx[..., t, :],
            write_val[..., t, :],
            log_decay[..., t, :],
            v[..., t, :],
            beta[..., t],
            read_idx[..., t, :],
            read_val[..., t, :],
        )
        outputs.append(out)
    return torch.stack(outputs, dim=-2), memory


# ── one chunk ─────────────────────────────────────────────────────────────


def _partner(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """`values (P, C·W)` at `index (P, C, K, C)` → `(P, C, K, C)`.

    Gathers from the flat per-row array rather than from a broadcast view of
    it: the backward of a gather allocates its input's shape, and a broadcast
    view's shape is the `C²·W²` one this layout exists to avoid.
    """
    return torch.gather(values, 1, index.reshape(index.shape[0], -1)).view(index.shape)


def _ratio(pair: torch.Tensor, exponent: torch.Tensor) -> torch.Tensor:
    """`exp(exponent)` on the cells in ``pair``, zero elsewhere.

    Every selected exponent is a difference `G_t − G_s` of the SAME slot's
    cumulative decay with `s <= t`, so it is `<= 0` by construction and the
    result lies in `(0, 1]`.  Unselected cells hold whatever the gathers
    produced; they are routed to `0` *before* the `exp`, because masking after
    an `exp` that has already produced `inf` does not help — and the gradient
    of a `where` whose unselected branch is `inf` is `nan`, not zero.
    """
    return torch.where(pair, exponent, exponent.new_zeros(())).exp() * pair


def _chunk(
    table: torch.Tensor,
    scratch: torch.Tensor,
    w_idx: torch.Tensor,
    w_val: torch.Tensor,
    w_logd: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    r_idx: torch.Tensor,
    r_val: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One chunk: flat table and `(P, C, …)` inputs → output, successor table.

    Indices are into the FLAT table, each row's slots offset by `p·N`, so two
    entries of different rows can never compare equal.  ``scratch`` is
    `(P, C, W)` indices of rows past the real table, one per write entry, where
    the entries that do not place a slot's new row send theirs.
    """
    rows, chunk, n_writes = w_idx.shape
    n_reads = r_idx.shape[-1]
    d_v = table.shape[-1]

    position = torch.arange(chunk, device=w_idx.device)
    # (t, s) masks, laid out to broadcast against (P, C_t, K, C_s).
    at_or_before = (position[None, :] <= position[:, None]).view(1, chunk, 1, chunk)
    before = (position[None, :] < position[:, None]).view(1, chunk, 1, chunk)

    # ── partners: which write at position s names this entry's slot ───────
    # The compare is C²·W·(W+R) and transient; only the (P, C, K, C) results
    # below survive it.  At most one match per (entry, s), by the distinct-
    # writes precondition, so `argmax` over the last axis finds it.
    w_match = w_idx[:, :, :, None, None] == w_idx[:, None, None, :, :]
    r_match = r_idx[:, :, :, None, None] == w_idx[:, None, None, :, :]
    w_has = w_match.any(-1)
    r_has = r_match.any(-1)
    column = position.view(1, 1, 1, chunk) * n_writes
    w_partner = column + w_match.to(torch.uint8).argmax(-1)
    r_partner = column + r_match.to(torch.uint8).argmax(-1)
    del w_match, r_match

    w_val_flat = w_val.reshape(rows, chunk * n_writes)
    w_logd_flat = w_logd.reshape(rows, chunk * n_writes)
    zero = w_logd.new_zeros(())

    # ── cumulative log-decay, per slot, at every entry ────────────────────
    # G at an entry is the sum of log-decays of every write to its slot at or
    # before its position.  Summed over the partner axis with the same values
    # in the same order wherever two entries share a history, so a read with
    # no write between it and the partner's gets an exponent of exactly 0.
    w_partner_logd = _partner(w_logd_flat, w_partner)
    r_partner_logd = _partner(w_logd_flat, r_partner)
    g_w = torch.where(w_has & at_or_before, w_partner_logd, zero).sum(-1)
    g_r = torch.where(r_has & at_or_before, r_partner_logd, zero).sum(-1)
    # And over the whole chunk: how far each written slot decays by the exit.
    g_end = torch.where(w_has, w_partner_logd, zero).sum(-1)

    # The partner's own G, fetched from the entry that wrote it.
    g_w_flat = g_w.reshape(rows, chunk * n_writes)
    w_partner_g = _partner(g_w_flat, w_partner)
    r_partner_g = _partner(g_w_flat, r_partner)
    w_partner_val = _partner(w_val_flat, w_partner)
    r_partner_val = _partner(w_val_flat, r_partner)

    # ── the pairwise matrices, decay inside the sum ───────────────────────
    pair_w = w_has & before
    pair_r = r_has & at_or_before
    a = (
        w_val.unsqueeze(-1)
        * w_partner_val
        * _ratio(pair_w, g_w.unsqueeze(-1) - w_partner_g)
    ).sum(2)
    qk = (
        r_val.unsqueeze(-1)
        * r_partner_val
        * _ratio(pair_r, g_r.unsqueeze(-1) - r_partner_g)
    ).sum(2)

    # `a` is strictly lower-triangular (the `before` mask), which is what
    # `inv_unit` expects: `(I + diag(β) A)⁻¹` by one batched triangular solve.
    transform = inv_unit(beta.unsqueeze(-1) * a)

    # ── the state-dependent part: gather, solve, read ─────────────────────
    rows_w = table.index_select(0, w_idx.reshape(-1)).view(rows, chunk, n_writes, d_v)
    rows_r = table.index_select(0, r_idx.reshape(-1)).view(rows, chunk, n_reads, d_v)
    retrieved = torch.einsum("pcw,pcwd->pcd", w_val * g_w.exp(), rows_w)
    read_back = torch.einsum("pcr,pcrd->pcd", r_val * g_r.exp(), rows_r)
    delta = transform @ (beta.unsqueeze(-1) * (v - retrieved))
    out = read_back + qk @ delta

    # ── the state update: one writer per slot ─────────────────────────────
    # Every write entry of a slot would compute the same new row; the first
    # one places it.  A replacing scatter with duplicate destinations would
    # hand the full gradient to EVERY duplicate source -- wrong, and the kind
    # of wrong that trains -- so the others are sent to scratch rows instead:
    # one per entry, past the real table, never gathered from, sliced off at
    # the end.  Their rows get no gradient because nothing reads them.
    #
    # Scratch rather than a boolean selection `w_idx[first]`, because every
    # shape here then depends on the configuration and never on the data: no
    # host sync, and nothing for `torch.func.vmap` to refuse.  And every
    # destination is distinct, so the write is deterministic without asking.
    first = ~(w_has & before).any(-1)
    carry = w_partner_val * _ratio(w_has, g_end.unsqueeze(-1) - w_partner_g)
    new_rows = g_end.exp().unsqueeze(-1) * rows_w + torch.einsum(
        "pcws,psd->pcwd", carry, delta
    )
    destination = torch.where(first, w_idx, scratch)
    table = table.index_copy(0, destination.reshape(-1), new_rows.reshape(-1, d_v))
    return out, table


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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunkwise-parallel sparse delta rule.  Shapes: the module docstring.

    Args:
        chunk_size: `C`, any positive integer.  Numerically inert — nothing
            here bounds accumulated decay, so nothing depends on it — and a
            memory dial: autograd keeps `O(T·C·(W+R))` per sequence.
        check_writes: verify the distinct-writes precondition.  On by default,
            because a violation returns a plausible wrong answer.  A caller
            whose indices are distinct **by construction** may turn it off:
            the check reads a value back to the host, which costs a sync and
            is a data-dependent branch that ``torch.func.vmap`` refuses.  With
            it off, every shape in this function depends on the configuration
            alone.

    Returns:
        `(…, T, d_v)` outputs and the `(…, N, d_v)` final table.  Rows no
        position writes are the incoming rows, bit for bit.

    Raises:
        ValueError: on inconsistent shapes, or a position that writes one slot
            twice (see :func:`_check_distinct_writes`).
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    _check_shapes(memory, write_idx, write_val, log_decay, v, beta, read_idx, read_val)
    if check_writes:
        _check_distinct_writes(write_idx)

    *lead, seq_len, n_writes = write_idx.shape
    n_reads = read_idx.shape[-1]
    n_slots, d_v = memory.shape[-2:]
    rows = math.prod(lead)
    device = write_idx.device

    # One flat table for the whole batch, plus one scratch row per write entry
    # of a chunk (see `_chunk`).  Broadcasting a shared initial table to the
    # batch materialises one copy per stream here -- the moment each is about
    # to be written anyway.
    n_real = rows * n_slots
    n_scratch = rows * chunk_size * n_writes
    table = torch.cat(
        [
            memory.expand(*lead, n_slots, d_v).reshape(n_real, d_v),
            memory.new_zeros(n_scratch, d_v),
        ]
    )
    scratch = (n_real + torch.arange(n_scratch, device=device)).view(
        rows, chunk_size, n_writes
    )
    offset = (torch.arange(rows, device=device) * n_slots).view(rows, 1, 1)
    write_idx = write_idx.reshape(rows, seq_len, n_writes) + offset
    read_idx = read_idx.reshape(rows, seq_len, n_reads) + offset
    write_val = write_val.reshape(rows, seq_len, n_writes)
    log_decay = log_decay.reshape(rows, seq_len, n_writes)
    read_val = read_val.reshape(rows, seq_len, n_reads)
    v = v.reshape(rows, seq_len, d_v)
    beta = beta.reshape(rows, seq_len)

    # A sequence that does not fill its last chunk is padded, not rejected, and
    # the padding is exact.  Padded positions write distinct slots (so the
    # precondition holds) with weight 0 and log-decay 0: no write, no decay.
    # A padded entry that is its slot's first writer in the chunk writes back
    # `exp(0) · row + 0`, which is the row.  Padded reads have weight 0, and the
    # outputs they produce are discarded.
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
    padded_len = write_idx.shape[1]
    n_chunks = padded_len // chunk_size

    # unbind, NOT x[:, n] inside the loop: indexing a tensor in the loop makes
    # autograd accumulate into a full-size zero buffer once per iteration.
    def chunks(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return x.reshape(rows, n_chunks, chunk_size, *x.shape[2:]).unbind(1)

    outputs = []
    for chunk in zip(
        *(
            chunks(x)
            for x in (write_idx, write_val, log_decay, v, beta, read_idx, read_val)
        )
    ):
        out, table = _chunk(table, scratch, *chunk)
        outputs.append(out)

    out = torch.stack(outputs, dim=1).reshape(rows, padded_len, d_v)[:, :seq_len]
    return out.reshape(*lead, seq_len, d_v), table[:n_real].reshape(*lead, n_slots, d_v)
