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

The chunkwise path holds its table between chunks in one of two ways, and they
agree bit for bit in the forward: an **arena** written in place, whose gradient
is carried by hand so that no chunk's backward costs anything in `N`, and a
**functional** table, a new one per chunk under plain autograd, which every
``torch.func`` transform accepts and which is differentiable twice.  The arena
is the default wherever it can run.  See :class:`_Arena`.

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

Finding the partner is one sort of the chunk's write keys and a binary search
per `(entry, position)` (:func:`_partners`), answerable exactly to a dense slot
compare kept as its specification (:func:`_partners_dense`).  What autograd
keeps is `O(C²·(W+R))` per chunk, which is linear in `C` over a sequence: the
chunk size is a memory dial here as well as a speed one.

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
from torch.autograd.function import once_differentiable

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


# ── partners ──────────────────────────────────────────────────────────────
#
# For every entry `(t, k)` -- a write or a read -- and every position `s` of
# its chunk: does `s` write this entry's slot, and if so, which of its write
# entries does?  At most one does, by the distinct-writes precondition.  The
# answer is two `(P, C, K, C)` arrays: `has`, and `partner`, the flat index
# `s·W + k'` of that write into the chunk's `(P, C·W)` write entries.  Where
# there is no partner, `partner` is `s·W` -- any valid index would do, since
# every use is masked by `has`, and this one is what the specification leaves.


def _partners_dense(
    w_idx: torch.Tensor, r_idx: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The specification: every entry against every write in the chunk.

    `O(C²·W·(W+R))` booleans per chunk, then a reduction over `W` -- written to
    be read.  :func:`_partners` is answerable to it, exactly.
    """
    rows, chunk, n_writes = w_idx.shape
    column = torch.arange(chunk, device=w_idx.device).view(1, 1, 1, chunk) * n_writes
    found: list[torch.Tensor] = []
    for index in (w_idx, r_idx):
        match = index[:, :, :, None, None] == w_idx[:, None, None, :, :]
        found += [match.any(-1), column + match.to(torch.uint8).argmax(-1)]
    return found[0], found[1], found[2], found[3]


def _partners(
    w_idx: torch.Tensor, r_idx: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """What the kernel runs: one sort, then one lookup per `(entry, s)`.

    A write's key is its slot and then its position, `slot·C + t` -- unique,
    because a position writes a slot at most once, so sorting the chunk's keys
    needs no tie-break and is the same on every device.  Whether position `s`
    writes the slot of an entry is then whether the key `slot·C + s` is among
    them: a binary search, `O(C·(W+R)·C·log(C·W))` per chunk against the
    specification's `O(C²·W·(W+R))`, landing directly in the layout the
    kernel uses.  No scatter, and every shape is the configuration's.
    """
    rows, chunk, n_writes = w_idx.shape
    position = torch.arange(chunk, device=w_idx.device)
    keys = (w_idx * chunk + position.view(1, chunk, 1)).reshape(rows, chunk * n_writes)
    keys, order = torch.sort(keys, dim=-1)
    last = chunk * n_writes - 1
    column = (position * n_writes).view(1, 1, 1, chunk)
    found: list[torch.Tensor] = []
    for index in (w_idx, r_idx):
        shape = (*index.shape, chunk)
        query = (index.unsqueeze(-1) * chunk + position).reshape(rows, -1)
        # `searchsorted` returns where the query would go, which is past the
        # end when it exceeds every key; clamped, the equality below says no.
        at = torch.searchsorted(keys, query).clamp(max=last)
        has = (keys.gather(1, at) == query).view(shape)
        partner = torch.where(has, order.gather(1, at).view(shape), column)
        found += [has, partner]
    return found[0], found[1], found[2], found[3]


def _chunk(
    rows_w: torch.Tensor,
    rows_r: torch.Tensor,
    scratch: torch.Tensor,
    w_idx: torch.Tensor,
    w_val: torch.Tensor,
    w_logd: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    r_idx: torch.Tensor,
    r_val: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One chunk: gathered rows and `(P, C, …)` inputs → output and the write.

    ``rows_w`` / ``rows_r`` are the table's rows at ``w_idx`` / ``r_idx`` as
    the chunk enters, `(P, C, W|R, d_v)`.  Returns the output `(P, C, d_v)` and
    the write the chunk makes: every entry's ``destination`` and the
    ``new_rows`` to place there.  The table itself is the caller's -- see
    :class:`_FunctionalTable` and :class:`_Arena` -- so this function is the
    same arithmetic whichever way it is held.

    Indices are into the FLAT table, each row's slots offset by `p·N`, so two
    entries of different rows can never compare equal.  ``scratch`` is
    `(P, C, W)` indices of rows past the real table, one per write entry, where
    the entries that do not place a slot's new row send theirs.
    """
    rows, chunk, n_writes = w_idx.shape

    position = torch.arange(chunk, device=w_idx.device)
    # (t, s) masks, laid out to broadcast against (P, C_t, K, C_s).
    at_or_before = (position[None, :] <= position[:, None]).view(1, chunk, 1, chunk)
    before = (position[None, :] < position[:, None]).view(1, chunk, 1, chunk)

    # ── partners: which write at position s names this entry's slot ───────
    w_has, w_partner, r_has, r_partner = _partners(w_idx, r_idx)

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

    # ── the state-dependent part: solve, read ─────────────────────────────
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
    return out, destination, new_rows


# ── the table across chunks: two ways to hold it ──────────────────────────
#
# `_chunk` reads rows and names a write; it never touches the table.  What
# holds the table between chunks decides what the table costs, and the two
# holders below compute the same values -- bit for bit in the forward.
#
# Measured on one machine (a Pascal card, fp32; the design record has the
# table), the functional holder spends 36% of a training step on table-sized
# work at `N = 128²` and 85% at `N = 256²` -- almost all of it autograd's
# bookkeeping in the backward, about five full-table passes per chunk.  That
# is a cost in `N` the recurrence does not have.  The arena removes it.


def _gather_rows(table: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """`table (rows, d_v)`, `index (…)` → `(…, d_v)`."""
    return table.index_select(0, index.reshape(-1)).view(*index.shape, table.shape[-1])


class _FunctionalTable:
    """The table as a value: each write returns a new table.  Plain autograd.

    Runs under every ``torch.func`` transform and is differentiable to any
    order, which is why it stays.  It pays for that in the table: each chunk's
    write copies it, and the backward of a gather and of a replacing write each
    allocate table-sized gradients, which then have to be summed.
    """

    def __init__(self, table: torch.Tensor) -> None:
        self.table = table

    def gather(
        self, w_idx: torch.Tensor, r_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _gather_rows(self.table, w_idx), _gather_rows(self.table, r_idx)

    def write(self, destination: torch.Tensor, rows: torch.Tensor) -> None:
        d_v = self.table.shape[-1]
        # `scatter` rather than `index_copy`: the same replacing write, but the
        # backward of `index_copy` retains its whole source -- to read its
        # shape -- where `scatter`'s retains only this index, a broadcast view.
        # Measured, that source was a fifth of everything a training step kept.
        index = destination.reshape(-1, 1).expand(-1, d_v)
        self.table = self.table.scatter(0, index, rows.reshape(-1, d_v))

    def result(self, n_real: int) -> torch.Tensor:
        return self.table[:n_real]


class _Arena:
    """The table as one buffer, written in place; its gradient carried sparsely.

    Four autograd Functions are the only things that touch the buffer: start,
    gather, write, finish.  Everything between them -- the pairwise terms, the
    solve, the new rows -- is ordinary autograd.  Nothing about the table is
    differentiated by hand except the two facts that define it: a gather reads
    rows, and a write replaces them.

    The table's gradient never exists per chunk.  It is ONE buffer, allocated
    once per backward, handed down a chain of zero-storage *token* tensors as
    their gradient: each Function takes the previous token and returns the
    next, so autograd's own dependency order runs their backwards in exactly
    the reverse of the forward.  Per chunk, the backward then touches only the
    rows the chunk touched:

    * the write's backward reads the gradient at its destinations -- that is
      the gradient of the new rows -- and zeroes those rows, because a replaced
      row's old value reached nothing past the write;
    * the gather's backward adds the gathered rows' gradients back in.

    Every token is consumed exactly once, so autograd never sums two of them;
    the buffer passes through by reference and is the Functions' own to
    mutate.  A fresh backward -- ``retain_graph``, or gradcheck's repeated
    passes -- starts a fresh buffer.  Double backward is refused rather than
    wrong (``once_differentiable``); the functional holder offers it.
    """

    def __init__(self, memory: torch.Tensor, n_scratch: int) -> None:
        self.table: torch.Tensor | None = None  # filled by `_Start`
        self.token: torch.Tensor | None = _Start.apply(memory, self, n_scratch)

    def gather(
        self, w_idx: torch.Tensor, r_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows_w, rows_r, self.token = _Gather.apply(self.token, self, w_idx, r_idx)
        return rows_w, rows_r

    def write(self, destination: torch.Tensor, rows: torch.Tensor) -> None:
        self.token = _Write.apply(self.token, rows, self, destination)

    def result(self, n_real: int) -> torch.Tensor:
        final = _Finish.apply(self.token, self, n_real)
        # Drop the last token, so no graph node is reachable from the arena and
        # nothing here outlives the forward that built it.  The Functions keep
        # only indices and shapes -- never the arena, never the table.
        self.token = None
        return final


def _token(table: torch.Tensor) -> torch.Tensor:
    """A table-shaped tensor with no storage: it exists to carry a gradient."""
    return table.new_zeros(()).expand_as(table)


class _Start(torch.autograd.Function):
    """`memory (…, N, d_v)` → the arena's buffer, real rows then scratch rows."""

    @staticmethod
    def forward(ctx, memory, arena, n_scratch):  # type: ignore[override]
        d_v = memory.shape[-1]
        n_real = memory.numel() // d_v
        table = memory.new_empty(n_real + n_scratch, d_v)
        # A copy, so the caller's table -- a state, perhaps a broadcast view of
        # a learned one -- is never written.  One per forward, not per chunk.
        table[:n_real].view(memory.shape).copy_(memory)
        table[n_real:].zero_()
        arena.table = table
        ctx.shape = memory.shape
        ctx.set_materialize_grads(False)
        return _token(table)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):  # type: ignore[override]
        if grad is None:
            return None, None, None
        n_real = math.prod(ctx.shape[:-1])
        return grad[:n_real].view(ctx.shape), None, None


class _Gather(torch.autograd.Function):
    """Rows at ``w_idx`` and ``r_idx``; backward adds their gradients in."""

    @staticmethod
    def forward(ctx, token, arena, w_idx, r_idx):  # type: ignore[override]
        table = arena.table
        ctx.save_for_backward(w_idx, r_idx)
        ctx.table_shape = table.shape
        ctx.set_materialize_grads(False)
        return _gather_rows(table, w_idx), _gather_rows(table, r_idx), _token(table)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_w, grad_r, grad):  # type: ignore[override]
        w_idx, r_idx = ctx.saved_tensors
        for index, rows in ((w_idx, grad_w), (r_idx, grad_r)):
            if rows is None:
                continue
            if grad is None:
                grad = rows.new_zeros(ctx.table_shape)
            grad.index_add_(0, index.reshape(-1), rows.reshape(-1, rows.shape[-1]))
        return grad, None, None, None


class _Write(torch.autograd.Function):
    """Replace the rows at ``destination`` -- distinct by construction -- in place."""

    @staticmethod
    def forward(ctx, token, rows, arena, destination):  # type: ignore[override]
        table = arena.table
        table.index_copy_(0, destination.reshape(-1), rows.reshape(-1, table.shape[-1]))
        ctx.save_for_backward(destination)
        ctx.rows_shape = rows.shape
        ctx.set_materialize_grads(False)
        return _token(table)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):  # type: ignore[override]
        if grad is None:
            return None, None, None, None
        (destination,) = ctx.saved_tensors
        index = destination.reshape(-1)
        grad_rows = grad.index_select(0, index).view(ctx.rows_shape)
        grad.index_fill_(0, index, 0.0)
        return grad, grad_rows, None, None


class _Finish(torch.autograd.Function):
    """The real rows of the final buffer; backward seeds the table gradient."""

    @staticmethod
    def forward(ctx, token, arena, n_real):  # type: ignore[override]
        table = arena.table
        ctx.table_shape = table.shape
        ctx.n_real = n_real
        ctx.set_materialize_grads(False)
        return table[:n_real]

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_final):  # type: ignore[override]
        if grad_final is None:
            return None, None, None
        # A buffer of our own: the incoming gradient belongs to whoever
        # produced it, and everything downstream of here mutates this one.
        grad = grad_final.new_zeros(ctx.table_shape)
        grad[: ctx.n_real] = grad_final
        return grad, None, None


def _transforms_active() -> bool:
    """Is a ``torch.func`` transform -- vmap, grad, jvp, … -- running?

    The arena's Functions carry the table through a side channel that no
    transform can batch or trace, so under one the functional holder is the
    only holder.  This is the check ``torch.autograd.Function.apply`` itself
    makes to decide how to dispatch.  It is private API; the vmap tests in
    ``tests/test_pytree.py`` are what would notice it moving.
    """
    return torch._C._are_functorch_transforms_active()


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
    in_place: bool | None = None,
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
        in_place: how the table is held between chunks.  ``True`` writes one
            buffer in place and carries its gradient sparsely (:class:`_Arena`),
            so no step costs anything in `N` beyond one copy in and one
            gradient buffer out.  ``False`` is the functional holder
            (:class:`_FunctionalTable`): a new table per chunk under plain
            autograd, which every ``torch.func`` transform accepts and which is
            differentiable twice.  ``None``, the default, is ``True`` unless a
            transform is running.  The two agree bit for bit in the forward;
            their gradients differ only in the order a row's contributions are
            summed.

    Returns:
        `(…, T, d_v)` outputs and the `(…, N, d_v)` final table.  Rows no
        position writes are the incoming rows, bit for bit.

    Raises:
        ValueError: on inconsistent shapes, a position that writes one slot
            twice (see :func:`_check_distinct_writes`), or ``in_place=True``
            under a ``torch.func`` transform.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    transformed = _transforms_active()
    if in_place is None:
        in_place = not transformed
    elif in_place and transformed:
        raise ValueError(
            "in_place=True cannot run under a torch.func transform: the table's "
            "gradient travels outside what the transform can see. Use "
            "in_place=False, or None to choose automatically."
        )
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
    entering = memory.expand(*lead, n_slots, d_v)
    table: _Arena | _FunctionalTable
    if in_place:
        table = _Arena(entering, n_scratch)
    else:
        table = _FunctionalTable(
            torch.cat([entering.reshape(n_real, d_v), memory.new_zeros(n_scratch, d_v)])
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
        w_idx, *_, r_idx, _ = chunk
        rows_w, rows_r = table.gather(w_idx, r_idx)
        out, destination, new_rows = _chunk(rows_w, rows_r, scratch, *chunk)
        table.write(destination, new_rows)
        outputs.append(out)

    out = torch.stack(outputs, dim=1).reshape(rows, padded_len, d_v)[:, :seq_len]
    final = table.result(n_real)
    return out.reshape(*lead, seq_len, d_v), final.reshape(*lead, n_slots, d_v)
