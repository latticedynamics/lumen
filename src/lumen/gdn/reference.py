"""Gated DeltaNet reference kernels — fp32, torch-only, correct before fast.

Two paths live here and they compute the same recurrence:

* the **chunkwise** path, which is what the layer runs.  It splits the sequence
  into blocks of `C`, hoists everything that does not depend on the state out
  into one batched pass, and leaves a sequential loop whose body is a single
  matmul per chunk.
* the **sequential** path, one position at a time, which is both the decode
  step and the *oracle*.  It is the recurrence written in the most obvious
  possible way, and the chunkwise path is required to reproduce it to 1e-9.

Keeping the obvious implementation around, permanently, is the cheapest defence
against a clever one drifting.  It is not dead code; it is the specification.

Why the chunkwise form exists at all
------------------------------------
Two facts, in this order.  Writing `M` for the state, read as `oᵀ = qᵀM`::

    M_t = α_t (I − β_t k_t k_tᵀ) M_{t−1} + β_t k_t v_tᵀ

**1. The decay comes out.**  With `γ_t = ∏_{s≤t} α_s` inside a chunk, the
substitution `M_t = γ_t M̃_t` turns the gated recurrence into an *ungated*
delta rule on `M̃` carrying values `v_t / γ_t`.  Exact, not an approximation.

**2. What is left is affine in `M`.**  The in-chunk pseudo-values are
`u_n = A diag(β)(ṽ_n − K_n M) = P_n − W_n M`, so the chunk update becomes::

    M_n = γ_n (E_n M + F_n),    E_n = I − K_nᵀ W_n,    F_n = K_nᵀ P_n

`P`, `W`, `E`, `F` and the intra-chunk attention are all state-free, so they are
computed for every chunk at once and the loop does one matmul.

**3. The decay is carried relatively, and this is the numerical story.**
The two facts above are the derivation; they are not how the code computes it.
Everything that reaches an answer is the ratio `γ_t / γ_i` for `i ≤ t`, which
lies in `(0, 1]` because `g = log γ` is non-increasing.  Forming it as
`γ_t · (1/γ_i)` computes a bounded number as one shrinking factor times one
growing one, and materialises the growing one.  `1/γ` reaches fp32's ceiling at
`−Σ log α ≈ 88.7`; long before that, an accumulation mixing terms whose `1/γ`
differ by more than `2²⁴` loses the small ones into the ulp of the large ones —
and in a delta rule the small term is the *recent* position, so the precision is
lost exactly where it is needed.

So `rel[t, i] = exp(g_t − g_i)` is formed directly and neither half is ever
built.  Every factor is in `(0, 1]`, there is no range to exceed, and the only
reachable failure is underflow to zero — which is the *correct* answer, meaning
that position has fully decayed.  The form this replaced failed by reaching
`inf`, meeting a zero, and returning `nan` for the whole batch.

Written with hats for the relatively-scaled quantities, the chunk update is::

    M_n = Ê_n M + F̂_n,    Ê_n = γ_C I − Kᵀ diag(e) Ŵ_n,    F̂_n = Kᵀ diag(e) P̂_n

with `e_i = exp(g_C − g_i)` the chunk-exit factor.  The boundary decay that used
to multiply the whole update is inside both terms now, because they no longer
share it.

This is also why the layer has no `max_chunk_decay`: that parameter existed to
bound `1/γ`, and bounding it capped the shortest half-life a head could express
at `chunk_size · ln2 / max_chunk_decay`, which made a GPU tiling parameter into
a modelling constraint.  Nothing here has an opinion about `chunk_size` now.

`E` and `W` depend only on the **key group** — the value axis never enters them
— which is what makes a layout sharing one key across many states cheap.

Shape convention used throughout this package
---------------------------------------------
`G_k` key groups, `m` states per key group (the rectangle rule in
:mod:`lumen.gdn.layout`), `H = G_k · m` states total::

    q          [B, G_k, m, T, d_k]      one query per state
    k          [B, G_k, 1, T, d_k]      one key per key group
    v          [B, G_k, m, T, d_v]      assembled by `assign_values`
    beta       [B, G_k, 1, T]           write strength, per key group
    log_alpha  [B, G_k, 1, T]           decay, per key group, <= 0
    M          [B, G_k, m, d_k, d_v]

`beta` is per key group **by necessity** — it enters the UT/WY inverse that the
key group shares.  `log_alpha` is per key group **by choice**: the inverse never
sees it, so it could carry an `m` axis at essentially no cost.  See the design
record; that seam is documented rather than taken.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from lumen.gdn.layout import HeadLayout


# ── the unit-lower-triangular inverse ─────────────────────────────────────


def inv_unit_lower_small(matrix: torch.Tensor) -> torch.Tensor:
    """`(I + N)⁻¹` for strictly-lower-triangular `N`, by forward substitution.

        X[i] = e_i − Σ_{m<i} N[i,m] X[m]

    **Do not be tempted by the Neumann/binary-expansion identity**
    ``Σ_j (−N)^j = ∏_i (I + (−N)^(2^i))``.  It is algebraically exact — `N` is
    nilpotent — but it forms intermediate powers that overflow in fp32 once the
    keys are correlated, which is exactly what a trained model produces.  The
    true result stays order 1 while the expansion reaches order 1e8.

    That trap is why this function is kept even though it is off the hot path.
    """
    size = matrix.shape[-1]
    lead = matrix.shape[:-2]
    eye = torch.eye(size, device=matrix.device, dtype=matrix.dtype).expand(*lead, size, size)

    rows = [eye[..., 0, :]]
    for i in range(1, size):
        previous = torch.stack(rows, dim=-2)
        rows.append(eye[..., i, :] - (matrix[..., i, :i, None] * previous).sum(-2))
    return torch.stack(rows, dim=-2)


def inv_unit_lower(matrix: torch.Tensor, block: int = 16) -> torch.Tensor:
    """Blocked forward substitution — same object, shorter Python loop.

    Inverts all diagonal blocks in one batched pass, then substitutes over the
    strict lower blocks.  At `C = 64, block = 16` that is 15 substitution steps
    and 10 batched matmuls instead of 63 sequential steps — a shorter autograd
    graph for identical numbers.

    Off the hot path (:func:`inv_unit` is faster still), and kept because the
    three implementations agreeing is a standing test.
    """
    size = matrix.shape[-1]
    width = min(block, size)
    if width == size:
        return inv_unit_lower_small(matrix)
    if size % width:
        raise ValueError(f"chunk size {size} must be divisible by block {width}")

    n_blocks = size // width
    lead = matrix.shape[:-2]

    blocked = matrix.reshape(*lead, n_blocks, width, n_blocks, width).movedim(-3, -2)
    diagonal = torch.diagonal(blocked, dim1=-4, dim2=-3).movedim(-1, -3)
    inverted = inv_unit_lower_small(diagonal)

    out: list[list[torch.Tensor | None]] = [[None] * n_blocks for _ in range(n_blocks)]
    zero = torch.zeros_like(inverted[..., 0, :, :])
    for i in range(n_blocks):
        out[i][i] = inverted[..., i, :, :]
        for j in range(i):
            acc = sum(blocked[..., i, s, :, :] @ out[s][j] for s in range(j, i))
            out[i][j] = -inverted[..., i, :, :] @ acc

    rows = [
        torch.cat([out[i][j] if j <= i else zero for j in range(n_blocks)], dim=-1)
        for i in range(n_blocks)
    ]
    return torch.cat(rows, dim=-2)


def inv_unit(matrix: torch.Tensor) -> torch.Tensor:
    """`(I + N)⁻¹` for strictly-lower-triangular `N`, via a triangular solve.

    The same object the two functions above compute, by the same forward
    substitution, batched into one library call instead of a Python loop.
    ``unitriangular=True`` assumes the unit diagonal, so `N`'s own diagonal and
    upper triangle are ignored and it goes in as-is.

    This is the hot path.
    """
    eye = torch.eye(
        matrix.shape[-1], device=matrix.device, dtype=matrix.dtype
    ).expand_as(matrix)
    return torch.linalg.solve_triangular(matrix, eye, upper=False, unitriangular=True)


# ── head layout → tensors ─────────────────────────────────────────────────


def assign_values(values: torch.Tensor, layout: HeadLayout) -> torch.Tensor:
    """`(B, T, G_v, d_v)` → `(B, G_k, m, T, d_v)`, one value per state.

    Free — a view or a stride-0 broadcast — for every layout with
    :attr:`~lumen.gdn.layout.HeadLayout.reindex_is_free`, which is all of the
    named ones.  A ragged assignment gathers and copies.

    The broadcast survives the kernel's chunk reshape, so the crossed layout
    really does carry one value tensor rather than `G_k` of them; splitting the
    time axis is view-compatible with a zero stride on the key axis.
    """
    batch, seq_len, n_value_groups, d_v = values.shape
    if n_value_groups != layout.n_value_groups:
        raise ValueError(
            f"layout has {layout.n_value_groups} value groups, got a tensor "
            f"with {n_value_groups}"
        )

    values = values.permute(0, 2, 1, 3)  # (B, G_v, T, d_v)
    rows = layout.rows

    if layout.rows_are_uniform:
        row = rows[0]
        selected = (
            values if row == tuple(range(n_value_groups)) else values[:, list(row)]
        )
        return selected.unsqueeze(1).expand(
            batch, layout.n_key_groups, len(row), seq_len, d_v
        )

    flat = [group for row in rows for group in row]
    if flat != list(range(len(flat))):
        values = values[:, flat]
    return values.reshape(
        batch, layout.n_key_groups, layout.states_per_key_group, seq_len, d_v
    )


# ── the recurrence ────────────────────────────────────────────────────────


def chunk_gated_delta(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    log_alpha: torch.Tensor,
    chunk_size: int,
    state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunkwise-parallel gated delta rule.

    Args:
        q: `(B, G_k, m, T, d_k)`, unit-norm.
        k: `(B, G_k, 1, T, d_k)`, unit-norm.
        v: `(B, G_k, m, T, d_v)` — see :func:`assign_values`.
        beta: `(B, G_k, 1, T)` write strength.
        log_alpha: `(B, G_k, 1, T)` log decay, `<= 0`.
        chunk_size: `C`, a power of two dividing `T`.
        state: `(B, G_k, m, d_k, d_v)` incoming state, or ``None`` for zeros.

    Returns:
        `(B, G_k, m, T, d_v)` outputs and the `(B, G_k, m, d_k, d_v)` final state.
    """
    batch, n_key_groups, per_group, seq_len, d_k = q.shape
    d_v = v.shape[-1]

    # A sequence that does not fill its last chunk is padded rather than
    # rejected.  Callers do not get to choose their sequence lengths -- a
    # corpus is whatever length it is -- and a library that raises here is one
    # every caller has to write a wrapper around.
    #
    # The padding is exact, not approximate.  Padded positions carry beta = 0
    # (no write) and log_alpha = 0 (no decay), so the recurrence there is
    # M_t = M_{t-1}: the final state is the state after the last real position.
    # And they sit at the END, so the causal mask keeps them out of every real
    # position's output.  Nothing is discarded except the padded outputs.
    remainder = seq_len % chunk_size
    if remainder:
        pad = chunk_size - remainder
        q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
        beta, log_alpha = (F.pad(t, (0, pad)) for t in (beta, log_alpha))

    n_chunks = (seq_len + (chunk_size - remainder if remainder else 0)) // chunk_size
    padded_len = n_chunks * chunk_size

    def chunkify(x: torch.Tensor, last: int) -> torch.Tensor:
        return x.reshape(*x.shape[:3], n_chunks, chunk_size, last)

    q = chunkify(q, d_k)
    k = chunkify(k, d_k)
    v = chunkify(v, d_v)
    beta = beta.reshape(batch, n_key_groups, 1, n_chunks, chunk_size)
    # The `1` here is the documented seam: nothing below this line reads
    # log_alpha through the UT/WY inverse, so it could carry `per_group` and
    # give every state its own timescale.  Left as-is in this version.
    log_alpha = log_alpha.reshape(batch, n_key_groups, 1, n_chunks, chunk_size)

    # ── decay absorption, in relative form ────────────────────────────────
    # `g` is non-increasing, so for the only entries anything reads -- `i <= t`
    # -- the exponent below is <= 0 BY CONSTRUCTION and every value lies in
    # (0, 1].  Nothing here can exceed 1, so there is no range to blow and no
    # clamp to impose on the model to keep it in range.
    #
    # The failure mode inverts, and that is the point: the only thing that can
    # happen is underflow to zero, and underflow to zero is the right answer --
    # it means that position has fully decayed.  The `gamma`/`1/gamma` split
    # this replaces failed the other way, reaching `inf`, meeting a zero, and
    # poisoning the batch with `nan`.
    cumulative = log_alpha.cumsum(-1)
    gamma = cumulative.exp()
    # rel[t, i] = exp(g_t - g_i).  The clamp is for the strictly-upper triangle,
    # which no consumer reads but which `exp` would overflow *before* any mask
    # is applied -- masking after an exp that has already produced `inf` does
    # not help.  On every entry that is read it is a no-op.
    rel = (cumulative[..., :, None] - cumulative[..., None, :]).clamp(max=0.0).exp()
    # exp(g_C - g_i), the chunk-exit factor: the last row of the same matrix.
    exit_decay = rel[..., -1, :]

    # ── UT/WY transform, once per key group ───────────────────────────────
    beta_k = beta[..., None] * k
    strict = torch.ones(
        chunk_size, chunk_size, device=q.device, dtype=torch.bool
    ).tril(-1)
    # The decay rides INSIDE the inverse rather than around it, which costs
    # nothing and is the whole trick.  For lower-triangular `A`, `B` and
    # `D[i,j] = d_i/d_j`, `(A*D)(B*D) = (AB)*D` -- so `A -> A*D` is an algebra
    # homomorphism on lower-triangular matrices and therefore commutes with
    # inversion.  Folding `rel` into `N` before the solve yields exactly
    # `inv(I + N) * rel`, with no intermediate outside (0, 1] ever formed.
    transform = inv_unit((beta_k @ k.transpose(-1, -2)) * strict * rel)[:, :, 0]

    # ── everything state-free, batched over all chunks ────────────────────
    # Hatted quantities carry a factor of gamma_i relative to the unscaled ones:
    # `w_hat[i] = gamma_i w[i]` and `pseudo_hat[i] = gamma_i pseudo[i]`.  That
    # factor is what the readout and the state update would otherwise have had
    # to divide back out.
    w = transform @ (gamma[..., None] * beta_k)[:, :, 0]
    pseudo = transform[:, :, None] @ (beta[..., None] * v)
    k_t = k.transpose(-1, -2)[:, :, 0]
    # E = gamma_C I - K^T diag(exp(g_C - g)) W_hat.  The chunk-boundary decay is
    # inside the transition now instead of multiplying the loop body, because
    # the two terms it used to scale together no longer share a factor.
    transition = gamma[..., -1][:, :, 0][..., None, None] * torch.eye(
        d_k, device=q.device, dtype=q.dtype
    ) - k_t @ (exit_decay[:, :, 0][..., None] * w)
    causal = torch.ones(chunk_size, chunk_size, device=q.device, dtype=torch.bool).tril(0)
    intra = (q @ k_t[:, :, None]) * causal * rel

    # The `m` axis is folded into the value axis: `transition` is shared across
    # states in a key group, so this makes `E @ M` one batched matmul over
    # B*G_k with no broadcast.  Expanding it every iteration costs more than
    # the matmul does.
    carry = (
        (k_t[:, :, None] @ (exit_decay[..., None] * pseudo))
        .permute(0, 1, 3, 4, 2, 5)
        .reshape(batch, n_key_groups, n_chunks, d_k, per_group * d_v)
    )
    memory = (
        q.new_zeros(batch, n_key_groups, d_k, per_group * d_v)
        if state is None
        else state.permute(0, 1, 3, 2, 4).reshape(
            batch, n_key_groups, d_k, per_group * d_v
        )
    )

    # ── the only sequential part: one matmul per chunk ────────────────────
    # unbind, NOT transition[:, :, n].  Slicing inside the loop makes autograd
    # allocate and accumulate into a full-size zero buffer once *per iteration*;
    # on this layer's shapes that memory traffic dominated the step.  unbind's
    # backward is a single stack.
    transitions = transition.unbind(2)
    carries = carry.unbind(2)

    entering = []
    for n in range(n_chunks):
        entering.append(memory)
        # No boundary factor out front any more -- it is folded into both terms.
        memory = transitions[n] @ memory + carries[n]
    stacked = torch.stack(entering, dim=2)

    # ── readout, all chunks at once ───────────────────────────────────────
    unfolded = stacked.reshape(
        batch, n_key_groups, n_chunks, d_k, per_group, d_v
    ).permute(0, 1, 4, 2, 3, 5)
    u = pseudo - w[:, :, None] @ unfolded
    # carry term + intra-chunk term.  The causal mask is diagonal-INCLUSIVE:
    # position t reads its own write.
    #
    # `gamma` scales only the carry term now.  The intra-chunk term already
    # carries its decay as `rel[t, i]` inside `intra`, which is the whole
    # difference: the ratio `gamma_t / gamma_i` is formed as a single bounded
    # exponential rather than as a product of one shrinking and one growing
    # factor computed apart from each other.
    out = (q @ unfolded) * gamma[..., None] + intra @ u

    memory = (
        memory.reshape(batch, n_key_groups, d_k, per_group, d_v)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    out = out.reshape(batch, n_key_groups, per_group, padded_len, d_v)
    return out[..., :seq_len, :], memory


def recurrent_gated_delta(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    alpha: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One position of the recurrence — the decode step, and the oracle.

    Same shapes as :func:`chunk_gated_delta` with the time axis removed:
    `q (B, G_k, m, d_k)`, `k (B, G_k, 1, d_k)`, `v (B, G_k, m, d_v)`,
    `beta`/`alpha` `(B, G_k, 1)`, `state (B, G_k, m, d_k, d_v)`.

    Written to be read, not to be fast.  Every optimisation in
    :func:`chunk_gated_delta` is answerable to this function.
    """
    decay = alpha[..., None, None]
    write = beta[..., None, None]

    read = (k.unsqueeze(-2) @ state).squeeze(-2)
    # alpha scales the CARRY only; the fresh write lands undecayed.
    state = decay * (state - write * k.unsqueeze(-1) * read.unsqueeze(-2)) + (
        write * k.unsqueeze(-1) * v.unsqueeze(-2)
    )
    out = (q.unsqueeze(-2) @ state).squeeze(-2)
    return out, state


def sequential_gated_delta(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    log_alpha: torch.Tensor,
    state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The whole sequence, one position at a time — the correctness oracle.

    Same signature as :func:`chunk_gated_delta` minus ``chunk_size``.  `O(T)`
    Python iterations and not meant for training; it exists so the chunkwise
    path has something to be checked against.
    """
    batch, n_key_groups, per_group, seq_len, d_k = q.shape
    d_v = v.shape[-1]
    if state is None:
        state = q.new_zeros(batch, n_key_groups, per_group, d_k, d_v)

    alpha = log_alpha.exp()
    outputs = []
    for t in range(seq_len):
        out, state = recurrent_gated_delta(
            q[..., t, :], k[..., t, :], v[..., t, :], beta[..., t], alpha[..., t], state
        )
        outputs.append(out)
    return torch.stack(outputs, dim=-2), state
