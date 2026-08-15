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
    log_alpha  [B, G_k, 1|m, T]         decay, per key group OR per state, <= 0
    M          [B, G_k, m, d_k, d_v]

`beta` is per key group **by necessity** — it enters the UT/WY inverse that the
key group shares.  `log_alpha` may be either, and the axis is what selects it.

Per-state decay and the shared solve
------------------------------------
The seam §3.4 documents is open here.  `log_alpha` with a state axis of `m`
gives every state its own forgetting timescale; with an axis of `1` it is the
shared arrangement, and that path is byte-for-byte the one that shipped.

The obvious worry is that per-state decay costs `H` triangular solves instead of
`G_k` — the exact expense the rectangle rule exists to avoid.  It does not, and
the reason is worth stating because it is not obvious.

Everything reads `A ⊙ rel`, and by the homomorphism above `inv((I+N) ⊙ rel) =
inv(I+N) ⊙ rel`.  So there are two routes: **damp then solve** (fold each state's
`rel` in, solve `m` times) or **solve then damp** (solve `I + N` once — it holds
no `α` at all — and scale each state's copy after).  The second keeps the solve
per key group.  It is only safe if the *undamped* inverse stays bounded, since
nothing shrinks it on the way through.

It does, and `beta_max = 2` is exactly why.  Forward substitution on `I + N` is
the delta rule itself: with `Z_i = Σ_{m≤i} k_m ⊗ X[m]`,

    X[i] = e_i − β_i k_iᵀ Z_{i−1},   Z_i = (I − β_i k_i k_iᵀ) Z_{i−1} + k_i ⊗ e_i

and for a unit key `‖I − β k kᵀ‖₂ = max(1, |1 − β|)`, which is **exactly 1 for
every β in [0, 2]** — identity at 0, reflection at 2, expanding only past it.
So `‖Z‖` grows at most linearly in `C` and the inverse cannot blow up.  Measured
on near-identical unit keys in fp64, `max|inv(I+N)|` is 1.98 at `C = 16` and 1.98
at `C = 256` — flat — and at `β = 2.1` it is 5.8 rising to 1.7e8 over the same
range.  The boundary is sharp and the shipped ceiling sits on the safe side of it.

This is a *different* object from the `1/γ` that §3.8 rejects.  That one is
unbounded above and reaches fp32's ceiling at accumulated decay 88.7; this one is
bounded by 2 however long the chunk.  Hence: solve then damp.

**`beta_max > 2` breaks this**, and it breaks the shared path too — `rel → 1` as
`α → 1`, so slow forgetting leaves the solve undamped either way.  See
:func:`chunk_gated_delta`.
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
        beta: `(B, G_k, 1, T)` write strength.  The `1` is **forced**: `beta`
            enters the UT/WY inverse the key group shares.
        log_alpha: `(B, G_k, 1, T)` for one timescale per key group, or
            `(B, G_k, m, T)` to give every state its own.  Log decay, `<= 0`.
        chunk_size: `C`, a power of two dividing `T`.
        state: `(B, G_k, m, d_k, d_v)` incoming state, or ``None`` for zeros.

    Returns:
        `(B, G_k, m, T, d_v)` outputs and the `(B, G_k, m, d_k, d_v)` final state.

    Raises:
        ValueError: if the state axis of `beta` or `log_alpha` is neither `1`
            nor (for `log_alpha`) `m`.  Checked rather than broadcast: a wide
            `log_alpha` used to flow through several `[:, :, 0]` slices that
            silently applied **state 0's decay to every state** and returned a
            plausible tensor of the right shape.  A wrong answer that does not
            raise is the one failure this module can least afford.
    """
    batch, n_key_groups, per_group, seq_len, d_k = q.shape
    d_v = v.shape[-1]

    if beta.shape[2] != 1:
        raise ValueError(
            f"beta must be per key group, got state axis {beta.shape[2]}; it "
            f"enters the UT/WY inverse that the key group shares, so varying it "
            f"per state would need a different inverse per state"
        )
    n_decay = log_alpha.shape[2]
    if n_decay not in (1, per_group):
        raise ValueError(
            f"log_alpha's state axis must be 1 (one timescale per key group) or "
            f"{per_group} (one per state), got {n_decay}"
        )
    # Which of the two routes above.  The shared case keeps the shipped path
    # exactly -- decay folded in before the solve -- because every experimental
    # record downstream was produced by it.
    per_state_decay = n_decay != 1

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
    # The seam of §3.4, now taken.  This axis is 1 or `m`, and every decay-derived
    # quantity below inherits it -- so the shared case broadcasts against the
    # per-state ones exactly as it always did, and the per-state case does not.
    log_alpha = log_alpha.reshape(batch, n_key_groups, n_decay, n_chunks, chunk_size)

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
    # For lower-triangular `A`, `B` and `D[i,j] = d_i/d_j`, `(A*D)(B*D) = (AB)*D`
    # -- so `A -> A*D` is an algebra homomorphism on lower-triangular matrices
    # and therefore commutes with inversion.  `rel` is such a `D`.  Which side of
    # the solve it goes on is therefore free to choose, and the two cases choose
    # differently; see the module docstring for why both are safe.
    #
    # Either way the state axis is KEPT rather than sliced away with `[:, :, 0]`.
    # Carrying a size-1 axis through costs nothing -- the arithmetic is identical
    # and the shared path stays bit-for-bit what it was -- and it means the
    # per-state case is the same expressions with a wider axis, instead of a
    # second copy of the kernel.
    strict_n = (beta_k @ k.transpose(-1, -2)) * strict
    if per_state_decay:
        # Solve once, undamped, then damp per state: `inv(I+N) * rel`.  `N` holds
        # no alpha, so this is still ONE solve per key group and the rectangle
        # rule's O(C^3)-paid-G_k-times survives per-state decay intact.
        transform = inv_unit(strict_n) * rel
    else:
        # Damp first: `inv((I+N) * rel)`.  Identical object, and no intermediate
        # outside (0, 1] is ever formed.  This is the shipped path, unchanged.
        transform = inv_unit(strict_n * rel)

    # ── everything state-free, batched over all chunks ────────────────────
    # Hatted quantities carry a factor of gamma_i relative to the unscaled ones:
    # `w_hat[i] = gamma_i w[i]` and `pseudo_hat[i] = gamma_i pseudo[i]`.  That
    # factor is what the readout and the state update would otherwise have had
    # to divide back out.
    w = transform @ (gamma[..., None] * beta_k)
    pseudo = transform @ (beta[..., None] * v)
    k_t = k.transpose(-1, -2)
    # E = gamma_C I - K^T diag(exp(g_C - g)) W_hat.  The chunk-boundary decay is
    # inside the transition now instead of multiplying the loop body, because
    # the two terms it used to scale together no longer share a factor.
    transition = gamma[..., -1][..., None, None] * torch.eye(
        d_k, device=q.device, dtype=q.dtype
    ) - k_t @ (exit_decay[..., None] * w)
    causal = torch.ones(chunk_size, chunk_size, device=q.device, dtype=torch.bool).tril(0)
    intra = (q @ k_t) * causal * rel

    carry = k_t @ (exit_decay[..., None] * pseudo)

    # ── the only sequential part: one matmul per chunk ────────────────────
    # unbind, NOT transition[..., n, :, :].  Slicing inside the loop makes
    # autograd allocate and accumulate into a full-size zero buffer once *per
    # iteration*; on this layer's shapes that memory traffic dominated the step.
    # unbind's backward is a single stack.
    if per_state_decay:
        # `E` reads gamma, exit_decay and w -- all alpha -- so it is per state
        # now and cannot be shared across the states of a key group.  The fold
        # below is therefore unavailable: same FLOPs in the loop, one more
        # batched axis instead of a wider value axis.
        memory = (
            q.new_zeros(batch, n_key_groups, per_group, d_k, d_v)
            if state is None
            else state
        )
        transitions = transition.unbind(3)
        carries = carry.unbind(3)

        entering = []
        for n in range(n_chunks):
            entering.append(memory)
            memory = transitions[n] @ memory + carries[n]
        unfolded = torch.stack(entering, dim=3)
    else:
        # The `m` axis is folded into the value axis: `transition` is shared
        # across states in a key group, so this makes `E @ M` one batched matmul
        # over B*G_k with no broadcast.  Expanding it every iteration costs more
        # than the matmul does.
        folded = carry.permute(0, 1, 3, 4, 2, 5).reshape(
            batch, n_key_groups, n_chunks, d_k, per_group * d_v
        )
        memory = (
            q.new_zeros(batch, n_key_groups, d_k, per_group * d_v)
            if state is None
            else state.permute(0, 1, 3, 2, 4).reshape(
                batch, n_key_groups, d_k, per_group * d_v
            )
        )
        transitions = transition[:, :, 0].unbind(2)
        carries = folded.unbind(2)

        entering = []
        for n in range(n_chunks):
            entering.append(memory)
            # No boundary factor out front any more -- it is folded into both.
            memory = transitions[n] @ memory + carries[n]
        stacked = torch.stack(entering, dim=2)

        unfolded = stacked.reshape(
            batch, n_key_groups, n_chunks, d_k, per_group, d_v
        ).permute(0, 1, 4, 2, 3, 5)
        memory = (
            memory.reshape(batch, n_key_groups, d_k, per_group, d_v)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

    # ── readout, all chunks at once ───────────────────────────────────────
    u = pseudo - w @ unfolded
    # carry term + intra-chunk term.  The causal mask is diagonal-INCLUSIVE:
    # position t reads its own write.
    #
    # `gamma` scales only the carry term now.  The intra-chunk term already
    # carries its decay as `rel[t, i]` inside `intra`, which is the whole
    # difference: the ratio `gamma_t / gamma_i` is formed as a single bounded
    # exponential rather than as a product of one shrinking and one growing
    # factor computed apart from each other.
    out = (q @ unfolded) * gamma[..., None] + intra @ u

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
    `beta (B, G_k, 1)`, `alpha (B, G_k, 1|m)`, `state (B, G_k, m, d_k, d_v)`.

    Per-state decay needs no code here — `alpha[..., None, None]` broadcasts
    against a state of `(B, G_k, m, d_k, d_v)` whether the axis is `1` or `m`.
    That is not an accident worth relying on silently, so it is a stated part of
    the contract and there is a test for it: the specification was per-state
    ready before the fast path was, which is the order that lets the fast path
    be checked at all.

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
