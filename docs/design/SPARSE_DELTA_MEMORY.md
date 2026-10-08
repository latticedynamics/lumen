# Sparse Delta Memory — design record

**Status:** landed in 0.6.0, as `lumen.sdm`; in 0.6.1 the table is written in
place (§3.11) and partners are found by search (§3.1); in 0.6.2 only the
table-touching work stays in the chunk loop, and the pairwise terms can be
recomputed (§3.12); in 0.7.0 an opt-in Triton backend computes the same
function as kernels (§3.13); in 0.8.0 a decode step given `donate=True`
writes the table in place (§3.6). The implementation is expected to match
this record; where the two disagree, one of them is a bug.

Unlike Gated DeltaNet and Undertow, this is **not a consolidation.** No
existing implementation was merged, so there is no port to verify and no
experimental record downstream to protect — merge → verify → switch does not
apply, and "numerically transparent" has nothing to be transparent *to*. What
does apply is everything else: a formal definition, decisions with their
evidence, an oracle the fast path answers to, and defaults that say whether
they were earned.

Why it is here at all: SDM decouples **state size** from **parameter count** and
from **per-token compute**. Per token, reads and writes touch `W + R` rows of a
table of `N` rows, so compute is `O((W+R)·d_v)` whatever `N` is, and the only
parameters that grow with `N` are the two address projections, which grow as
`√N`. A small model can carry a very large state. That is the property wanted;
the rest of this record is about getting it without spending the things Lumen
exists to protect.

---

## 1. What the layer computes

Per head, a memory table `M ∈ ℝ^(N × d_v)` — `N` slots, one value vector each.
At position `t` a write selects `W` slots `I_t` with weights `k_t ∈ ℝ^N`
(nonzero only on `I_t`), a read selects `R` slots with weights `q_t` (nonzero
only on its own set), and

```
    M_t  =  (I − β_t k_t k_tᵀ) Λ_t M_{t−1}  +  β_t k_t v_tᵀ
    y_t  =  M_tᵀ q_t

    Λ_t  =  diag(λ_t),   λ_{t,n} = α_t^{ω_{t,n}},   ω_{t,n} = 0 for n ∉ I_t
```

Read after write, as in Gated DeltaNet. Two choices of `ω` on the write set:

- **write-set decay** (`ω = 1` on `I_t`) — the paper's. Every written slot
  decays by the full `α_t`, every other slot not at all.
- **key-weighted decay** (`ω = k_t`) — present in the paper's reference code,
  off by default there, not discussed in the paper. Decay scales with the write
  weight, so a slot selected at weight `0.001` barely decays.

**Forgetting is event-driven, not time-driven.** A slot nobody writes to is
frozen — decay included. Its half-life is counted in writes *to it*, not in
positions. That is the mechanism behind the paper's perplexity continuing to
fall out to 1M tokens, and it is what lets a learned initial memory (§3.5)
survive a context rather than being decayed out of it within a hundred tokens,
which is what happens to a learned initial state under scalar decay.

**The order is decay, then delta.** The delta rule reads the *decayed* memory:
`δ_t = β_t (v_t − kᵀ Λ_t M_{t−1})`, `M_t = Λ_t M_{t−1} + k_t δ_tᵀ`. Under
write-set decay `Λ_t k_t = α_t k_t` because `k_t` lives on the same support, so
`Λ_t` commutes with `(I − β k kᵀ)` and the order is immaterial. Under
key-weighted decay it does not commute, and the order above is the reference
code's. It is written this way throughout so that one derivation covers both.

**Addressing.** Product keys. Two score vectors `s¹, s² ∈ ℝ^√N` from linear
projections of `x`; the `N` slot scores are their outer sum, and
`top_W(s¹ ⊕ s²) = top_W(top_W(s¹) ⊕ top_W(s²))`, so the top `W` of `N` is found
from `W²` candidates without forming `N` scores. The weights are a softmax over
the `W` selected scores — so `k ≥ 0`, `Σ k = 1`, `‖k‖₂ ≤ 1`. Reads are the same
with their own projection and `R`. Selection is not differentiable; the softmax
weights are, so the address projections learn through the values of the scores
they select, with no straight-through estimator.

**Relation to Gated DeltaNet — held as a test, not quoted as a claim.** With
`N = d_k`, `W = R = N` and dense unit keys, `Λ_t = α_t I` and the recurrence is
Gated DeltaNet exactly. The paper claims this; §7 gate 2 checks it against
Lumen's own GDN oracle.

**Stated as a model class:** a gated delta rule whose key space has dimension
`N ~ 10⁵` with `W`-sparse simplex keys, and whose decay is **diagonal with
support tied to the write set.** It is the same family as the channel-wise-gated
delta rules (a per-key-coordinate decay), specialised so that the gate and the
write share an address.

---

## 2. Sources, and where they disagree

**The paper** (Cabannes et al., *Sparse Delta Memory: Scaling the State of
Linear RNNs through Sparsity*, 2026) and **its reference implementation**. The
implementation is licensed CC-BY-NC 4.0 and Lumen is MIT: it was read, and
nothing is copied — not the kernels, not its test oracle. Everything here is
derived from the recurrence in §1.

Where the sources disagree, and which side this record takes:

| point | paper | reference code | here |
|---|---|---|---|
| delta target (Eq. 4) | `v − M̃[i]`, the slot's own content | `v − Σ_j k_j M̃[j]`, the joint read | joint read — Eq. 4 is a typo; the paper's Fig. 2 and appendix agree with the code |
| `W`, `R` | 64 | 128 in every released config | untested; §3.9 |
| local window | 128 | 1024, commented as the paper's value | not this layer's concern |
| reading norm | RMSNorm | LayerNorm | house RMSNorm (§3.7) |
| output gate | "gated" | sigmoid | house SiLU (§3.7) |
| BatchNorm on address projections | not mentioned | credited in kernel notes with preventing slot collapse; **off in every config** | excluded (§3.8) |
| key-weighted decay | not mentioned | implemented, off | available, off (§3.4) |
| learned initial memory | default, ablated | default | available, zero-initialised (§3.5) |

---

## 3. Decisions

### 3.1 The decay is diagonal, and it is carried relatively

This is the derivation Gated DeltaNet's record deferred — *"a diagonal decay
does not factor the same way. New derivation and new kernel."* It is taken
here because SDM cannot exist without it.

Within a chunk of `C` positions, entering state `M₀` (the chunk's, not the
sequence's), write `G_{t,n} = Σ_{r≤t} log λ_{r,n}` — the cumulative log-decay
of slot `n`, inclusive of `t`. Every factor that reaches an answer is a ratio
`Γ_{t←s}[n] = exp(G_{t,n} − G_{s,n})` for `s ≤ t`, and since `log λ ≤ 0` the
exponent is `≤ 0`: **bounded in `(0, 1]` by construction**, exactly as §3.8 of
the GDN record requires. Unrolling the recurrence:

```
    δ_t  =  β_t ( v_t − r_t − Σ_{s<t} A[t,s] δ_s )
    y_t  =  u_t + Σ_{s≤t} QK[t,s] δ_s
    M_C[n] = exp(G_{C,n}) M₀[n] + Σ_s exp(G_{C,n} − G_{s,n}) k_{s,n} δ_s

    A[t,s]  = Σ_n k_{t,n} k_{s,n} exp(G_{t,n} − G_{s,n})      s < t
    QK[t,s] = Σ_n q_{t,n} k_{s,n} exp(G_{t,n} − G_{s,n})      s ≤ t
    r_t     = Σ_n k_{t,n} exp(G_{t,n}) M₀[n]                  retrieved, pre-chunk
    u_t     = Σ_n q_{t,n} exp(G_{t,n}) M₀[n]                  read, pre-chunk
```

So `(I + diag(β) A) δ = diag(β)(V − r)`: the same UT/WY solve as GDN, with the
decay **inside the sum over slots** instead of factoring out as a scalar. That
is the difference, and it is why GDN's homomorphism trick (`(I+N) ⊙ D`) has no
analogue here — there is no single `D` to fold. The ratio is formed per slot,
per pair, directly.

**The factored form is refused, and SDM reaches its failure faster than GDN
does.** The obvious per-channel implementation computes `k ⊙ e^{G}` and
`k ⊙ e^{−G}` and multiplies — materialising `e^{−G}`, the `1/γ` that GDN §3.8
removed, which overflows fp32 once a slot's accumulated log-decay inside the
chunk passes `−88.7`. Under scalar decay that takes a long chunk of steady
forgetting. Under write-set decay the decay arrives in lumps: each write to a
slot adds `|log α_t|` at once, and the paper's trained gate reaches near zero on
some writes — slots being wiped. A few hard wipes of one slot inside one chunk
is enough, and a slot that is hot is exactly one that gets written repeatedly.
The pairwise form has no range to exceed: every factor it builds is a ratio in
`(0, 1]`, and the only reachable failure is underflow to zero, which is the
right answer for a wiped slot.

**What a chunk touches.** Every quantity above is nonzero only on slots some
position in the chunk writes or reads. `A[t,s]` needs slots shared by *two
writes*; `QK[t,s]` slots shared by a read at `t` and a write at `s`. Because a
position writes each slot at most once (§3.2), **for a given entry `(t, w)` and
a given other position `s`, there is at most one partner** — so the pairwise
sums can be laid out as a dense `[C, W, C]` (and `[C, R, C]`) array with one
cell per (entry, partner position), filled by gathers and reduced by plain sums.
No scatter-add into a repeated destination appears in the forward, so the
forward is deterministic on every device. (The backward of any gather is a
scatter-add; that is inherent to sparse addressing, and it is deterministic
under `torch.use_deterministic_algorithms`.) Equivalently: within a chunk, SDM
is a diagonal-decay delta rule on the slots that chunk touches.

**Finding partners: a sort, not a compare.** The obvious way to find an
entry's partner at each position is to compare its slot against every write in
the chunk — a dense compare, `O(C²·W·(W+R))` booleans per chunk, reduced over
`W`. The first kernel did that, and it is kept as the specification
(`_partners_dense`). The kernel sorts the chunk's write keys `slot·C + t`
instead — unique, because a position writes a slot at most once, so the sort
needs no tie-break and orders them the same on every device — and answers each
`(entry, s)` with a binary search for `slot·C + s`: `O(C·(W+R)·C·log(C·W))`,
landing directly in the dense layout, with no scatter and no shape the data
chooses. It reproduces the compare **exactly**, down to the index it leaves
where there is no partner, so everything downstream is the same arithmetic on
the same numbers; that is a test, on layouts built to reach the extremes —
every position writing the same slots, no slot written twice, reads that never
meet a write — and on a device.

Measured on one machine (a Pascal card, `B = 4`, 2 heads, `W = R = 64`,
`N = 128²`): 6.9 ms per chunk for the compare against 1.0 ms for the search at
`C = 32`, 2.0 against 0.4 at `C = 16`, 25.6 against 2.5 at `C = 64`. A whole
training step at the defaults went from 398 ms to 221; the forward alone from
287 to 109. By the per-chunk timings, the search is about 15% of a step where
the compare was over half.

What autograd keeps is `O(C²·(W+R))` per chunk, so `O(T·C·(W+R))` per sequence
— linear in `C`, which makes the chunk size a memory dial as well as a speed
one. Most of those cells are empty. A layout holding only real pairs would
shrink them, at the price of a shape the data chooses (§3.10).

**The state update keeps one writer per slot.** A slot written by several
positions in a chunk gets its new row computed once, at its first write entry,
and placed with a non-accumulating index write. That keeps the forward free of
atomics and keeps autograd honest: a replacing scatter with duplicate
destinations silently hands the full gradient to *every* duplicate source,
which is wrong, and it is the kind of wrong that trains. **Measured, by
mutation:** placing every entry's row instead of the first's leaves the forward
equal to the oracle to `4e-16` — every duplicate computes the same row — and
moves the gradients by up to `28` against values of order one, on every input
but the read weights. Only the gradient gate catches it, which is why that gate
exists.

**Chunk size is inert.** Nothing here bounds accumulated decay, so no clamp is
needed and none exists; `chunk_size` is a tiling parameter with no opinion about
half-lives — GDN §3.8's property, inherited rather than re-earned. Held by a
test at decay strong enough that a factored form would fail.

### 3.2 A position writes each slot at most once

Product keys guarantee it — the `W` selected indices are distinct `(i, j)`
pairs. The kernels **depend** on it (the single-partner layout above, the
replacing scatter in the oracle) and **check** it, raising rather than
returning a plausible wrong answer. Reads carry no such requirement: a
repeated read index would simply be summed. The layer skips the check, and
§3.10 says why that is safe.

### 3.3 `β_max`, and why the solve survives

GDN's argument transfers because it never needed unit keys, only `‖k‖ ≤ 1`:
forward substitution on `I + diag(β) A` *is* the delta rule, whose step
`(I − β k kᵀ) Λ` has spectral norm `≤ max(1, |1 − β‖k‖²|) ≤ 1` for
`β ∈ [0, 2/‖k‖²] ⊇ [0, 2]`. Softmax and l2 keys both have `‖k‖ ≤ 1`, and the
diagonal decay only shrinks. So `β_max ≤ 2` is safe here too, and past 2 it is
refused for the same reason.

**The default is 2, the house value, and it is untested.** The paper uses
`β = sigmoid(·)`, i.e. `β_max = 1`. With softmax keys the effective erase along
`k` is `β‖k‖²`, which the paper's own access statistics put near `0.02–0.07` at
`β ≈ 0.5` — so the reflection regime `β‖k‖² > 1` is unreachable unless keys
concentrate to `‖k‖² > 0.5`, and the ceiling mostly decides how strong the
delta correction is *allowed* to get. 2 leaves the model room to answer §4.2
for itself; 1 would pre-empt it.

### 3.4 Decay parametrisation: the house gate, per head

`log α = −softplus(a_proj(x))`, one per head, `a_proj.bias` initialised to `−3`
— GDN's form, not the paper's `−exp(A_log)·softplus(a + dt_bias)`. Reasons: a
block holding either mixer sees one gate convention; and at `−3`,
`α₀ = exp(−softplus(−3)) ≈ 0.953`, which happens to sit where the paper's gate
converges (mean `≈ 0.95` in their training statistics). The per-head spread the
paper's `A ∈ [0, 16]` initialisation buys is moot at the one or two heads SDM
runs with.

The per-entry log-decay handed to the kernel is `ω ⊙ log α`, with `ω` chosen by
`decay_weighting="write_set" | "key"` (§1). **The kernel does not know which** —
it takes a log-decay per write entry — so the choice is one line in the layer
and cannot fork the kernel. `"write_set"` is the default because it is what the
paper trained; `"key"` removes a discontinuity (the slot ranked `W` decays by
the full `α`, the slot ranked `W+1` not at all) and is **untested**.

### 3.5 The initial memory: two options, different in kind

`initial_memory="zero" | "learned"`.

The paper's own ablation (its Table 3, at 1.4B parameters) separates what each
buys:

| | code NLL | avg accuracy | RULER |
|---|---|---|---|
| bigger state (SDM zero-`M₀` − GDN) | −0.004 | −0.6 | +8.0 |
| learned `M₀` (SDM learned − SDM zero) | −0.023 | +1.2 | +3.2 |

The state buys long-context recall. The learned `M₀` buys most of the modelling
loss and all of the short-context accuracy — and a learned table read through
product keys **is** a memory layer (Berges et al., 2024), so a learned-`M₀`
SDM is a memory layer and a fast-weight memory **sharing one table**, with the
write gate deciding when in-context content overwrites pretrained content. That
is a different kind of object from the zero-`M₀` layer, not a better setting of
it, and the layer offers both.

Decisions:

- **`initial_memory` is required, with no default** — GDN's `layout` argument,
  applied again: two different models behind one field means a default would be
  the library choosing a model on the caller's behalf. It costs one argument per
  construction.
- **Learned `M₀` is zero-initialised**, so at construction the two options are
  the same layer, exactly — the house pattern (centring, per-state decay):
  turning it on changes nothing until training moves it. The paper initialises
  it with a truncated normal; that comparison has not been run. A zero `M₀`
  still trains: `∂y/∂M₀[n] = q_n` is nonzero wherever a read lands.
- **`M₀` is state-shaped, not batch-shaped:** `(H, N, d_v)`, broadcast to the
  batch by `init_state`, so a batch of `B` streams does not cost `B` copies
  until each is first written.
- **How `M₀` is optimised is the caller's decision**, and this record says so
  plainly because the obvious default is questionable: the paper's code puts it
  in the ordinary AdamW group, weight decay 0.1. Every step then shrinks every
  slot — read or not — so pretrained content nobody reads decays by optimiser
  step even while it is frozen in context. The layer exposes the parameter
  structurally (`initial_memory_parameters()`, as `residual_out_projections()`
  exposes the output) so a caller can give it its own group without matching
  names.
- **Parameter accounting.** A learned `M₀` is `H · N · d_v` parameters, and they
  count like any others. The paper leaves them out of its parameter counts
  ("state, not parameters"); an iso-parameter claim that excludes a table the
  size of the model is an iso-*active*-parameter claim.

### 3.6 The state is a table, and a step returns a successor

`SparseDeltaMemoryState(memory)`, frozen, `(B, H, N, d_v)`. `step()` returns a
successor rather than mutating — the house guarantee that branching a stream
cannot leave two branches sharing a buffer.

For GDN that guarantee is free. **Here it costs a table copy per stream per
step:** `O(B·N·d_v)` per head per position, against `O(B·(W+R)·d_v)` of actual
work. It is easy to read that as a cost of large tables, and it is not only
that: the copy scales with the batch exactly as it scales with `N`. At `B = 1` a
step is launch-bound and flat in `N`, which hides it. Measured on one machine (a
Pascal card, fp32, one layer at `d_model = 512`, 2 heads, `W = R = 64`), against
the same arithmetic written into the table in place:

| `N` | `B = 1` | `B = 32` | `B = 128` |
|---|---|---|---|
| 32² | 1.22 → 1.20 ms | 1.34 → 1.16 ms | 4.10 → 1.89 ms |
| 64² | 1.20 → 1.18 ms | 3.06 → 1.14 ms | 11.07 → 2.22 ms |
| 128² | 1.21 → 1.18 ms | 9.70 → 1.15 ms | 37.62 → 2.27 ms |

In a five-block trunk with two such layers at 64² or 128², the copy was
nothing at `B = 8`, and 61% and 86% of the whole step at `B = 128`. At
`N = 512²` it is most of a step already at `B = 8`: 36 ms against 0.2.

**So the guarantee is the default, and the copy can be waived:**
`step(x, state, donate=True)`. Donating a state hands its buffers to the layer,
which may write the successor into the table it was given. The caller must not
read a donated state again; the successor is the stream. A caller who does not
opt in keeps the functional path unchanged. The two paths are one function,
`recurrent_sparse_delta`, with `scatter` or `scatter_` as its last write, so
they agree bit for bit, and a test holds it.

Why a keyword and not a separate `step_`. The path that needs it is a trunk
stepping many streams, so whatever is chosen has to pass through `Block.step`
and `Stack.step`. A method on this layer alone would not reach that path, and a
fourth method on every component would break the surface they share. Gated
DeltaNet and Undertow accept the keyword and ignore it, because their successors
cost no more than their arithmetic. `Block` forwards it only when it is set, so a
stateful sub-layer from outside the library that predates it keeps stepping.

Why not copy-on-write at the granularity of pages of rows. Product keys spread a
position's writes across the table on purpose. With 64-row pages, 64 writes touch
essentially every page of a 32² table and up to a quarter of a 128² one. So
paging saves little at the sizes where the copy bites, and every kernel would
then have to read a paged table.

**Donating is a permission, not a demand.** The layer copies anyway when the
table is not its own to write:

- any view. This includes the broadcast that `init_state` hands out, which
  under `"learned"` shares the parameter's storage. At `B = 1` that view is not
  even overlapping, so an in-place write would silently change the learned
  table;
- a table autograd is recording;
- a table under a `torch.func` transform.

The first donated step from `init_state` is therefore a copy, the same one the
successor rule makes anyway, and every step after it is in place.

`init_state` hands out the initial table as a **broadcast view**, `(B, H, N,
d_v)` over `(H, N, d_v)`: no stream costs a copy until its first write, which is
the copy the successor rule makes anyway. A training pass makes it once, into
a buffer it then writes in place (§3.11): nothing is copied per chunk, and
autograd retains the gathered rows and never a table — held by a test.

**Re-reads are out.** GDN's `reread(q', cache)` is cheap because the entering
state per chunk is `d_k × d_v`. Here it is the table — `O((T/C)·N·d_v)` to
retain, and a fresh query's read set is unknown in advance, so nothing smaller
suffices. Not offered; the readout is still linear in the query, so it is
reachable if someone wants to pay for it. A read of the *current* state for one
position, `read(x, state)`, is offered: it is a step with the write switched
off.

### 3.7 Output side: the house's

Per-head RMSNorm over `d_v`, a SiLU gate from `x`, then a projection — GDN's
and Undertow's, so a block can hold any of the three without knowing which. The
paper's variant (RMSNorm in its text, LayerNorm and a sigmoid gate in its code)
differs in details nobody has shown to matter; matching the house is the choice
that buys something.

Inputs follow the paper where the paper specifies the recurrence: `v = W_v x`
with no activation, address scores linear in `x`, no short convolution.

### 3.8 Excluded, and why

| excluded | why |
|---|---|
| BatchNorm on the address projections | in training its batch statistics include **future positions** — a small causality leak — and slot selection shifts between batch statistics (training) and running statistics (inference). Off in every released config of the reference code, absent from the paper. If slot collapse appears, it is a finding to record, and a load-balancing term or a causal normaliser are the candidates |
| short causal convolution | the paper has none, and names it as the only difference from GDN. Reachable later as a layer detail; not an untested default to ship |
| a bf16 table, fp8/int4 snapshots | a faster path, earned by measurement. The reference table is the model dtype, and the gradient into `M₀` is not accumulated through bf16 atomics. (Triton kernels were in this row; §3.13 is what earned them out of it) |
| context parallelism | not the layer's business |
| `reread` | §3.6 |

### 3.9 Untested defaults, listed as such

Nothing here is canonised by being chosen. Each of these has a default because
a layer needs one, and none has been compared by anyone:

- `W`, `R` — 64 here and in the paper's text, 128 in its released configs.
- `β_max` — 2 here (house), 1 in the paper (§3.3).
- `decay_weighting` — write-set (paper) vs key-weighted (reference code, off).
- `key_norm` — softmax (paper, default) vs `"l2"`, the selected scores
  normalised to a unit vector: signed weights and `‖k‖ = 1`, so `β‖k‖² = β` and
  the delta rule runs at GDN's strength. **Shipped as the direct experiment for
  §4.2.** Key-weighted decay is refused under it: `ω = k` would turn a negative
  weight into a positive log-decay — growth — and the natural analogue
  (`ω = k²`, the slot's share of the key's mass) is a new definition nobody has
  asked for yet.
- `M₀` initialisation — zero (here) vs truncated normal (paper); `M₀` optimiser
  treatment (§3.5).
- `chunk_size` — inert numerically; a speed-and-memory dial, and the default of
  32 is not a measured optimum. The optimum sits where a chunk's fixed cost —
  launches, the Python loop — meets the pairwise terms' growth in `C`, and it
  has moved with every change to the kernel. Measured on one machine:
  - at `d_model = 512` (`B = 4`, `T = 1024`), 16 was fastest before the
    partner search (§3.1), 32 after it (203–230 ms a step at every table size
    from 32² to 256²), and 16 again once the table-free work left the loop
    (§3.12): 172 ms, against 204 at 32 and 256 at 64;
  - at `d_model = 128` on 32K rows (§3.12), it depends on how many rows share a
    micro-batch: one row runs fastest at 32–64, four with the pairwise terms
    recomputed at 16.

  Memory grows with `C` unless the pairwise terms are recomputed. A faster
  device shrinks the pairwise cost without the fixed one, so none of this is
  settled for others.

### 3.10 Every shape depends on the configuration, never on the data

`lumen.pytree`'s tests run every mixer under `torch.func.vmap` over stacked
parameter sets — how a population-based search evaluates many weight draws in
one call. A first version of the chunkwise kernel was refused there twice over,
and both refusals came from choices, not necessities:

- **Placing each slot's new row by boolean selection** (`w_idx[first]`) — a
  data-dependent shape. Replaced by **scratch rows**: one per write entry, past
  the real table, where every entry that is not its slot's first writer sends
  its row. Never gathered from and sliced off at the end, so those rows get no
  gradient because nothing reads them. Every destination is distinct, so the
  write is deterministic without asking and autograd routes exactly one
  gradient per slot. The cost is `B·H·C·W` extra rows, typically a percent or
  two of the table.
- **The distinct-writes check** — a value read back to the host, so a sync and
  a data-dependent branch. Kept in the kernel and on by default, because a
  violation returns a plausible wrong answer; **the layer turns it off**,
  because `_address` produces distinct indices by construction (distinct
  `(row, column)` pairs from two `topk`s). An override of `_address` inherits
  that obligation, and its docstring says so.

Under a transform the kernel holds its table functionally — a new table per
chunk — because the in-place arena's gradient travels by a side channel no
transform can see (§3.11). The transformed forward pays the table copy per
chunk that an untransformed one no longer does.

Held by tests: a vmapped stack of SDM blocks equals the models run one at a
time **exactly**, and two vmapped chunks with the table carried equal one
vmapped pass.

### 3.11 The table is written in place, and its gradient is carried by hand

The recurrence costs `O((W+R)·d_v)` per position whatever `N` is. The first
chunkwise path did not. It held the table as a value — each chunk's write
returned a new one under plain autograd — and while the forward's copy per
chunk was expected, the backward's cost was not: autograd's gradient for a
gather is a table-sized buffer of zeros with rows added in, and for a replacing
write a table-sized copy with rows zeroed, so every chunk's backward allocated
three table-sized buffers and summed them. Measured on one machine (a Pascal
card, fp32, `d_model = 512`, `B = 4`, `T = 1024`, 2 heads, `W = R = 64`),
table-sized work was **36%** of a training step at `N = 128²` and **85%** at
`N = 256²`. Training cost grew with `N`, which is the one thing this layer
exists not to do.

The chunkwise kernel now holds the table in an **arena**: one buffer, copied
in once and written in place. Four `autograd.Function`s — start, gather,
write, finish — are the only code that touches it. Everything between them —
the pairwise terms, the solve, the new rows — is ordinary autograd, so no
derivative of the recurrence is written by hand. What is written by hand are
the two facts that define the table: a gather reads rows, and a write replaces
them.

The table's gradient is one buffer per backward, and it travels as the gradient
of a chain of zero-storage *token* tensors: each Function takes the previous
token and returns the next, so autograd's own dependency order runs their
backwards in exactly the reverse of the forward. A chunk's backward touches
only the rows the chunk touched. The write reads the gradient at its
destinations — that is its new rows' gradient — and zeroes those rows, since a
replaced row's old value reached nothing past the write. The gather adds the
gathered rows' gradients back in. Each token is consumed exactly once, so the
buffer passes by reference and is never summed, and a fresh backward starts a
fresh buffer.

Measured on the same machine, table-sized work fell to **0.7%** of a step at
`N = 128²` and **1.6%** at `N = 256²`. Forward and backward, ms:

| | `C = 8` | `C = 16` | `C = 32` | `C = 64` |
|---|---|---|---|---|
| `N = 32²` | 549 *(519)* | **296** *(321)* | 379 *(416)* | 612 *(644)* |
| `N = 128²` | 519 *(1126)* | **303** *(720)* | 398 *(622)* | 632 *(752)* |
| `N = 256²` | 616 *(3604)* | **311** *(1952)* | 407 *(1233)* | 642 *(1053)* |

*(in italics: the functional holder, before)*. At `C = 16` a 64× larger table
costs 5% more time. (These precede §3.1's partner search, which moved the
fastest chunk size to 32; there, the same 64× costs 13%.) One cell moved the wrong way, by 6%: the smallest chunk at the smallest
table, where there is no table cost to remove and 128 chunks each make four
more calls.

**The functional holder stays**, as `in_place=False`, for the two things an arena
cannot do. It runs under a `torch.func` transform, where the arena's side
channel is invisible, so under a transform it is chosen automatically (§3.10);
the check is the one `torch.autograd.Function.apply` itself makes, which is
private API, and the vmap tests are what would notice it moving. And it is
differentiable twice: the arena's Functions are `once_differentiable`, so a
second derivative through them is refused rather than wrong. The two holders
agree bit for bit in the forward; their gradients differ only in the order a
row's contributions are summed.

**The functional holder lost one avoidable cost on the way.** Its write was
`index_copy`, whose backward retains the whole source — every chunk's new rows
— to read its shape; `scatter` is the same replacing write and retains only its
index. At the defaults that source was a fifth of everything a training step
kept, 512 of 2727 MiB. The arena never kept it.

Held by tests: both holders pass every chunkwise value and gradient gate of §7
on their own; the forward agrees bit for bit, recording or not; gradients agree
to `1e-12` in fp64; the arena backpropagates twice (`retain_graph`) to
identical gradients; the functional holder passes `gradgradcheck`, and the
arena refuses a double backward; the caller's table is never written; neither
holder retains anything table-sized; each chunk retains its gathered write rows
exactly once — the gate that fails if `index_copy` returns; and a `Stack` with
`recompute` reaches the same gradients through the arena.

### 3.12 Only the table is sequential

Most of what a chunk computes depends on nothing another chunk wrote: its
partners (§3.1), the cumulative decays, the pairwise matrices `A` and `QK`, the
solve `(I + diag(β) A)⁻¹`, each write's decay to the chunk's end, and the carry
of each position's `δ` into each written slot. Only the rows it gathers from the
table do, and with them the retrieval, the read, `δ`, the output and the new
rows. Gated DeltaNet's reference is built on that split already: everything
state-free is computed for all chunks at once, and the loop over chunks is one
matmul each.

The first chunkwise path ran all of a chunk inside the loop. On a wide layer
the device was busy regardless and it did not show; on a narrow one it was the
whole cost. Measured on one machine (a Pascal card, fp32; `d_model = 128`,
2 heads, `N = 128²`, `W = R = 64`, rows of 32,768 positions), a chunk's forward
and backward cost 4.7–4.8 ms **whatever its size**: at every `C` from 8 to 64,
for one row or two, and at `W = R` from 16 to 64. The step was the number of
chunks times the fixed cost of issuing a chunk's operations from Python —
hundreds of operator calls each. A Gated DeltaNet block of the same width
takes 0.10 s for a whole row.

The kernel now computes the table-free terms for a **group** of consecutive
chunks at once, batched (`_chunk_terms`), then walks the group one chunk at a
time doing only what needs the table (`_chunk_apply`), about a dozen
operations. A group is as many chunks as keep one pairwise array,
`(chunks·P, C, max(W, R), C)`, within 2²³ cells: unbounded, a forward-only pass
over a long sequence would hold every chunk's pairwise terms at once, where the
loop held one. The number depends on the configuration alone (§3.10), and the
grouping is numerically inert. It is the same arithmetic, reorganised: measured
against the kernel before it, outputs and gradients agree bit for bit on the
host in fp32 and fp64, and on a device under deterministic algorithms (without
them, by less than the old kernel's own run-to-run spread). The budget trades
launches against a transient:
a forward-only pass holds one group's pairwise terms at once. Measured, 2²⁰
cells gives back the loop's forward-only peak at the configuration of §3.11
(336 MiB, against 795 at 2²³) and costs 5% of that step and 20–35% of the
narrow layer's below; each doubling to 2²³ narrows the gap. 2²³ is the default,
and the transient it buys does not grow with `T`.

**Recomputing the pairwise terms.** The table-free terms are also most of what
the backward keeps, `O(T·C·(W+R))` per sequence against the gathered rows'
`O(T·(W+R)·d_v)`; on the narrow layer above, about 8 of the 10.8 GiB one row
keeps at `C = 32`. With `recompute_pairwise` on — a kernel argument, and a
layer attribute off by default, like `Stack.recompute` — the backward keeps
only what the table-touching steps need, and rebuilds a group's terms, batched,
when it reaches them. Of the pairwise arrays one stays, the carry, which the
new rows' step consumes. Because the rebuild is issued once per group, it costs
little: a first prototype that checkpointed each chunk instead doubled the
per-chunk cost.

Same machine and configuration, forward and backward per micro-batch:

| | one row, before | one row | one row, recomputed | four rows, recomputed |
|---|---|---|---|---|
| `C = 8` | 19.7 s · 4.7 GiB | 7.5 s · 4.7 GiB | — | — |
| `C = 16` | 9.8 s · 6.7 GiB | 3.5 s · 6.8 GiB | 3.5 s · 3.1 GiB | **4.4 s** · 11.4 GiB |
| `C = 32` | 4.9 s · 10.8 GiB | **1.8 s** · 10.8 GiB | 1.9 s · 3.5 GiB | 5.8 s · 12.6 GiB |
| `C = 64` | 2.4 s · 18.9 GiB | 1.7 s · 19.0 GiB | 2.3 s · 4.1 GiB | 8.5 s · 14.7 GiB |
| `C = 128` | out of memory | out of memory | 3.9 s · 5.1 GiB | 15.1 s · 18.9 GiB |

(peak memory for the layer alone.) Without the pairwise terms to keep, four rows
fit where at most two did, and the table's size still does not matter: from
`N = 64²` to `512²`, four rows at `C = 64` take 8.5–8.6 s. After the change the
device is busy about three quarters of the time at one row, `C = 32`; at four
rows it is busy throughout, on elementwise work over the pairwise arrays — which
is where §8's next lever is aimed.

Held by tests: the grouping is inert (one chunk per group, uneven groups, all at
once) under both holders; recomputing gives the same outputs and gradients bit
for bit, on the host and on a device under deterministic algorithms, keeps one
pairwise array per group where the plain path keeps many, and nests under
`Stack.recompute`; under a `torch.func` transform it is refused while gradients
are enabled, and ignored without them, when there is nothing to rebuild.

### 3.13 A second route: the Triton backend

`backend="triton"` computes the same function as `chunk_sparse_delta` with
Lumen's own kernels (`lumen.sdm.triton_kernels`). It is opt-in and never
detected, for Undertow's reason (`docs/design/UNDERTOW.md` §3.4), and it answers
to the same oracle as the reference: outputs and every gradient to fp32
round-off. Where it cannot run -- a CPU tensor, fp64, a `torch.func`
transform with gradients, a chunk longer than 64 -- the reference runs
instead, exactly. Under `vmap` without gradients it does run: the forward is a
custom operator whose batching rule moves the mapped axis in among the
sequences, which the kernels take any number of (§3.10's consumer, many
parameter sets in one call, gets the forward's gain).

The reference's costs on a long sequence were not the recurrence's (§3.12,
§8): the terms made a dozen passes over dense `(C, K, C)` arrays that are
almost entirely zero, and the walk issued a dozen small operations per chunk.
Both halves are kernels here, one group of chunks per autograd node. What was
decided, and why:

**Partners by segment.** The partner relation of §3.1 is a sort, read the
other way. Sorting a chunk's writes by `slot·C + position` makes the writes
that share a slot one contiguous segment in position order, and each entry
walks its own segment -- the partners it actually has, one or two at the
access statistics product keys produce and `C` at worst -- instead of `C`
cells of which nearly all are empty. The cumulative decays are summed member by
member in position order, so two entries with the same history add the same
terms in the same order and an exponent that should be exactly zero is (§3.1).
`A` and `QK` are built a row at a time. A first version searched by brute force
in registers -- every entry against every write -- and was correct and slow:
the search is `C·W` steps per chunk, and its registers allowed one program per
multiprocessor. The segments were 30 times faster at the narrow layer of
§3.12.

**The carry is one number per write.** Every write entry of one slot carries
the same row, nonzero only at that slot's own writes; so the `(C, W, C)` carry
is a single coefficient per write, `c_m = k_m · exp(G_end − G_m)` (an exponent
`≤ 0`), and the new row of a slot is

    M_C[n] = exp(G_end,n) · M₀[n] + Σ_{m writes n} c_m · δ_{position(m)}

summed along the slot's segment. Its transpose in the backward is a gather: the
gradient reaching `δ_s` from the new rows is `Σ_w c[s,w] · g[slot(s,w)]`. The
array the reference keeps for this, the largest a training step kept, is not
formed, and neither is the matmul that applied it.

**One owner per sum, no atomics.** A scatter-add -- the reference backward's
`index_add` -- reorders its sums from run to run. Here every reduction has an
owner: the terms sum each pair's two ends from the two entries' own segments
(sharing a slot is symmetric), and the walk's backward hands each slot a chunk
touched to its first writer, or for a slot only read to its first reader,
which sums everything that reached the old row in segment order and writes
once. So the backward is reproducible without asking. It is also portable: a
design with atomics depends on how a compiler lowers them, and on one machine
(Triton 3.5, a Pascal card) every form of `tl.atomic_add` failed to assemble --
it carries a memory-order qualifier that architecture's assembler refuses. An
inline-assembly reduction worked, and brought its own hazard: a tile smaller
than the program is replicated across threads, Triton's own atomics mask the
replicas, and an inline instruction cannot know it is one. Two replicas add
twice. The tests caught it; the owners made it moot.

**The walk is persistent, and its columns never mix.** A program owns one
sequence's table and a block of its value columns and loops over a group's
chunks in order, with a barrier between a chunk's reads of the table and its
writes. Every operation in the recurrence is per value column, so programs that
share a sequence never communicate. The backward walk carries only the table's
gradient; what a chunk's reads hand back depends only on the output's
gradient, so it is summed for every chunk at once before the walk begins, and
the reductions over the value width run after it, in parallel.

**Launch geometry: one wave.** The walk is a chain of dependent loads per
program, so latency bounds it, and two things set the chain's length. A narrow
column block gathers a whole chunk's rows in one round -- 8 columns is one
32-byte sector, the unit the memory system moves anyway -- and shortens the
chain. But each program holds a chunk's tiles and, on the machine measured,
fills its multiprocessor's registers, so a grid larger than the card runs in
waves, each paying the whole chain. The rule is the narrowest block whose grid
fits one wave. It picked the fastest of the configurations tried at every shape
measured. Products of a chunk's `C × C` matrices use `tl.dot` where every side
is at least 16 and a broadcast below that; past `C = 64` the solve's tile no
longer fits the shared memory `tl.dot` stages it in on that machine, and those
chunks -- also the slowest measured -- run the reference.

**What stays in torch.** The sort that builds the segments (integers only) and
the solve `(I + diag(β) A)⁻¹`, one batched triangular solve whose backward is
written out: `dB = −Xᵀ dX Xᵀ` on the strict lower triangle.

Measured on one machine (a Pascal card, fp32, `N = 128²`, `W = R = 64`,
`C = 32`; the kernel alone, forward and backward):

| shape | reference | Triton |
|---|---|---|
| `d_model = 512`, 4 × 1,024 | 184 ms · 2.4 GiB | 52 ms · 1.8 GiB |
| `d_model = 256`, 2 × 8,192 | 596 ms · 6.3 GiB | 101 ms · 2.5 GiB |
| `d_model = 256`, 1 × 16,384 | 843 ms · 6.2 GiB | 112 ms · 2.4 GiB |
| `d_model = 128`, 1 × 32,768 | 1,708 ms · 10.4 GiB | 149 ms · 2.4 GiB |

The gain grows as rows narrow and sequences lengthen, where the reference was
bound by launching operations. Memory falls because the pairwise arrays are not
kept, or formed; four rows of 32,768 at `C = 16`, out of memory on the reference
in 22 GiB, train in 9.4. On this path `C` matters little: 16 was 2--8% faster
than 32, which kept less memory. The default is unchanged.

Two further measured choices. The walks' loops over entry blocks are not
unrolled: unrolling kept every block's tiles live, at the register limit and
spilling, where the rolled loop ran 9--22% faster. And the work that does not
wait on the walk -- the next group's table-free half in the forward, a group's
reductions after its own walk in the backward -- runs on a second stream beside
it, for 2--7%: concurrent kernels slow the latency-bound walk, which gives back
much of the overlap.

Held by tests (`tests/test_sdm_triton.py`): §7.17.

---

## 4. What the paper's evidence supports

### 4.1 State for recall, table for knowledge

§3.5's table. The long-context claim is the state's and survives the ablation;
it is the new content of the paper. The headline — better than GDN at every
rung of its scaling ladder, better than full attention at 8B — is confounded
with a learned `M₀` that is 56–150% of the non-embedding parameter count and
excluded from it. By the paper's own training statistics, only ~42% of slots
receive any write over a ~21M-token window; the rest act as read-only
parametric memory. **The control that would separate the two is GDN plus a
product-key memory layer of equal size**, and it is not in the paper. GDN with a
learned initial state is not that control: its state is 0.5 MB, and scalar
decay erases it within the context.

### 4.2 Is it a delta rule? — a hypothesis nobody has tested

With softmax keys, `‖k‖² = Σ p²`. The paper's access statistics (roughly 25–35
effective write keys of 64, top-1 mass roughly 0.2–0.3) put it near
`0.03–0.15`, so the delta correction along `k` is `β‖k‖² ≈ 0.02–0.07` per write
— against `β` itself in GDN with unit keys, an order of magnitude weaker. The
heavy erasing must then be `α` on the write set, and the paper's minimum forget
gate near zero is slots being wiped.

The hypothesis: **SDM works as sparse gated overwrite more than as a delta
rule**, and its erase address is its write address. There is no delta-vs-additive
ablation in the paper. It is cheap to test at small scale: `key_norm="softmax"`
against `"l2"`, and the delta rule against a plain additive write, on a recall
task. If it holds, *where* the layer erases and *where* it writes become
separable questions, and the second design choice worth testing is giving the
erase an address of its own.

---

## 5. Relation to existing work

**No novelty is claimed for the model class.** What is here is a derivation
the chunkwise form needs, a set of configuration decisions, and the evidence for
each.

- **Sparse Delta Memory** (Cabannes et al., 2026) — the model class; §1–§4.
- **Product-key memory** (Lample et al., 2019) and **memory layers at scale**
  (Berges et al., 2024) — the addressing, and what a learned `M₀` is.
- **Gated DeltaNet** (Yang et al., 2025) and the WY/UT chunkwise form — the
  recurrence and its parallelisation; §3.1 here is the diagonal-decay
  generalisation of the GDN record's §3.8.
- **Channel-wise gated linear attention and delta rules** — per-key-coordinate
  decay, of which the write-set decay is a sparse special case.
- **Sparse access memory** (Rae et al., 2016) — sparse reads and writes over a
  large memory, and restoring overwritten rows in the backward pass rather than
  storing a table per step.
- **Fast-weight product-key memory** (Zhao & Jones, 2026) — the closest prior
  work the paper names.

---

## 6. What the component is

`lumen.sdm`, beside `lumen.gdn`:

```python
@dataclass(frozen=True)
class SparseDeltaMemoryConfig:
    d_model: int
    n_heads: int
    n_slots: int                                   # per head; a perfect square
    initial_memory: Literal["zero", "learned"]     # required — §3.5
    n_writes: int = 64
    n_reads: int = 64
    key_norm: Literal["softmax", "l2"] = "softmax"
    decay_weighting: Literal["write_set", "key"] = "write_set"
    beta_max: float = 2.0                          # untested — §3.3
    chunk_size: int = 32                           # to be measured
    norm_eps: float = 1e-5
    dropout: float = 0.0
    backend: Literal["reference", "triton"] = "reference"   # §3.13

class SparseDeltaMemory(nn.Module):
    def init_state(self, batch, device=None, dtype=None) -> SparseDeltaMemoryState
    def forward(self, x, *, state=None, return_state=False)
    def step(self, x_t, state)
    def read(self, x_t, state)                       # a step with no write
    def residual_out_projections(self) -> tuple[nn.Module, ...]
    def initial_memory_parameters(self) -> tuple[nn.Parameter, ...]   # §3.5
```

`d_v = d_model / n_heads`; there is no `expand_v`, because `n_slots` is the
knob that grows the state without growing compute, and `d_v` grows both. Seams
for subclassing, named for their purpose as in GDN: `_address` (product keys),
`_features` (everything the kernel takes), `_scan` (the kernel), `_out`.
Exported from `lumen` and `lumen.sdm`; the state is a registered pytree.

`recompute_pairwise`, an attribute defaulting to `False`, is a memory dial
(§3.12) for the reason `Stack.recompute` is one: not part of the function, not
in the `state_dict`, and the same weights may be trained with it on and served
with it off. `_scan` passes it to the kernel while training.

Kernels in `lumen.sdm.reference`, not exported:

- `recurrent_sparse_delta` — one position; the decode step.
- `sequential_sparse_delta` — the whole sequence one position at a time; **the
  oracle.** Written to be read.
- `chunk_sparse_delta` — §3.1; what the layer runs. `check_writes=False` for
  callers whose indices are distinct by construction (§3.10). `in_place`
  chooses how the table is held (§3.11): in place by default, functionally under
  a `torch.func` transform or when asked. `group` sets how many chunks have
  their table-free terms computed together, sized by a budget when left out;
  `recompute_pairwise` rebuilds those terms in the backward instead of keeping
  them (§3.12).
- `read_sparse_delta` — one position, read-only; the decode step's readout.

`lumen.sdm.triton_kernels.chunk_sparse_delta` is the same signature less
`in_place`, the route `backend="triton"` takes (§3.13); it falls back to the
reference where its kernels cannot run.

Kernel shapes: `memory (…, N, d_v)`, `write_idx / write_val / log_decay
(…, T, W)`, `read_idx / read_val (…, T, R)`, `v (…, T, d_v)`, `beta (…, T)`,
any leading axes; `memory` may have fewer leading axes and is broadcast. The
kernel takes a log-decay **per write entry**, so it is agnostic to §3.4's
weighting.

---

## 7. Acceptance

Each of these is a standing test in `tests/test_sdm.py`.

1. **Chunkwise == sequential**, `1e-9` in fp64, at chunk sizes 1 through 64,
   with slots written by several positions in one chunk, reads of slots written
   in the same chunk, and reads of slots never written — and a test that the
   fixture really produces those collisions. fp32 to an fp32 bound.
2. **The GDN reduction.** `N = d_k`, `W = R = N`, dense unit keys: the SDM
   oracle equals Lumen's `sequential_gated_delta` to `1e-9` in fp64 — and so
   does the chunkwise path.
3. **The freeze is exact.** A slot no position writes is **bit-identical** after
   a pass, in both paths, including the slots a ragged last chunk's padding
   touches (`torch.equal`, not a tolerance).
4. **Chunk size is inert.**
5. **No overflow at extreme decay** — every write a full wipe, fp32 and fp64:
   finite outputs, and each slot holds exactly its last write.
6. **Any sequence length**, padding exact; **split-and-resume** equals one pass.
7. **Key-weighted decay reaches the oracle** — the kernel is agnostic (§3.4),
   and that is checked rather than assumed.
8. **Duplicate writes within a position are refused**, not mishandled (§3.2).
9. **Gradients** match the oracle for every differentiable input, including the
   entering table, and pass `gradcheck`.
10. **The layer:** `step == forward`; prefill-then-step and chunked forward equal
    one pass; `read` is a step with the write switched off, bit for bit; product
    keys find the exact top `K` against a brute force over all `N`; a learned
    `M₀` at initialisation **is** the zero layer, bit for bit; backward reaches
    every parameter, including a zero table; zero and learned checkpoints do not
    silently load into each other; state size is flat in generated length.
11. **`vmap`** over stacked parameter sets equals the models run one at a time,
    exactly, and streams (§3.10).
12. **On a device** (GPU-marked, run locally): chunkwise equals sequential; the
    forward is bit-identical across runs; the two holders agree; and the
    backward is bit-identical across runs under
    `torch.use_deterministic_algorithms`.
13. **Partners by search equal partners by compare**, exactly, at chunk sizes 1
    through 64, on layouts built to reach every extreme, and on a device
    (§3.1).
14. **The two holders** (§3.11) each pass 1–7 and 9 (8 is checked before a
    holder is chosen); agree bit for bit in the
    forward and to `1e-12` in their gradients; the arena backpropagates twice
    and refuses a second derivative, which the functional holder supports; neither
    writes the caller's table or retains anything table-sized; each chunk
    retains its gathered write rows once; recompute through the arena is
    exact.
15. **Grouping is inert** (§3.12): one chunk per group, uneven groups and all
    chunks at once agree to `1e-9` in fp64, under both holders; a group below
    one is refused.
16. **Recomputing the pairwise terms** (§3.12) gives outputs and gradients
    equal bit for bit, under both holders, on the host and on a device under
    `torch.use_deterministic_algorithms`; of everything pairwise-sized, it
    keeps exactly one array per group — the carry — where the plain path keeps
    many; it nests under `Stack.recompute` to the same gradients; under a
    `torch.func` transform it is refused while gradients are enabled
    (`torch.func.grad` included, whatever surrounds it) and without them is
    the plain path. The layer's dial is off by default and outside the
    `state_dict`, and reaches the kernel only while training: on, a training
    layer keeps less; an evaluating one keeps exactly what it would with the
    dial off, and still runs under `vmap` over stacked parameter sets, as a
    training one does without gradients.
17. **The Triton backend** (§3.13, `tests/test_sdm_triton.py`): as close to the
    fp64 oracle as the reference is -- outputs, final table and every
    gradient -- with slots written several times in a chunk and read twice at
    one position, a ragged last chunk, chunks of 1, 6 and 64, one write per
    position, more entries than one tile, and value widths of 3 and 40; the
    forward without gradients is the forward with them, bit for bit; a table
    handed back and resumed matches one pass; the caller's table is never
    written; the layer agrees across backends, gradients included; with every
    slot selected it is Gated DeltaNet (every segment the whole chunk); an
    unwritten slot is bit-identical after a pass; every write a full wipe stays
    finite and leaves each slot its last write; the backward is bit-identical
    across runs **with deterministic algorithms off**; under `vmap` without
    gradients the kernels run and agree with each parameter set alone; and where
    the kernels do not run -- a CPU tensor, fp64, gradients under a transform, a
    chunk past 64 -- the reference runs, bit for bit. Reference is the default,
    and an unknown backend, or Triton without triton, is refused at
    construction.

---

## 8. What is still open

- **Whether SDM is a delta rule** (§4.2): softmax against l2 keys, delta against
  additive write.
- **What the learned table is worth against a memory layer** (§4.1): the
  equal-size control.
- **How `M₀` should be optimised** (§3.5): ordinary weight decay shrinks what
  the context has frozen.
- **The untested defaults** of §3.9.
- **Where the time goes now.** On the reference path, measured on one machine
  at the defaults (`d_model = 512`), a training step was 204 ms against Gated
  DeltaNet's 42, about half of it elementwise work over the pairwise arrays and,
  on narrow layers, the host issuing the loop's operations. That lever was
  pulled: §3.13 forms no pairwise array and issues one launch per group. What is
  left is the Triton path's own cost, and it is a different kind. Its walks are
  chains of dependent gathers -- latency, not bandwidth: at `d_model = 512` a
  step moves perhaps a third of what the card's memory could in the time it
  takes. The candidates are fewer rounds per chunk, keeping less (the read rows
  could be rebuilt by undoing the writes in reverse, since the write rows are
  kept), and, for one or two sequences, more parallelism than splitting value
  columns gives. Unmeasured on any card with tensor cores, or with more shared
  memory per block than 48 KB -- where the cap on chunk length may not bind.
