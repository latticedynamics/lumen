# Sparse Delta Memory — design record

**Status:** landed in 0.6.0, as `lumen.sdm`; in 0.6.1 the table is written in
place (§3.11) and partners are found by search (§3.1). The implementation is
expected to match this record; where the two disagree, one of them is a bug.

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

For GDN that guarantee is free. **Here it costs a table copy per step:**
`O(N·d_v)` per head per position, against `O((W+R)·d_v)` of actual work. For
small tables that is tens of MB per step and acceptable; at `N = 2¹⁸,
d_v = 512` it is half a gigabyte per step per layer. The reference keeps the
guarantee. An in-place decode path — explicit about aliasing, opt-in — is a
later decision with its own measurement, not a refactor of this one.

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
| CUDA/Triton kernels, a bf16 table, fp8/int4 snapshots | a faster path, earned by measurement. The reference table is the model dtype, and the gradient into `M₀` is not accumulated through bf16 atomics |
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
- `chunk_size` — inert numerically; a speed-and-memory dial. Measured on one
  machine, with the table in place (§3.11) and partners found by search
  (§3.1), the default of 32 was fastest at every table size from 32² to 256²
  (203–230 ms a step, against 242–270 at 64 and 289–339 at 16); memory grows
  with it. Before the search it was 16. The optimum sits where a chunk's fixed
  cost — launches, the Python loop — meets the pairwise terms' growth in `C`;
  it has already moved once on one card, and a faster device shrinks the second
  without the first, so it is not settled for others.

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

Kernels in `lumen.sdm.reference`, not exported:

- `recurrent_sparse_delta` — one position; the decode step.
- `sequential_sparse_delta` — the whole sequence one position at a time; **the
  oracle.** Written to be read.
- `chunk_sparse_delta` — §3.1; what the layer runs. `check_writes=False` for
  callers whose indices are distinct by construction (§3.10). `in_place`
  chooses how the table is held (§3.11): in place by default, functionally under
  a `torch.func` transform or when asked.
- `read_sparse_delta` — one position, read-only; the decode step's readout.

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

---

## 8. What is still open

- **Whether SDM is a delta rule** (§4.2): softmax against l2 keys, delta against
  additive write.
- **What the learned table is worth against a memory layer** (§4.1): the
  equal-size control.
- **How `M₀` should be optimised** (§3.5): ordinary weight decay shrinks what
  the context has frozen.
- **The untested defaults** of §3.9.
- **Where the time goes now.** Measured on one machine at the defaults, a
  training step is 221 ms against Gated DeltaNet's 42 at the same width.
  Finding partners is about 15% of it and table-sized work 1%; the rest is
  elementwise work on the pairwise terms, gathers and their backward, and
  batched matmuls, in roughly that order. At small chunks the device waits on
  the host — 57% busy at `C = 16`, `N = 256²` — so the next lever is fusing a
  chunk's elementwise work, a compiled or custom kernel measured against this
  one, not a new algorithm.
- **An in-place decode** (§3.6). Measured on one machine, the successor copy is
  most of a step once the table is large: at `B = 8`, 36 ms per step at
  `N = 512²` against 0.2 ms for the same arithmetic in place, which also costs
  0.2 ms at every smaller table. Whether that buys an opt-in `step_` against the
  house's successor guarantee is an API decision, not an optimisation.
