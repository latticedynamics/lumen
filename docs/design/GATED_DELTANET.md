# Gated DeltaNet — design record

**Status:** design settled, no code merged yet. This record is written before
the implementation, and the implementation is expected to match it. Where the
two disagree, one of them is a bug.

Gated DeltaNet is the component Lumen was built for. Four independent
implementations of the same layer accumulated across four projects over about a
year. They were not copies — each had drifted, and three had grown something the
others lacked. That is the failure mode this library exists to end: not wasted
typing, but **findings stranded in whichever repository happened to discover
them.**

This record exists so the consolidated layer inherits every one of those
findings deliberately, and so the decisions behind it never have to be
rediscovered.

---

## 1. What the layer computes

A **gated delta rule** associative memory. Each state is a matrix
`M ∈ ℝ^(d_k × d_v)`, read as `o_tᵀ = q_tᵀ M_t`, updated by

```
M_t  =  α_t (I − β_t k_t k_tᵀ) M_{t−1}  +  β_t k_t v_tᵀ
```

with `‖k_t‖ = ‖q_t‖ = 1`, `α_t ∈ (0,1]` a decay, and `β_t ∈ (0, β_max)` a write
strength. The bracketed factor removes whatever is currently stored at address
`k_t` before the new value is written there — the delta rule — and `α_t` forgets
everything uniformly. At `β_max = 2` the factor `1 − β` reaches negative values,
so the update is a reflection rather than only a contraction.

**Projections.** `q, k, v = xW_q, xW_k, xW_v`, optionally through a short causal
depthwise convolution, then SiLU; `q` and `k` are l2-normalised and `v` is not.
Magnitude lives in `β`, where erase and write cannot disagree about it. Total
key width is `expand_k · d_model` and total value width `expand_v · d_model`, so
`d_k` and `d_v` are those divided by the number of states.

**Heads.** A head *is* a state, and a state is a `(key group, value group)`
pair — §3.1. `β` is per key group by necessity and `q` is per state; `α` is
either, selected by `decay=` and defaulting to the key group — §3.4.1.

**Output.** Per-head RMSNorm over `d_v`, a SiLU gate, then a projection —
matching Undertow's output side, so a block can hold either mixer without
knowing which.

**Chunkwise form.** Sequence split into chunks of `C`. Two facts make it
parallel. First, with `γ_t = ∏_{s≤t} α_s` inside a chunk, the substitution
`M_t = γ_t M̃_t` turns the gated recurrence into an **ungated** delta rule on
`M̃` with values `v_t/γ_t` — exact, not an approximation. Second, the resulting
in-chunk system is solved by the WY/UT transform

```
A  =  ( I + tril(diag(β) K Kᵀ, −1) )⁻¹
```

after which the chunk recurrence is **affine in `M`**, so everything that does
not depend on `M` is hoisted out and batched across all chunks, leaving a loop
body of exactly one matmul per chunk.

**Streaming state.** The matrix `M` per state, plus the short-conv cache.
Constant in generated length — a fixed-size memory is the whole proposition.

---

## 2. What the four lineages contributed

| lineage | uniquely had |
|---|---|
| **Consuming model** | crossed key/value grouping (§3.1); independent `expand_k`/`expand_v` with the argument for expanding the *key* side; a frozen config dataclass; the in-chunk decay clamp; fp32 and no Triton, by design |
| **Streaming** | `init_state()` / `step()` / `forward(..., state=, return_state=)`; per-head learned `β` via sigmoid, zero-initialised so `β` starts *exactly* at the fixed-strength engine it replaced; a per-head geometric decay band, optionally learnable; structural masking |
| **Trained** | the lineage with real training behind it — and, on inspection, little else the others lack: most of its length is tokenizer and corpus handling, which is exactly the coupling §5 excludes |
| **Episodic** | **key/query centring** (§3.3); an optional kNN episodic store with a gated read |

Two further modules are **diagnostic instruments**, not implementations, and are
deliberately not merged into the layer: one measures state stable rank, key
participation ratio and an exact horizon decomposition; the other measures
memory half-life from the transport operator's norm and separates
decay-forgetting from erasure-forgetting. Instruments and the thing they measure
stay separable. The first of them decides §3.3, which is precisely why it must
not live inside the layer it judges.

The streaming lineage already has a downstream architecture subclassing it and
reusing its chunk scan unmodified. That is the strongest available evidence that
its API shape survives real reuse, so **the consolidated layer takes its shape.**

---

## 3. Decisions — settled

### 3.1 A head is a `(key group, value group)` pair

Standard practice gives every head its own key *and* its own value: `H` states
from `2H` projections. One lineage instead defined `G_k` key groups and `G_v`
value groups and instantiated a state for **every pair** — `H = G_k · G_v`
distinct associative memories out of `G_k + G_v` projections. No two are
duplicates, because the grouping is *crossed* rather than nested.

That structure is a genuine generalisation in one direction and a genuine
restriction in another, and the restriction is the interesting half:

> A Cartesian product equals its own diagonal only when both factors are
> singletons. So a layer that instantiates *all* pairs cannot express the
> ordinary one-key-one-value arrangement at all — the two families meet at
> `H = 1` and nowhere else.

A layer that cannot be configured into the standard arrangement is a layer that
can never be checked against one. **So the consolidated layer parameterises the
assignment rather than the grid:** a head is a `(key group, value group)` pair,
and the configuration chooses the multiset of pairs.

| assignment | is |
|---|---|
| all pairs | the crossed layout |
| the diagonal | the ordinary arrangement — one key and one value per head |
| one key group, `H` value groups | a single shared address space, `H` payloads |
| `H` key groups, one value group | the grouped-query-attention shape |

Named constructors are provided for those four; an arbitrary assignment is
accepted and documented as the slow path.

**The constraint that keeps it fast:** sorted by key group, the heads must form
a `G_k × m` rectangle. All four rows satisfy it, and it is what lets the UT/WY
inverse be built **once per key group** and reused across every value group
sharing it — the `O(C²d_k + C³)` term is paid `G_k` times, not `H` times. For
the crossed layout the value tensor is then a stride-0 broadcast, and for the
diagonal it is a permutation. Both are views; neither copies.

What this buys is not performance. It is that the component has **a reference
point outside itself.** A published arrangement it can be configured into is a
thing its numbers can be checked against, and Lumen's entire premise is that a
difference in two projects' numbers should be a difference in their experiment.
That argument does not stop at the edge of these repositories.

What it does **not** buy, and the record should be read as claiming nothing
more: representing an arrangement is not reproducing an implementation.
Checkpoint compatibility would additionally require matching a short conv, a
`β` parameterisation, a decay parameterisation and a norm placement. That is a
separate and larger job.

### 3.2 Crossing is available, and is not the default

The crossed layout was the motivating idea of one lineage, and the only
controlled comparison of it — matched to a fraction of a percent in parameters,
several seeds per arm, one shared harness — went against it. Every
single-key-group seed beat every crossed seed, with disjoint ranges at the
strongest significance the seed count can express, and the crossed arm was also
materially slower. That last part follows from the layer's own compute argument
rather than from luck: the shared UT/WY inverse is built `G_k` times, and that
count is *minimised* at `G_k = 1`.

The mechanism check is the part worth recording, because it makes the result
sharper rather than merely repeating it. Crossing promised **horizon
diversity** — several key groups mean several independent forgetting timescales
in one layer — and it delivered exactly that, visibly, in the per-group horizon
decomposition. The aggregate memory behaviour was nonetheless indistinguishable
between the two arms. **Crossing produced the diversity, and the diversity
bought nothing.** The instrument also said why it could not have helped here:
the states were far from rank-limited, so extra address spaces were solving a
problem that model did not have.

So the crossed machinery stays — it is the correct generalisation, and the
negative result is one setting, at one scale, where state pressure was not the
binding constraint. But the optimum sits at the `G_k = 1` boundary, where
crossing degenerates, and **the consolidated layer is therefore simpler than the
implementation that motivated it.**

That the layer is *named for its recurrence and not for that structure* is a
deliberate consequence. The crossed layout is a configuration carrying a
documented negative result; naming the component after it would have promoted
the one idea its own evidence argues against.

### 3.3 The address side is the scarce resource — and there are three prices for it

This is the finding that only became visible with the lineages side by side, and
it is the clearest example of what consolidation is *for*.

Since `rank(M) ≤ min(d_k, d_v)`, the key axis bounds how many non-interfering
things a state can hold. Three of the four lineages are, without having said so,
three different answers to *how do you buy more addressable directions*:

**More key groups.** Settled negative — §3.2.

**Wider keys.** One lineage's founding thesis: expand the key side rather than
the value side, because value width only buys embedding room in front of the
output gate. The supporting comparison is better than the thesis: halving key
width cost real loss *in a configuration where the value dimension bound the
rank in both arms*, so the slot ceiling never moved. That isolates the effect to
key **resolution** rather than capacity — cross-talk between l2-normed keys
falling as `1/√d_k` — which is a different mechanism from the one the thesis
was originally argued from, and the one that survives.

**Centred keys.** The episodic lineage's contribution, and pure geometry:

> Only **direction** is an address, because keys are l2-normalised before use.
> So if a head's key cloud sits off the origin, every key it can emit is crowded
> into a cone, and the head cannot use the width it has.

That cost is not a vague worry; it is computable in advance. Write
`r = ‖μ‖ / rms‖d‖` for the offset ratio of a head's key cloud — the norm of its
mean against the typical spread about that mean. Two keys drawn from the cloud
have

```
E[ cos(k₁, k₂) ]  ≈  ‖μ‖² / (‖μ‖² + E‖d‖²)  =  r² / (1 + r²)
```

so the offset imposes a **cosine floor** below which no two addresses the head
can produce are ever more distinct than that. An `r` of 1 already floors every
pair at 0.5. Subtracting a learned per-head centre before normalising removes
the floor and hands the range back. The diagnostic that measures the recovery is
the **participation ratio** of the emitted unit keys — how many of its `d_k`
directions a head actually addresses.

That off-origin key clouds are a real property of trained models rather than a
theoretical possibility, and that centring recovers a substantial multiple of
the addressable range, was established in the episodic lineage. The size of that
recovery is not published here; the formula above is, and it is the useful part,
because it turns a one-model observation into something measurable on any model
before deciding.

Centring costs `2 · H · d_head` parameters — effectively free — and is
zero-initialised, so a checkpoint from an uncentred model loads and behaves
identically **by construction**.

**Set the second and third of these side by side and they look like the *same
claim*** — get more addressable directions — bought at wildly different prices.
They had never been tested against each other, because they lived in different
repositories. The uncomfortable possibility was that **key expansion pays
parameters to fix a problem a zero-cost centring already fixes.**

### The comparison, and what it settled

Four arms — key expansion wide/narrow crossed with centring off/on, several
seeds each, one harness — read on loss *and* on the geometry above.

**They are not the same claim, and the result is unusually clean.** Key width
won by a margin several times the seed-noise floor, with disjoint ranges, and
**by the same margin whether centring was on or off.** Centring was inside seed
noise at both key widths, also twice, and the interaction between the two
factors was an order of magnitude smaller than either. Narrow-and-centred —
the outcome that would have been most interesting, since it is cheaper on both
parameters and state — was decisively *worse* than wide-and-uncentred.

So key expansion is **not** paying for something centring gives free. The
possibility is closed.

**Why, and this is the part that generalises.** The diagnostics say the two
levers are not buying the same good:

- The off-origin geometry is **real** in these models — the measured offset
  ratios imply a cosine floor of a third or more between any two addresses a
  head can emit. The described effect is present, not imagined.
- The centres **moved substantially** during training, so the mechanism engaged
  and the null is not an inert flag.
- But the offset ratio **did not fall**, and the participation ratio did not
  rise. The heads were addressing well under a quarter of their available
  directions in every arm — nowhere near saturation.

**Nothing here is addressing-*capacity* limited, so no capacity lever pays.**
That is the same explanation the crossed layout's mechanism check produced
(§3.2): extra address spaces solve a problem these models do not have. Both
capacity levers — more key groups, and centred keys — come back null, for one
reason. Key width, meanwhile, is a **resolution** lever: it reduces cross-talk
between l2-normed keys as `1/√d_k`, and it wins in exactly the configuration
where the rank ceiling cannot move.

**Capacity and resolution are different goods, and only one of them is scarce
here.** That is worth more than either individual result, because it says what
to measure before spending parameters on addressing: check the participation
ratio first. If the heads are not near saturation, capacity is not the
constraint.

### What was not tested, and is a different experiment

The originating measurement subtracted the key cloud's **empirical mean**. The
layer learns its centre by gradient descent. Those are not the same
intervention: gradient descent has no incentive to set the centre to the mean
unless doing so reduces loss, and the diagnostics above show it did not choose
to. So the geometric claim — that removing the offset recovers addressable
directions — is untouched by this result; what was tested is whether a *freely
learned* centre finds that solution and benefits from it. It does not, here.

A centring tied to a running mean rather than learned would test the geometric
claim directly. It is a different component and nobody has run it.

**`expand_k` remains a required argument with no default** (§3.5). The result
above is one training scale on one corpus, and it favours the wide setting
consistently; that is a reason for the record to say so, not a reason for the
library to choose. **Centring ships zero-initialised and off by default** — it
costs almost nothing, it is bit-identical when off, and the case for turning it
on has not been made.

### 3.4 `β` must live on the key group; `α` need not — and that seam is now taken

The layout of §3.1 binds three things at once: which addresses a state uses,
which payload it writes, and **how fast it forgets** — because the lineage it
came from puts both `α` and `β` on the key group. One of those three bindings is
an implementation choice and the other is forced, and it is worth knowing which.

The UT/WY inverse is `A = (I + tril(diag(β) K Kᵀ, −1))⁻¹`. It is a function of
`β` and `K` — **and not of `α` at all.** The decay absorption of §1 substitutes
`M_t = γ_t M̃_t` and the `γ` cancels straight through the recurrence, leaving

```
M̃_t  =  (I − β_t k_t k_tᵀ) M̃_{t−1}  +  β_t k_t (v_t / γ_t)ᵀ
```

an ungated delta rule whose transform never sees `α`. Therefore:

- **`β` is per key group by necessity.** It enters the inverse that the layout
  shares. Varying it per state would mean a different `A` per state, and the
  reuse that makes the crossed layout cheap disappears.
- **`α` can be per state, and the `O(C³)` solve stays shared.** Not "at
  essentially no cost", which is what this section claimed while the seam was
  still hypothetical — see §3.4.1 for what it actually costs and what it does
  not. The loop is still one matmul per chunk.

That asymmetry is not idle, because of what §3.2 found. Crossing's promised
benefit *was* horizon diversity, and it obtained that diversity by splitting the
address space — which is what it lost on. Meanwhile a second lineage, entirely
independently, gave every head its own decay outright: a geometric band of
timescales spread across the heads, fixed or learnable, with no grouping
structure implied at all. Someone else thought timescale diversity was worth
engineering deliberately, and paid nothing for the addresses.

**Crossing bought horizon diversity the expensive way, and there is a free way.**
That reframes the negative result rather than softening it: the crossed layout
was not wrong to want several timescales, it was wrong about what it had to give
up to get them.

The seam is now **taken**, as `decay="key_group" | "state"`. What follows is why
it is a name rather than a boolean, what it costs, and the one claim above that
did not survive contact with the 0.3 kernel.

### 3.4.1 Taking it: solve-then-damp, and why `β_max = 2` is what makes it free

**A name, not a flag.** The obvious surface is `per_state_decay: bool = False`.
It was rejected because this section names *three* arrangements, not two — the
third is the geometric band of fixed timescales, which is not a projection at
all. A boolean can only say *not the other thing*, so admitting the band later
would mean a second flag cross-checked against the first, which is exactly the
failure mode §6.4 of the 0.2 record climbed out of by deriving `n_heads` from
the layout. `decay="state"` says what the layer *is* and leaves room.

**The parameterisation is an offset, not a wider projection.** `log_alpha[h] =
−softplus(a_proj(x)[g(h)] + a_offset[h])`, with `a_offset` zero-initialised.
Three things follow, and the first is why it is shaped this way: at zero it is
the shared layer exactly, so the arrangement can be turned on without also
changing what the layer computes. It costs `H` scalars rather than
`d_model · (H − G_k)` weights. And the geometric band becomes a different *init*
of this same parameter instead of a third code path. Symmetry breaks as soon as
training starts, because each state's `α` now multiplies a different `M`.

**The cost, corrected.** The claim above — "essentially no cost" — was written
against the pre-0.3 kernel and is not right for the one that shipped. `E` reads
`γ_C`, the chunk-exit factor and `Ŵ`, all of which carry `α`, so the transition
becomes per state. Two consequences:

- `w` (`O(C²d_k)`) and `transition` (`O(C·d_k²)`) go `m×`, and the `m`-into-the-
  value-axis fold in the loop is unavailable, since it exists precisely because
  `E` is shared. Same FLOPs in the loop, one more batched axis.
- **The `O(C³)` triangular solve does *not*.** This is the part worth the
  section.

Everything downstream reads `A ⊙ rel`, and `inv((I+N) ⊙ rel) = inv(I+N) ⊙ rel`
by the homomorphism §3.8 already relies on. So there are two routes: **damp then
solve** (fold each state's `rel` in, solve `m` times — the rectangle rule's
`O(C³)`-paid-`G_k`-times gone) or **solve then damp** (solve `I + N` once, since
it holds no `α` at all, and scale each state's copy after). The second is only
safe if the *undamped* inverse stays bounded, because nothing shrinks it on the
way through.

It does, and `β_max = 2` is exactly why. Forward substitution on `I + N` is the
delta rule itself: with `Z_i = Σ_{m≤i} k_m ⊗ X[m]`,

```
X[i] = e_i − β_i k_iᵀ Z_{i−1},    Z_i = (I − β_i k_i k_iᵀ) Z_{i−1} + k_i ⊗ e_i
```

and for a unit key `‖I − β k kᵀ‖₂ = max(1, |1 − β|)` — **exactly 1 for every β in
[0, 2]**. Identity at 0, reflection at 2, expanding only past it. So `‖Z‖` grows
at most linearly in `C` and the inverse cannot blow up.

Measured on one machine, near-identical unit keys, fp64, `max|inv(I+N)|`:

| `β_max` | C=16 | C=64 | C=256 |
|---|---|---|---|
| 1.00 | 1.000 | 1.000 | 1.000 |
| 2.00 | 1.981 | 1.982 | 1.983 |
| 2.10 | 5.79 | 176 | 1.7e8 |
| 2.50 | 546 | 5.7e10 | 7.0e42 |

Flat in `C` up to the ceiling, geometric past it. And in fp32 against an fp64
oracle, across correlation, half-life and timescale-spread grids, solve-then-damp
was **equal or better** than damp-then-solve everywhere — never worse.

This is a *different object* from the `1/γ` that §3.8 rejects, and the difference
is the whole argument: `1/γ` is unbounded above and reaches fp32's ceiling at
accumulated decay 88.7, while this is bounded by 2 however long the chunk. The
resemblance is superficial and the conclusion inverts.

**So `β_max = 2` stops being a taste ceiling and becomes an invariant**, and it
is enforced on construction rather than documented. Note this was *already* true
of the shared path: `rel → 1` as `α → 1`, so slow forgetting leaves the solve
undamped either way. A `β_max` of 3 worked while a model forgot quickly and broke
once it learned to remember — which is the worst shape a bug can have.

**Still open**, and deliberately not chosen: whether per-state decay is *better*.
It is an arrangement the layer can now express and an experiment nobody has run.
The default stays where the contributing lineages put it.

### 3.5 Defaults are earned, and these two were

A default is a recommendation whether or not it is labelled one, so the rule is
that nothing is canonised that has not been compared. Two things have been.

**`expand_k` defaults to a wide key side, and `expand_v` to a narrow one** —
inverting the more common ratio. §3.3's four-arm comparison is the evidence: key
width won by a margin several times the seed-noise floor with disjoint ranges,
independently at both centring settings. It is one training scale on one corpus,
which is why the record states the strength of the result rather than presenting
it as settled physics — but a library that refuses to have a default forces
every caller to invent one, with less evidence than this.

**The head layout defaults to a single shared address space** (§3.1), for the
same reason and from the same family of comparisons.

**Centring ships zero-initialised and off.** It costs almost nothing, it is
exactly inert when off, and §3.3 found no benefit from turning it on. It is
present because the mechanism is real and the parameter is nearly free, not
because it is recommended.

What is **not** canonised: two disagreements the lineages have about the decay,
independent of each other and of §3.4's question — whether it is
**content-dependent** (a projection of the input) or a **fixed or learnable
per-head band**, and where it **lives**. No controlled comparison exists. This
version ships one form, says so, and does not pretend the choice was made on
evidence.

`β_max = 2` is the easy case: all four lineages already agree, and it is an
existing option in delta-rule implementations generally.

### 3.6 fp32 reference, no accelerated path in this version

The reference path is fp32 and depends on nothing but `torch`, because that is
the path that runs everywhere.

There is no Triton path here, and the reason is the house rule rather than
neglect: **an accelerated path ships when it has been measured to beat the
reference on the target machine, and not before.** Undertow's did, and shipped
opt-in with its numbers in the record; this one has not been written, so there
is nothing to report and nothing is claimed. When it is, the same standard
applies — measured here, on this implementation, labelled as one machine's
numbers, and selected explicitly rather than auto-detected, so that two installs
of this layer are never quietly two different objects.

Two numerical traps from the contributing lineage are preserved in the reference
because both are expensive to rediscover:

- The unit-lower-triangular inverse is computed by forward substitution, **not**
  by the Neumann/binary-expansion identity. That identity is algebraically exact
  — the matrix is nilpotent — but it forms intermediate powers that overflow in
  fp32 once the keys are correlated, which is exactly what a trained model
  produces.
- The chunk loop consumes its per-chunk operands by unbinding them in advance
  rather than indexing inside the loop. Indexing a tensor inside the loop makes
  autograd allocate and accumulate into a full-size zero buffer *once per
  iteration*, and on this layer's shapes that memory traffic dominated the
  step.

### 3.7 Bit-exactness before adoption

The consolidated layer must reproduce the originating lineage's outputs to
**1e-9 in fp64** before any project is switched to it, forward and single-step.

**The dtype is part of the tolerance and not a detail.** In fp64 the chunkwise
and sequential paths agree to round-off — order 1e-15 — so 1e-9 is a wide
margin that still catches any structural error. In fp32 the same two paths
differ by order 1e-6 for reasons that are arithmetic rather than algorithmic:
the chunkwise form reassociates the sum. A gate of 1e-9 stated without a dtype
would be one the shipped path cannot pass, and specifying an unmeetable
tolerance is worse than specifying a loose one.

So there are two claims, and only the first is tight:

1. **The algorithm is the same**, checked in fp64 at 1e-9. This is where a port
   error would show.
2. **The fp32 path is the fp64 path evaluated in fp32**, checked at a bound
   consistent with fp32 round-off on these shapes and *not* tightened past what
   the arithmetic allows.

Undertow's `1e-6` is a different kind of claim again — a windowed path against a
dense oracle, two genuinely different algorithms for one function. Here the
comparison is one algorithm reached by two routes, which is what earns the
tighter number once the dtype is named.

Consolidation that silently changes numerics invalidates every experimental
record that depended on the old code. Merge → verify → switch → evolve, as four
steps. Not one.

### 3.8 The decay is carried relatively, and the chunk size is not a modelling choice

The chunkwise form absorbs the decay by writing the state as `M_t = γ_t M̃_t`,
turning the gated recurrence into an ungated delta rule on `M̃` carrying values
`v_t / γ_t`. That is the derivation. It is a separate question how the code
should *compute* it, and the two answers are not numerically equivalent.

**What appears in an answer is a ratio, and the ratio is bounded.** Every term
reaching an output carries `γ_t / γ_i` for some `i ≤ t`. Writing `g = log γ`,
which is non-increasing because `α ≤ 1`, that ratio is `exp(g_t − g_i)` with a
non-positive exponent: it lies in `(0, 1]` **by construction**, for every pair
the computation can form.

**Factorising it splits a bounded quantity into two unbounded ones.** The
obvious implementation materialises `γ_t` and `1/γ_i` separately and multiplies
them back together at the end. The first shrinks, the second grows, and their
product is the well-behaved thing — but the growing factor now exists as a
number, and it has to survive every operation between the two.

Two failures follow, and they are of different severities.

- **Overflow.** In a format with maximum finite value `F`, `1/γ` is
  unrepresentable once accumulated decay within a chunk exceeds `ln F` — about
  `88.7` in binary32, about `709` in binary64. Past that the value is `inf`, and
  since a fully-decayed contribution is multiplied by something approaching
  zero, the next step is `inf · 0 = nan` and the whole batch is lost.
- **Dynamic range within one accumulation.** Long before overflow, a sum mixing
  terms whose `1/γ` differ by more than the format's precision — `2⁻ᵖ` for a
  `p`-bit significand — loses the small terms into the rounding error of the
  large ones. In a delta rule the small term is the *recent* position and the
  large one is the stale position being forgotten, so the precision is lost
  exactly where the layer is supposed to be paying attention.

**The relative form computes the ratio directly.** `rel[t, i] = exp(g_t − g_i)`,
formed once and used where `i ≤ t`. Neither half is ever built, nothing exceeds
`1`, and the only reachable failure is underflow to zero — which is the *correct
answer*, meaning that position has fully decayed. The failure mode inverts from
"produce a poison value" to "round a negligible contribution to nothing."

**Carrying it costs one masked multiply, because the decay commutes with the
inverse.** The UT/WY transform inverts `I + N` for strictly-lower-triangular
`N`. For lower-triangular `A`, `B` and any `D[i,j] = d_i / d_j`,

```
    (A ⊙ D)(B ⊙ D) = (A B) ⊙ D
```

since the intermediate factors telescope. So `A ↦ A ⊙ D` is an algebra
homomorphism on lower-triangular matrices, `I ⊙ D = I`, and therefore it
commutes with inversion:

```
    (I + N ⊙ D)⁻¹ = (I + N)⁻¹ ⊙ D
```

Folding the relative decay into `N` *before* the solve yields the correctly
scaled inverse for free. The scaled quantities differ from the unscaled ones by
a factor of `γ_i` per row, which is exactly the factor the readout and the state
update would otherwise have had to divide back out.

The cost is a `[C, C]` tensor where a `[C]` vector sufficed, and it folds into
masks the kernel already builds, so it is two elementwise operations on tensors
already in flight.

**Rejected: recentring the exponent.** Shifting the accumulated decay to
`±half` instead of `0..full` doubles the usable range at the same precision. It
is a partial fix that still changes numerics, and if the verification cost is
being paid it should buy the whole thing.

#### Why this deletes a configuration option rather than retuning one

An implementation that materialises `1/γ` needs a ceiling on accumulated decay
within a chunk to stay in range. Such a ceiling is a bound on `Σ log α` over `C`
positions, so it implies a **shortest expressible half-life** of

```
    C · ln 2 / ceiling
```

which is *linear in the chunk size*. The chunk size exists to tile the sequence
for the hardware. Under a ceiling it also decides how fast a head is permitted
to forget, and raising it for throughput silently lengthens the shortest memory
the model can represent — a performance dial reaching into the hypothesis class.

That is the class of defect this repository exists to catch, so the option is
**removed rather than given a better default**. A parameter whose correct value
depends on a parameter chosen for unrelated reasons is not a parameter worth
tuning. With the relative form there is no range to bound and nothing to
configure: `chunk_size` becomes what it claims to be.

The clamp's reach is visible in the call graph as well as the algebra. It had to
be applied where the features are computed, upstream of *both* the chunkwise and
the single-step paths, because a model trained under a binding floor must decode
under the same one. But the single-step path never forms a cumulative product
and never had the numerical problem. A guard can live next to its hazard; this
one could not, which is what a modelling constraint looks like from the outside.

**This change is not numerically transparent and is not meant to be.** It is the
`evolve` in merge → verify → switch → evolve, taken deliberately and after the
switch, so that "this layer differs from the one it replaced" and "the new
factorisation differs from the old one" remain separable questions. A consumer
carrying an experimental record across it is comparing two model classes, not
two implementations.

---

## 4. Relation to existing work

**No novelty is claimed for the recurrence.** It is an established layer and
this record names it accordingly; what is assembled here is a set of
configuration choices and one small addition, and the honest account is below.

- **The delta rule** is classical, and its use as an update for fast-weight
  memories in sequence models is established work — linear attention read as a
  fast-weight programmer, and the delta rule as the write operator that replaces
  rather than accumulates at an address.
- **The chunkwise-parallel form via the WY / UT representation** is likewise
  established: the WY representation of a product of Householder-like updates is
  standard numerical linear algebra, and its application to parallelising the
  delta rule over sequence length is published work.
- **Scalar data-dependent gating** — a decay multiplying the whole state — is
  the state-space-model line, and **combining it with the delta rule is
  published as Gated DeltaNet.** This layer is that layer. Where it differs is
  head layout (§3.1), the addressing choices of §3.3, and nothing about the
  recurrence.
- **`β` beyond 1**, making the update a reflection rather than a contraction, is
  an existing option in delta-rule implementations, not something introduced
  here.
- **Per-head decay bands** — a geometric spread of fixed forgetting rates across
  heads — are the retention-network line, and one contributing lineage took the
  idea directly from there.
- **Sharing projections across heads** is grouped-query and multi-query
  attention. The crossed layout of §3.1 is a different thing and the distinction
  is worth stating precisely: grouped-query sharing is **nested** — a group of
  queries shares one key/value pair — whereas crossing takes the **product** of
  two independent groupings, so that no two states are duplicates. The
  grouped-query shape is one row of §3.1's table, not the general case.
- **A channel-wise decoupled erase and write gate** is a further rung, published
  separately, and is deliberately not climbed here (§5).

**Key centring (§3.3) is the one piece with no attribution offered.** Normalising
queries and keys before the score is well established; subtracting a *learned
per-head centre* before that normalisation, specifically to recover addressable
directions lost to an off-origin cloud, is not something we have found in the
literature. That is a statement about our search, not about the literature: if
it has been published under some name, we have not found it and would rather be
told. It is in any case a small and natural modification, and the geometry that
motivates it is elementary.

---

## 5. What the consolidated layer is

Shape and streaming from the streaming lineage, state algebra from the
consuming-model lineage, centring from the episodic lineage, head layout generalised so the
ordinary arrangement is reachable:

- frozen `GatedDeltaNetConfig`, validated on construction, with `d_model` and
  `layout` the only required arguments — the width ratio has an earned default
  (§3.5), and `n_heads` is a **property derived from the layout** rather than a
  second field. A head count does not determine a head arrangement, so a config
  accepting one would have to choose on the caller's behalf (§3.1); making the
  layout explicit is the same argument as §3.2's, applied to the constructor
- `init_state()` / `step()` / `forward(..., state=, return_state=)` — the house
  interface, so a block can hold this or Undertow interchangeably
- head layout as a `(key group, value group)` assignment, with named
  constructors for the crossed, diagonal, shared-key and grouped-query shapes
  (§3.1); crossed retained as the generalisation, not as the default (§3.2)
- **any sequence length.** A sequence that does not fill its last chunk is
  padded internally with no write and no decay, which leaves the state and every
  real output exactly as they would have been. A corpus is whatever length it
  is, and a library that rejects one is a library every caller wraps
- `expand_k` and `expand_v` as dials, defaulted (§3.5)
- learned key/query centres, zero-initialised, off by default (§3.3)
- `β ∈ (0, β_max)`, `β_max = 2` and **enforced as a ceiling**, because past it
  the UT/WY solve expands geometrically in `chunk_size` (§3.4.1); scalar
  per-key-group `β`, and `α` per key group *or* per state via `decay=` (§3.4.1),
  defaulting to the key group
- an in-chunk decay clamp, bounding `1/γ` within a chunk
- fp32 chunkwise, `chunk_size` a power of two, no Triton dependency (§3.6)
- the sequential single-step form retained, both as the decode path and as the
  oracle the chunkwise path is verified against

Deliberately **out** of this version:

| excluded | why |
|---|---|
| kNN episodic store | an orthogonal memory *architecture* — it carries its own gate and a contrast-loss stash. Its own component, not a layer detail |
| channel-wise decoupled erase/write gates | scalar decay commutes out of the recurrence, which is what makes the chunkwise form work; a diagonal decay does not factor the same way. New derivation *and* new kernel |
| ~~per-state decay~~ | **landed** as `decay="state"` (§3.4.1). Off by default: it is an arrangement the layer can express and an experiment nobody has run |
| a fixed geometric band of timescales | reachable as an init of `a_offset` rather than new code, which is why `decay` is a name and not a boolean (§3.4.1). Not canonised — the comparison against a learned per-state decay has not been run |
| accelerated kernels | ship when measured to win here (§3.6) |
| byte tokenizer, corpus handling | not the layer's business — this coupling is what made one lineage the longest of the four |

---

## 6. Acceptance

Before any project switches to this layer:

1. **Chunkwise == sequential** to `1e-9` **in fp64**, at every head layout in
   §3.1 including the diagonal and the crossed cases, and across a chunk count
   greater than one. The same comparison in fp32 is held to a bound consistent
   with fp32 round-off, not to 1e-9 (§3.7).
2. **`step()` == `forward()`** to `1e-9` in fp64, and state carried across a
   chunked `forward` == one pass, split at several points.
3. ~~**The port reproduces the originating lineage**~~ — **discharged, and no
   longer runnable.** It was met exactly (`0.0e+00`, not merely within
   tolerance) and then deliberately spent by §3.8's reformulation, which changes
   what this layer computes on purpose. A gate whose premise the project has
   since chosen to abandon is history, not a check; the sequential path in the
   reference is what the kernel is now answerable to.
4. **Peak memory flat** across a 4× sweep of generated length.
5. **Zero-initialised centring is an exact identity** — same outputs, bit for
   bit, as the same weights without it. The claim in §3.3 is "by construction",
   and a construction claim should be held by a test rather than a comment.
6. The three inverse implementations agree, and the reference is the forward
   substitution (§3.6).
7. **The chunk size is inert**, at decay strong enough that a `1/γ`
   factorisation could not have represented it: outputs agree across a sweep of
   chunk sizes to fp64 round-off (§3.8). This is the gate the decay
   reformulation exists for — a numerics change with no behavioural test is
   indistinguishable from a refactor.
8. **No overflow at extreme decay.** Accumulated decay within a chunk set well
   past the fp32 `exp` ceiling produces finite outputs, and fully-decayed
   contributions are zero rather than `nan` (§3.8).
9. **The shortest expressible half-life does not depend on the chunk size.**
   Driving `α` low yields the same half-life at every chunk size, and one below
   `C · ln2 / 8` — the floor the removed ceiling would have imposed.
10. **Per-state decay reaches the oracle**, to `1e-9` in fp64 at every layout,
    across a chunk-size sweep, and across a split-and-resume — the same bar
    gate 1 sets for the shared arrangement, because it is the same recurrence.
11. **Per-state decay does not widen the solve.** The triangular inverse is
    built once per key group in *both* arrangements. Held structurally, by
    recording the shape the solve is called with, rather than in a benchmark
    that nobody runs (§3.4.1).
12. **Zero-initialised `decay="state"` is the shared layer** — to the fp64 gate,
    not bit for bit, and the difference is the point: the offset is zero so the
    decay values are identical, but a per-state `log_alpha` sends the kernel
    down solve-then-damp while the shared one damps first. Claiming bit-identity
    here, as gate 5 does for centring, would be false.
13. ~~**Turning on per-state decay does not change the shared path**~~ —
    **discharged at merge, and not runnable afterwards.** The kernel reproduced
    the pre-`decay` implementation **bit for bit** — not within a tolerance — over
    96 configurations: every named layout × {fp32, fp64} × `C ∈ {8,16,32,64}` ×
    `T ∈ {64, 100}`, checked with `torch.equal` against the implementation
    extracted from the preceding commit. Like gate 3, it compares against
    something that no longer exists once it has passed, so it is history rather
    than a check. What makes it hold is structural and does survive: the shared
    arrangement keeps damp-then-solve, and the state axis is carried as size 1
    rather than sliced away, which is a shape change and not an arithmetic one.
14. **`β_max > 2` is refused on construction.** The ceiling is an invariant of
    the UT/WY solve, not a preference (§3.4.1), and it is unsafe in the *shared*
    arrangement too.

Merge → verify → switch → evolve, as four steps. Not one.
