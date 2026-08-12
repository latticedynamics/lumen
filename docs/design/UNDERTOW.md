# Undertow — design record

**Status:** implemented. This record was written *before* the code — that is the
house rule — and the code was then held to it. The acceptance conditions in §6
are enforced by the test suite rather than asserted here, so where this document
and the implementation disagree, one of them is a bug.

Undertow is fixed-window causal attention with no positional encoding and an
optional graded window boundary. It is the second component Lumen consolidates,
and it arrived in the unusual position of having two independent implementations
that had *already* converged on the same interface — one written to make two
different mixers interchangeable inside a single block, the other written to
carry a periodic attention kernel. Neither knew about the other's solution to
the parts they did not share.

This record exists so that the merged layer inherits both deliberately.

---

## 1. What the layer computes

Input `x ∈ ℝ^(B×T×d)`, `h` heads, head dimension `d_h = d/h`, window `W ≥ 1`,
plateau `P` with `0 ≤ P < W` (or `P = None`, a hard window).

**Projections.** `Q, K, V = xW_q, xW_k, xW_v`, reshaped to `(B, h, T, d_h)`.
A gate `G = xW_g` is computed alongside. No positional encoding is applied to
any of them — no RoPE, no learned embedding, nothing.

**Window indexing.** Query `t` attends to window slots `w ∈ {0, …, W-1}`, where
slot `w` holds key position `t - (W-1) + w`. So `w = W-1` is the query's own
position and `w = 0` is the farthest key. Write the distance

```
δ(w) = (W-1) - w              δ(W-1) = 0 (self),  δ(0) = W-1 (edge)
```

**Decay profile.** `p : {0,…,W-1} → (0, 1]`, full strength inside the plateau
and a cosine ramp beyond it:

```
p(δ)  =  1                                     if δ < P
      =  ½·(1 + cos(π·(δ - P)/(W - P)))        if P ≤ δ ≤ W-1
```

With `P = None` the profile is identically 1. Note `p > 0` strictly — the ramp
approaches zero at `δ = W` but the window's farthest slot is `δ = W-1`, so the
profile never reaches it. **This is load-bearing:** it is what makes `log p`
finite everywhere, and §3.1 depends on it.

**Scores, masking, weights.**

```
s[t,w]  =  ⟨q_t , k_{t-(W-1)+w}⟩ / √d_h

valid(t,w)  ⟺  t - (W-1) + w  ≥  0

a[t,·]  =  softmax_w ( s[t,w] + log p(δ(w)) + (0 if valid else -∞) )
```

**Aggregation and output.**

```
o_t  =  Σ_w  a[t,w] · v_{t-(W-1)+w}                      per head

ô    =  o ⊙ γ / √(mean_{d_h}(o²) + ε)                    per-head RMSNorm
y    =  W_o ( ô ⊙ SiLU(G) )                              merge heads, gate, project
```

The per-head RMSNorm and the SiLU gate mirror the output side of Lumen's
Gated DeltaNet, so a block can hold either mixer without knowing which. That is
not decoration — §3.1 turns on it.

**Streaming state.** A ring buffer of the last `W-1` keys and values,
`(B, h, W-1, d_h)` each, plus a count of how many slots are filled. Constant in
generated length, which is the point of a fixed window.

---

## 2. What the two lineages contributed

| lineage | uniquely had |
|---|---|
| **Graded** | the two-phase decay profile; a dense `O(T²)` masked path usable as a correctness oracle; reverse-causal and bidirectional generalisations of the window; profile caching keyed on shape and device |
| **Fused** | the layer in the streaming-API shape; softmax; a Triton forward **and** a backward that reuses the forward kernels; the ring-buffer decode path; `-∞` on structurally-absent keys; zero-initialised output projection for identity splicing |

The fused lineage is the base. It is already softmax, already the right shape,
already streams. The graded lineage contributes the profile, the oracle, and the
directional variants that are deferred (§5).

Neither lineage had the whole thing, and the parts they disagreed about turned
out to be the interesting ones.

---

## 3. Decisions — settled

### 3.1 The ramp enters in log-space, and the placement question dissolves

The two lineages applied the same profile at different points. Graded multiplied
*before* normalising; fused multiplied *after* softmax without renormalising.
That looks like a substantive fork — one preserves total attention mass, the
other lets a query's output magnitude fade as its evidence ages — and it was
recorded as one. It is not.

Write `A` for the log-bias weights and `B` for the post-softmax weights:

```
A_w  =  exp(s_w)·p_w / Σ_u exp(s_u)·p_u
B_w  =  exp(s_w)·p_w / Σ_u exp(s_u)
```

**Same numerator; the denominators differ by a scalar that does not depend on
`w`.** So the aggregated outputs are parallel:

```
Σ_w A_w v_w  =  λ · Σ_w B_w v_w        λ = Σ_u exp(s_u) / Σ_u exp(s_u)·p_u  ≥ 1
```

`λ` is one positive number per (batch, head, position) — identical across every
channel of the vector. And the next operation is a per-head RMSNorm, which is
positively homogeneous of degree zero:

```
N(λo) = λo·γ / √(λ²·mean(o²) + ε)  =  o·γ / √(mean(o²) + ε/λ²)  =  N(o)|_{ε → ε/λ²}
```

The two forms produce **the same layer**, exactly in the `ε → 0` limit and to
within a shrunken epsilon otherwise. There is no experiment to run.

The exception is worth naming precisely, because it is where the argument stops
holding: when `mean(o²)` is itself comparable to `ε`, the epsilon is no longer
negligible and the forms separate. That is the deep-ramp regime — a query whose
entire window sits far out on the cosine tail — which is also the regime where
the post-softmax form drives total attention mass toward zero.

**The layer implements the log-space form, and offers no switch.** Three
reasons, in order of weight:

1. It shares one mechanism with the validity mask. Both the profile and the
   structural `-∞` are pre-softmax additive terms, so the layer has a single
   masking path instead of two that must be kept consistent.
2. It cannot produce the near-zero normaliser described above. Renormalisation
   is intrinsic to it.
3. It is the recognisable form (§4).

A knob whose only effect is to matter when an invariant is broken is not a knob,
it is a trap. What the record owes a reader instead is the **condition**: the
equivalence depends on the attention output passing through a per-head,
scale-invariant normalisation. A subclass that replaces or removes that
normalisation reopens the question, and should know it.

> This is what putting two implementations side by side is *for*. The
> disagreement was real, visible only in comparison, and dissolved under three
> lines of algebra — which is a better outcome than an experiment, and a cheaper
> one.

### 3.2 Absent keys are `-∞`, not zero padding

At `t < W-1` part of the window lies before the start of the sequence. The
obvious implementation pads `K` and `V` with zeros — but a zeroed key still
produces a score of `⟨q, 0⟩ = 0`, and `exp(0) = 1`, so **padding slots receive
real attention weight.** The layer quietly attends to nothing, in proportion to
how unremarkable the real keys look.

Those positions are structurally absent, not merely uninformative. Masking them
to `-∞` before the softmax makes the normalisation run over valid keys only, and
has a second benefit that is easy to miss: it makes the windowed path agree with
the dense oracle **at every position**, including the partial-window prefix.
Without it the prefix has to be excluded from the equivalence test, and an
excluded region is a place where a bug can live indefinitely.

### 3.3 The backward reuses the forward kernels

The windowed forward needs two kernels: a scores kernel
(`s[t,w] = ⟨q_t, k_{t-(W-1)+w}⟩`) and an aggregate kernel
(`o[t] = Σ_w a[t,w]·v_{t-(W-1)+w}`). Two of the four gradients are those same
kernels:

```
dQ[t]        =  Σ_w  g[t,w] · K[t-(W-1)+w]      — the aggregate kernel
dWeights[t,w] =  ⟨ g[t] , V[t-(W-1)+w] ⟩         — the scores kernel, scale 1
```

Only `dK` and `dV` have the other shape — each key accumulates gradient from the
`W` *future* queries that attended to it — and they stay in the reference
language.

This matters more than it looks. The backward is the larger term in training,
and an accelerated forward bolted to a slow backward is how a kernel ends up
losing end-to-end while winning every microbenchmark. Half of this backward is
covered by kernels that already had to exist.

### 3.4 fp32 reference, Triton opt-in and measured

The reference path is fp32 and depends on nothing but `torch`, because that is
the path that runs everywhere. Triton is imported under a guard; its absence is
a fallback, not an error.

**The backend is selected explicitly and never auto-detected.** This is the one
place where the obvious convenience is the wrong call. A fast path that switches
itself on whenever a package happens to be importable means two projects using
this layer are no longer running the same object — and then a difference in
their numbers stops being a difference in their experiment, which is the single
thing this library exists to prevent. `backend="reference"` is the default;
`backend="triton"` is a decision someone makes and can record.

Requesting a backend that cannot be built is an error at construction, not a
silent downgrade — degrading quietly is for capabilities nobody asked for. A
CUDA-only path meeting a CPU tensor is a different matter and falls back, since
that is a device gap rather than a missing capability.

**Measured, on one machine.** Tesla P40 (SM 6.1, Pascal, no tensor cores),
fp32, `d_model=512`, 8 heads, batch 4, via `lumen.bench`:

| window | T | fwd | **fwd+bwd** |
|---|---|---|---|
| 8 | 512 / 1024 / 2048 | 1.90× / 1.92× / 2.00× | **1.55× / 1.54× / 1.57×** |
| 32 | 512 / 1024 / 2048 | 3.54× / 3.73× / 4.05× | **2.23× / 2.31× / 2.37×** |

Two things in that table are the design showing through rather than luck.

The advantage **grows with the window**, because what the fused kernel removes
is `W` kernel launches and `W` passes over K or V — so a wider window has more
overhead to delete. And the **backward wins too**, which is the part that
usually decides these comparisons: an accelerated forward bolted to a slow
backward is how a kernel wins every microbenchmark and loses end to end. Half
this backward is the forward kernels (§3.3), so it comes along.

These are one machine's numbers and they are evidence, not specification. What
generalises is the mechanism: the win is bandwidth and launch overhead, not
arithmetic, which is why it does not depend on tensor cores — hardware this card
does not have.

### 3.5 No canonised window or plateau

`window` is a required argument. `plateau=None` — a hard window — is the
default, and is safe to default to precisely because it is the case where §3.1's
question does not arise at all: the profile is identically 1 and every placement
coincides.

Values for `W` and `P` that work are task-dependent and no controlled comparison
exists. Consolidating is not a licence to pick.

---

## 4. Relation to existing work

**No novelty is claimed for the mechanism.** Undertow is a name for a
configuration, not for a discovery, and the parts are individually standard.

- **Sliding-window causal attention** is established — Longformer, BigBird, and
  Mistral among others. The observation that stacking `L` such layers yields an
  effective receptive field of order `L·(W-1)` is Mistral's, stated explicitly.
- **Omitting positional encoding** in a decoder is likewise studied (NoPE); with
  a causal window, ordering information is available from the causal structure
  itself.
- **The graded boundary, in the log-space form of §3.1, belongs to the additive
  attention-bias family.** ALiBi is precisely a distance-dependent bias added to
  scores before the softmax; T5's relative position bias is the learned variant.
  Undertow differs only in the shape of the bias — a plateau followed by a
  cosine ramp to a hard cutoff, rather than an unbounded linear penalty.
- **StreamingLLM** is the closest relative and the comparison is worth drawing,
  because the difference is structural rather than cosmetic. StreamingLLM is an
  *inference-time* method applied to a model trained with full attention, and
  its attention sinks exist to absorb mass that such a model learned to place
  somewhere. A model trained windowed from the start never develops that
  expectation and has nothing to preserve. The distinction is **retrofit versus
  native**, not the presence or absence of a sink.
- **The post-softmax placement** discussed in §3.1 is the one piece that looks
  unusual — softmax attention almost always leaves rows as convex combinations,
  and that form does not. §3.1's answer is that under per-head output
  normalisation it is not a different layer at all.

The attributions above are to well-established work, and they are the honest
account of where each piece comes from. What this record does **not** claim is
exhaustive coverage of the literature: if the particular combination —
natively-trained windowed attention, no positional encoding, and a graded
boundary — has been published under some name, we have not found it, and would
rather be told than keep guessing. *Undertow* is what this implementation is
called. It is not a claim to be first.

---

## 5. What the consolidated layer is

Shape and streaming from the fused lineage, profile and oracle from the graded
lineage, one masking mechanism throughout:

- frozen `UndertowConfig`, validated on construction (`0 ≤ plateau < window`,
  `window ≥ 1`, `d % n_head == 0`)
- `init_state()` / `step()` / `forward(..., state=, return_state=)` — the house
  interface, so a block can hold this or Gated DeltaNet interchangeably
- causal look-back window; `window > seq_len` clamps rather than raising
- log-space profile, single pre-softmax masking path (§3.1, §3.2)
- per-head RMSNorm + SiLU gate on the output, matching the sibling mixer
- zero-initialisable output projection, so the layer can be spliced into a
  trained stack as an exact identity and earn its contribution from zero
- `backend` in the config: fp32 reference by default, Triton opt-in behind the
  same interface, measured (§3.4)
- a dense `O(T²)` oracle retained in the test suite, not the public surface

Deliberately **out** of v1:

| excluded | why |
|---|---|
| reverse-causal window (look-forward, bounded by `W`) | it has one prospective consumer and that architecture is unbuilt |
| bidirectional / centered window | no consumer; the accelerated path never covered it |
| pluggable non-softmax kernels | the graded lineage's periodic kernels are a component of their own, with their own numerics to preserve |
| per-head learnable ramp | an ALiBi-shaped generalisation of a fixed profile; untested, and not something to default into |
| loss masking over the context prefix | a *training* convention, not a layer detail — it belongs in the usage documentation, and the layer has never heard of logits |

---

## 6. Acceptance

Before any project switches to this layer:

1. **Windowed path == dense oracle**, `1e-6`, fp32, at **all** positions
   including the partial-window prefix (§3.2), on CPU and on CUDA.
2. **`step()` == `forward()`**, `1e-6`, including the first `W-1` positions
   where the ring buffer is partly unfilled and the two paths mask the absent
   region by different mechanisms — a filled-slot count on one side, positional
   validity on the other. Agreement there is a real test, not a formality.
3. **Peak memory flat** across a 4× sweep of generated length.
4. **The §3.1 equivalence pinned by a test** — a post-softmax reference must
   match the shipped log-space path after the head norm. The design record
   claims an invariance; the suite should hold it to that claim.
5. Any accelerated path matches the reference to `1e-6` and beats it end-to-end,
   measured here (§3.4).

Merge → verify → switch → evolve, as four steps. Not one.
