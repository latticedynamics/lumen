# Block and Stack — design record

**Status:** shipped in 0.4. The implementation is expected to match this record;
where the two disagree, one of them is a bug.

Lumen holds mixers and instruments. It does not hold the thing every measured
number is actually read off. A result is not read off a `GatedDeltaNet` — it is
read off a stack of residual blocks, and the block is where norm placement, the
residual arrangement, the depth-scaled init and the final norm all live. Every
one of those sits between the shared mixer and every measured quantity, and
every one is currently re-typed per consumer.

So the guarantee the README advertises —

> when the layer is *the same object*, a difference in the numbers is a
> difference in the experiment

— currently stops one level below where the numbers are made.

---

## 1. What the component is

Two objects, modality-free.

**`Block`** — one or more pre-norm residual sub-layers over a `(B, T, d_model)`
stream:

```
x  ←  x + Mixer(norm(x))
x  ←  x + MLP(norm2(x))        # only if the block was given an MLP
```

**`Stack`** — `n_layers` blocks and a final norm. `(B, T, d_model)` in,
`(B, T, d_model)` out. Nothing above it: no embedding, no LM head, no task head.

Both hold the house streaming contract the mixers already hold —
`forward(x, *, state=, return_state=)`, `init_state(batch)`, `step(x_t, state)`
— threading each block's state.

The residual arrangement is pre-norm with the residual added outside the norm.
That is not in question: both lineages do it and no comparison is being invited
here. What is in question is everything the arrangement is *parameterised by*,
which is §5 and §6.

---

## 2. What the two consumers contributed

Only one of them has code. That is unusual for this repository — the Gated
DeltaNet record had four implementations to reconcile — and it changes what this
record is for. There is nothing to un-drift yet. **The value here is that the
drift has not happened, and the extraction is cheapest now.**

### 2.1 The incumbent — the only implementation

What it uniquely contributed:

- **The two-sub-layer ordering, with an argument.** With Undertow the block is
  `x + Undertow(norm_local(x))` then `x + GDN(norm(x))`, and the ordering is a
  claim: Undertow resolves the last `ut_window` positions exactly and cannot see
  past them; the mixer sees the whole prefix but only through
  `rank(S) ≤ min(d_k, d_v)` slots. *Local detail is cheap to get right and
  expensive to store, so it is spent before the state rather than after.*
- **"The block owns the state construction."** `Block.init_state`'s docstring:
  Lumen's layers require a real state on `step` — `None` is a missing argument,
  not a zero-state — so the block builds it "rather than making every caller
  know its sub-layer count." That argument is correct and it generalises one
  level up, which §3.2 takes.
- **State-shape hiding.** A block returns `(s_local, s_mix)` when it has a local
  sub-layer and bare `s_mix` when it does not, so "a caller threading states
  never has to know the block's shape."
- **The finding that "block = mixer + MLP" is a claim, not a convention.** The
  module docstring used to say "MLP-free" and that was never true: the mixer's
  own output path *is* a SwiGLU — `v_proj` up, `g_proj` gate, `o_proj` down —
  which a layer reaches by driving `α` to zero and giving up its memory. The two
  functions compete for one piece of hardware. §5.2 is downstream of this.

### 2.2 The second consumer — requirements, no code

Deliberately unnamed: what it needs from a block is design input, and what it is
doing with the block is not this record's business. Stated as requirements only.

It contributes *requirements the LM consumer never generates*, and they are the
ones that decide three of the four open questions in the issue:

- **The batch dimension carries many independent streams, stepped in lockstep.**
  `init_state(N)` at N in the hundreds, then one `step` on `(N, 1, d_model)`, is
  the **primary inference path** rather than a decode convenience. Nothing about
  the LM consumer exercises this: its own streaming path is `init_state(1)`.
- **State immutability is a correctness requirement, not tidiness.** Streams get
  forked and continued independently, so a successor that shares a buffer with
  its predecessor is a silent bug rather than a style complaint. The mixers
  already return frozen successors. The *stack's* container has to as well —
  §3.3.
- **Weights are reconstructed from a stored seed, not loaded from a
  checkpoint.** So "construct the stack under this seed" must be
  bit-reproducible, which promotes the init policy from a training detail to
  part of the shared surface — §4.2.
- **The modality line is an operation, not an aesthetic.** The trunk is
  transplanted between different input and output adapters, in both directions,
  and existing checkpoints are the donors. "Stop at the modality line" is what
  makes that a `load` rather than a rewrite.
- **A cost model the other consumer does not have.** CPU, batch-1,
  dispatch-bound — tens of tiny torch ops where dispatch outweighs the
  arithmetic. Per-block overhead that is invisible under `B=8, T=2048` on a P40
  is not invisible here.

### 2.3 What the incumbent's two extractability blockers were

The consolidation had a recorded reason it was not simply done already: the
incumbent block was not extractable as-is, because it carried its own
experiment hooks and a per-layer substitution mechanism, neither of which is
general.

**Both blockers are now discharged, which was not true when that was written.**
The hooks were retired on their own merits, and the substitution mechanism
dissolves into §4.1's parameterisation rather than needing block support at all. Nothing stands between the incumbent block
and extraction any more.

---

## 3. Settled by evidence

### 3.1 The block calls the mixer in the house form, and the adapter disappears

An open question stood against the house API before this component existed:
two sub-layers in one consumer carried the same three-line adapter, turning
`forward(x, *, state=, return_state=)` into a positional `(y, state)`. The house
API meeting the most natural block-level polymorphism and losing, twice, in one
file.

It is worth being precise about what that evidence shows, because the natural
reading is wrong. The adapters do not exist because the house API is bad. **They
exist because the block is outside the library.** A Lumen `Block` calls
`mixer(x, state=s, return_state=True)` itself, once, and both adapters have
nothing left to convert.

So shipping the block **closes** that question rather than acting on it. The house API is
not changed, and the thing that made it look expensive is absorbed by the object
that was missing. Answers the issue's open question 1.

### 3.2 The stack owns `init_state` and `step`, composing them from blocks

The lineage had no stack-level streaming API. Its text-generation path
open-coded one:

```python
states = [blk.init_state(1) for blk in self.blocks]
...
x, states[i] = blk.step(x, states[i])
```

A list comprehension, a hard-coded batch of 1, and a mutating index assignment,
inside a decode helper — which is where a second consumer will not find it. The second consumer's primary inference path is `init_state(N)` at N in
the hundreds and a `step` over the whole stack, so it would re-type exactly
this, differently.

The lineage's own argument for the block owning the construction — *"rather than
making every caller know its sub-layer count"* — applies verbatim to the stack
and its block count. Take it one level up. Answers the issue's open question 3.

### 3.3 The stack's state is a frozen container, not a list

The lineage's model accumulated `new_states = []` and returned the bare list.
Under a single training loop that is harmless. Under the second consumer it is a live
failure: forking a stream must not leave two branches sharing anything mutable.

The mixers already got this right — `GatedDeltaNetState` is a frozen dataclass
and `step` returns a successor rather than mutating. The stack inherits the same
discipline: a frozen `StackState` holding a tuple of block states. This is also
what makes the state a typed object rather than "a list of whatever the blocks
returned", which matters as soon as a block has two stateful sub-layers.

Answers the issue's open question 1's second half — the state-threading
convention belongs to the block for its own sub-layers, and to the stack for its
blocks, and in neither case to the caller.

### 3.4 Parameter names are the interface, and the incumbent's names win

`nn.py` already documents the hazard one level down, for `head_norm`. It is
sharper here, because the transplant the second consumer performs *is* a
`state_dict` operation: it does not degrade under a naming difference, it either raises or
partially matches and quietly loads a subset.

The incumbent's keys — `blocks.{i}.norm.weight`, `blocks.{i}.norm_local.weight`,
`blocks.{i}.norm2.weight`, `blocks.{i}.mixer.*`, `blocks.{i}.local.*`,
`blocks.{i}.mlp.{up,gate,down_proj}.weight`, `norm_f.weight` — are adopted
verbatim unless there is a stated reason to move a specific one. Archived
checkpoints exist and are transplant donors; any other choice buys a legacy
remap entry that is permanent.

Note this constrains the *sub-layer attribute names*, which means it partly
constrains §4.1: whatever the block is parameterised by, the mixer has to land
on `self.mixer`.

### 3.5 No experiment hooks, and the hot path is why

Separability is the first reason: an instrument does not get merged into the
layer it inspects. The second consumer supplies an independent one. The
incumbent block tested a training-only hook once per block per call; at
`B=8, T=2048` on a GPU that is free, and at batch-1 CPU dispatch over hundreds
of streams × `n_layers` blocks × every step it is not — while guarding a
feature that had already been retired.

### 3.6 `Block` and `Stack` are independently adoptable, and declining both costs nothing

A correction to an assumption this record carried without stating it.

`Stack` is a **convenience** for consumers that want the higher-level object. A
consumer whose purpose is experimenting *at* block level may reasonably never
want either class — and the record had been reasoning as though the block's
value is realised only when a consumer adopts it. That is one route and not the
only one.

**What actually delivers the guarantee is the checkpoint keys, not the class.**
§3.4 adopts the lineage's keys verbatim. The consequence is that a stack built
here loads weights trained by an implementation that never imported any of this
— so two consumers interoperate at the artefact level whether or not either
switches. A consumer whose whole purpose is experimenting *at* block level is
expected to keep its own block; what it must not do is change the keys.

Three consequences that are now design constraints rather than observations:

- **The mixers stay usable standalone**, exactly as they are today. Nothing in
  this record may make a mixer harder to hold outside a `Block`.
- **`Block` is adoptable without `Stack`.** They are separate objects with
  separate reasons, and the depth-scaled init — the one thing that genuinely
  needs the stack — is the only coupling.
- **The un-adopted case has a cost, and it should be named rather than
  wished away.** Two blocks that agree today are not thereby kept in agreement.
  The oracle (§7 gate 1) proves agreement at one commit; nothing preserves it.
  What must be preserved is narrower and testable: **a checkpoint produced by
  one loads into the other and computes the same thing.** That is the invariant
  to gate on, and it survives either consumer changing its own block for its own
  experiment — which is what an experiment repository is *for*.

---

## 4. Settled by decision

### 4.1 Mixer *instance*, not mixer config

The issue asks it as a clean trade: instance keeps the stack agnostic, config
keeps construction declarative and serialisable. The lineage breaks the tie.

`Block.__init__` currently takes `cfg` and an `index`, and branches on
`index in cfg.swiglu_layer_set()` to decide whether the layer gets a SwiGLU
where its mixer would be. So the config-parameterised block already needs to
know about layer indices and a set of them, in order to express heterogeneity
across the stack — and heterogeneity is live work, not hypothetical, because
that mechanism is how a consumer acts on what it has measured about which
layers are actually using their state.

Under instance parameterisation this is not a feature. The stack builds whatever
it likes for index `i` and the block holds it. **The substitution stops being an
experiment hook and becomes a thing the caller can already do**, which
simultaneously discharges the second of §2.3's two extractability blockers.

The cost is real and should be stated: a stack built from instances is not
reconstructible from a config dataclass alone, so a consumer that wants
declarative construction writes a small builder. The incumbent has one already;
the second consumer reconstructs from a seed rather than from a config, so it
does not want one.

Recommendation: **instance**, with the stack taking a `Callable[[int], Block]`
factory so that "block `i` gets a SwiGLU" is a two-line lambda rather than a
config field. Answers the issue's open question 4.

**Factory rather than a pre-built `Sequence[Block]`** — taken 2026-08-15. A
sequence is simpler and makes heterogeneity equally free, with `n_layers`
falling out as `len(blocks)`. The factory wins on one thing: `n_layers` exists
*before* any block does. §4.2's depth-scaled init needs it, and under a sequence
the stack can only learn it from a list the caller has already fully built —
which means a caller that wants depth-scaled init has to know `n_layers` too,
and now two places hold it. The factory keeps `n_layers` in exactly one place
and hands it to construction, which is the only ordering under which the init
policy can belong entirely to the stack.

### 4.2 The stack owns the depth-scaled init — and must not do it by name

The issue asks whether depth scaling is a stack property or a consumer policy.
The second consumer decides it: reconstruction-from-seed means `Stack(...)`
under a given seed must reproduce weights bit-for-bit, so the init is part of the shared
object or the reconstruction contract lives outside the shared object.

**But the mechanism is the actual problem, and the issue does not reach it.**
The incumbent does this:

```python
for name, p in self.named_parameters():
    if name.endswith(("o_proj.weight", "down_proj.weight")):
        nn.init.normal_(p, std=0.02 / (2 * cfg.n_layers) ** 0.5)
```

That is a string match reaching *inside* the mixer for `o_proj.weight`. Two
things follow. The stack becomes coupled to the mixer's internal parameter
names — the exact fragility class Phase 5a named, now pointing the other way.
And it fails silently: rename `o_proj` inside Lumen and the stack keeps
constructing, initialising one fewer tensor, producing a model that trains and
is wrong in a way no test catches.

Recommendation: **a sub-layer declares its own write onto the residual stream**,
and the stack asks instead of grepping.

```python
def residual_out_projections(self) -> Iterable[nn.Module]: ...
```

`(self.o_proj,)` for both current mixers, `(self.down_proj,)` for the SwiGLU.

Three properties that make this worth the added surface:

- It is **additive**, like `RMSNorm` was — nothing breaks the day it lands and
  consumers adopt on their own schedule.
- It makes the policy **checkable**: a test asserts the returned set matches what
  the name-grep finds today, which is exactly the bridge that lets the switch be
  verified rather than believed.
- It fails **loudly**: a sub-layer that writes to the stream and returns nothing
  is an assertion, not a quietly skipped tensor.

##### Why it returns projections rather than parameters, which is not cosmetic

The first draft of this section proposed
`residual_out_parameters() -> Iterable[nn.Parameter]`, and the objection to that
version is real enough to have changed it.

A method returning *the parameters to be scaled* is an **init-policy concept
wearing a layer-interface costume**. The mixer's surface is otherwise four
members, all about computing outputs; this one would exist purely so somebody
else can initialise it. That does not remove the stack→mixer coupling, it
relocates it and blesses it as interface. Worse, it quietly canonises the
*shape* of the policy: "here are tensors to multiply at init time" assumes
depth scaling is a per-tensor init-time operation, which several schemes — μP
among them — are not. `CLAUDE.md` prohibits canonising an untested default;
canonising the untested *form* of a policy is the same error and harder to spot.

And "which parameters should be scaled" has as many answers as there are
schemes, so two consumers would eventually disagree **through the shared
surface**, which is strictly worse than disagreeing in their own code.

Naming the structure instead fixes all three. *Which projection writes to the
residual stream* is a fact about the layer's shape that is true whether or not
anybody ever initialises anything. The depth-scaled init becomes one caller of a
structural fact rather than the reason the method exists, an instrument is
another caller, and a different init scheme is a third — none of them canonised.

**Why a method and not simply a documented attribute name.** Both mixers already
have `self.o_proj`, so "the contract is that the write is called `o_proj`" is
tempting and costs nothing. It breaks on Lumen's own reuse story: reuse is by
subclassing and `_out` is a *named seam*, so a subclass may legitimately replace
the output path — and under an attribute contract it silently keeps an `o_proj`
that nothing calls any more. A method is overridable at the same seam that owns
the thing it describes.

Note this is an addition to `GatedDeltaNet` and `UndertowAttention`, so it is a
Lumen-side change that ships with the block rather than a block-only one.

**The residual objection, which is not answered.** A test asserting the sweep
touched exactly the expected number of tensors catches the silent-rename failure
too, in the stack, with no new public surface at all. The cheap fix and this one
prevent the same bug. What the method buys over the test is that it survives a
consumer Lumen has not seen — the test protects Lumen's stack, the method
protects anyone's. That is the trade, stated rather than hidden.

### 4.3 The pre-mixer slot keeps the name `local`, everywhere

Taken 2026-08-15. The slot is generic — any sub-layer with its own norm and
residual can sit there — and the incumbent checkpoint key for it is
`blocks.{i}.local.*`, which is an Undertow-shaped name. Three ways out: keep
`local` throughout; keep the key but call the constructor argument `pre_mixer`;
or rename everywhere and remap.

**Keep `local`, and use it for the attribute, the argument and the key alike.**

The middle option is the one to reject outright, and it was the version in §6's
first sketch. A constructor argument named `pre_mixer` that produces a state
dict key named `local` is not a compromise, it is the trap both other options
avoid — the next reader has to learn a mapping that exists for no reason.

Between the two consistent options:

- **§3.4 already decided this class of question**, two sections earlier: adopt
  the incumbent's keys unless there is a stated reason to move a *specific* one.
  "The name is slightly Undertow-shaped" is an aesthetic reason. Overriding a
  rule on aesthetics in the first case it applies to is how the rule stops
  meaning anything — and the rule is protecting a set of transplant-donor
checkpoints.
- **`local` names a role; `pre_mixer` names a position.** The slot exists
  because of §2.1's ordering argument — one sub-layer resolves recent positions
  exactly, the other carries everything through a fixed state, and local detail
  is spent before the state. "Local" is the correct word for the first of those.
  `pre_mixer` is correct only for as long as the ordering is never revisited,
  which is precisely the thing §8 declines to settle.
- The generic-slot worry is **speculative**. Nothing today puts a non-local
  sub-layer there. If something does, that is a new arrangement wanting its own
  note, and renaming is available then *with a reason* rather than pre-emptively.

**What is inherited along with it, deliberately.** The norm names are an
inconsistent scheme: `norm_local` is role-based, `norm2` is positional, and
plain `norm` is unqualified and belongs to the mixer. That is genuinely worse
than `local` is, and it is kept for the same reason — every fix is a permanent
remap entry on the archived runs, bought with nothing measurable. The docstring
carries the mapping instead, which is where it should have been in the first
place.

### 4.4 The stack initialises itself, and the trunk is constructed first

Taken 2026-08-15.

`Stack.__init__` performs both passes — the base `normal_(std=0.02)` over its
own `nn.Linear` modules and §4.2's depth-scaled sweep over the residual writes —
and exposes `reset_parameters()` so the whole thing is re-callable, which is
torch's convention and not an invention here.

**It draws its own weights and nothing else's.** A consumer's embedding and head
are the consumer's to initialise, which is what "stop at the modality line"
already demanded of every other decision in this record. The incumbent's
`self.apply(self._init)` walks the entire model including the embedding, and
that is the one thing about it that does not survive extraction.

**One ordering rule for consumers: construct the trunk first.** Then the trunk
occupies a stable *prefix* of the RNG draw sequence no matter what is built
around it, and "reconstruct this model from its seed" is well-defined without
`Stack` having to accept a generator. This is a documented rule rather than an
enforced one — there is no way to enforce it — so it is stated in the class
docstring and checked by acceptance gate 4 rather than asserted at runtime.

---

## 5. What is deliberately not canonised

`CLAUDE.md`: *do not canonise an untested default.* Three live disagreements,
and the issue found two of them.

### 5.1 `norm_eps` — which is two questions, not one

`RMSNorm`'s docstring records that the lineages disagree, `1e-5` against `1e-6`,
with no comparison run. That is true, and it is about the **head** norms inside
the two mixers.

The block introduces a **second, independent** eps: the one on the residual
stream, at `norm`, `norm_local`, `norm2` and `norm_f`. The incumbent passed a
single epsilon to all four. Nothing says the block-norm eps and the head-norm
eps should be the same number, and nothing has tested whether either matters.

So the block takes `norm_eps` as a required argument with no default, the same
way `rms_norm` does, and the record states that there are two knobs here rather
than one. Two experiments go on the list, not one.

### 5.2 The MLP sub-layer, which is not a default

Per §2.1: the mixer's output path is already an up/gate/down arrangement, so
"block = mixer + MLP" is a claim about what the layer is short of, not a neutral
convention. A layer can reach that latent gated MLP by driving its decay toward
zero and giving up its memory, so the two functions compete for one piece of
hardware; adding a second MLP alongside changes what that trade costs.

`d_mlp` has no default. `d_mlp=0` (no MLP sub-layer) and `d_mlp>0` are two
models, and the caller says which. Same shape as `expand_k`.

### 5.3 The residual-write count in the depth scaling — the `2`

Not in the issue, and it is the same class as the other two.

`std=0.02 / (2 * n_layers) ** 0.5`. The incumbent's comment calls the 2 the
GPT-2 "two residual writes per block" factor, *"which with `--d_mlp` is finally
literal rather than generous."* That comment is exactly right and it is also the
problem: **the factor is only literal in some of the block's configurations.**

| block composition | residual writes |
|---|---|
| mixer only | 1 |
| mixer + MLP | 2 |
| Undertow + mixer | 2 |
| Undertow + mixer + MLP | 3 |

A stack parameterised by instances knows its own block composition and *could*
count the writes instead of assuming two. It should not, silently: that changes
numbers against every archived run, and no comparison has been made. So the
constant stays `2`, the record says why it is only sometimes correct, and
"count the writes" goes on the experiment list with the other two.

---

## 6. The surface

Sketch, not signature-final. `norm_eps` and `d_mlp` required; no defaults where
§5 says there are none.

```python
class Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        mixer: nn.Module,
        *,
        norm_eps: float,
        d_mlp: int,
        local: nn.Module | None = None,       # own norm and residual, before
                                              # the mixer.  §4.3 on the name.
    ) -> None: ...

    def forward(
        self, x: Tensor, *, state: BlockState | None = None,
        return_state: bool = False,
    ) -> Tensor | tuple[Tensor, BlockState]: ...

    def init_state(self, batch: int, device=None, dtype=None) -> BlockState: ...
    def step(self, x: Tensor, state: BlockState) -> tuple[Tensor, BlockState]: ...


class Stack(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        block: Callable[[int], Block],
        *,
        norm_eps: float,
    ) -> None: ...
    # same four members; StackState holds tuple[BlockState, ...]
```

`BlockState` and `StackState` are frozen dataclasses (§3.3). Device and dtype on
`init_state` follow the module's own parameters unless overridden, matching the
mixers.

---

## 7. Acceptance

The consolidation rule is merge → verify → switch → evolve, and the verify step
here is cheap because there is exactly one incumbent.

1. **Bit-exactness against the incumbent.** Weights transferred, same input,
   `max|Δ| == 0` on the stack output — for all four block compositions in §5.3's
   table. This is the check that has to happen *before* any consumer switches.
2. **`state_dict` keys survive a `strict=True` load** of an archived run's
   weights, through whatever legacy remap the consumer carries. §3.4 is a claim about
   keys; this is the claim being checked rather than intended.
3. **Population-as-batch.** `stack.init_state(256)` and a `step` on
   `(256, 1, d_model)` produce the same trajectory as 256 independent
   `init_state(1)` streams. This is the second consumer's primary path and
   nothing in the LM consumer exercises it.
4. **Init determinism.** Two `Stack`s built under the same torch seed are
   bit-identical, including the depth-scaled sweep, and the count of RNG draws
   does not depend on the block composition beyond what actually has parameters.
   (`gdn/layer.py` already reasons about draw counts for `centre` — same
   discipline.)
5. **The residual-write set is right.** `residual_out_parameters()` returns
   exactly what the incumbent's name-grep finds, for every mixer and the SwiGLU.
6. **Streaming agrees with parallel.** `forward(x, return_state=True)` on a
   prefix, then `step` through the remainder, equals `forward` on the whole —
   at the stack level, which nothing currently tests.

---

## 8. What this record does not decide

- Anything above the modality line. Both consumers' adapters are their own.
- Whether the two-sub-layer ordering (Undertow before the mixer) is *right*. It
  is inherited with its argument and no comparison is invited here.
- Whether what a consumer has measured about its own stacks survives a different
  training objective. That is the consumer's experiment, and the instrument for
  it is the consumer's too.

One consequence of that last item constrains the block. Instruments that measure
a mixer's internal behaviour reach past its public surface — into the feature
projections, typically — and they break quietly rather than loudly when what
they reach for moves. **`Block` must therefore expose its mixer as a plain
attribute** (`self.mixer`, per §3.4) rather than wrapping or hiding it. That
costs nothing, and it is the difference between an instrument that keeps working
and one that reports a wrong number without saying so.

---
