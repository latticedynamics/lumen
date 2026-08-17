# Block and Stack

A residual block, and a stack of them with a final norm. `(B, T, d_model)` in,
`(B, T, d_model)` out, and nothing above that line — no embedding, no LM head,
no task head.

This is the level a result is actually read off. A number is never read off a
mixer; it is read off a stack, and the stack is where norm placement, the
residual arrangement, the depth-scaled initialisation and the final norm all
live. Sharing a mixer exactly and then re-typing those per project means two
runs can differ for reasons that have nothing to do with the layer that was
shared.

The design rationale — why the block takes instances rather than a config, why
two arguments have no defaults, what is deliberately left undecided — is in
[the design record](design/BLOCK.md). This page is how to use it.

```python
from lumen import Block, Stack, GatedDeltaNet, GatedDeltaNetConfig, HeadLayout

def layer(index: int) -> Block:
    mixer = GatedDeltaNet(GatedDeltaNetConfig(d_model=512, layout=HeadLayout.shared_key(8)))
    return Block(512, mixer, norm_eps=1e-5, d_mlp=0)

trunk = Stack(512, n_layers=8, block=layer, norm_eps=1e-5)
y = trunk(x)                                   # (B, T, 512) -> (B, T, 512)
```

---

## What a block is

Up to three pre-norm sub-layers, in order, each with its own residual:

```
x = x + local(norm_local(x))     # only if `local` was given
x = x + mixer(norm(x))
x = x + mlp(norm2(x))            # only if `d_mlp > 0`
```

`local` is a slot for a sub-layer placed *before* the mixer. It holds anything
with the same shape contract; the name records the arrangement that motivated it
— one sub-layer resolving recent positions exactly, the other carrying the whole
prefix through a fixed-size state, with local detail spent before the state
rather than after.

## Two arguments have no defaults, on purpose

```python
Block(d_model, mixer, *, norm_eps, d_mlp, local=None)
```

**`norm_eps`** — the epsilon for this block's residual-stream norms. There is no
canonical value: the mixers' own per-head norms disagree between lineages, and
the residual-stream norms here are a second, independent knob. Neither has a
default anywhere in this library. See [issue #6][i6] for the arithmetic of when
the choice starts to matter.

**`d_mlp`** — the hidden width of a gated feed-forward sub-layer, or `0` for
none. Not a default because it is not a convention: a mixer's own output path is
already an up/gate/down arrangement, so a block that adds a second one is making
a claim about what the mixer is short of.

[i6]: https://github.com/latticedynamics/lumen/issues/6

## Blocks come from a factory

`Stack` calls `block(index)` once per layer, so heterogeneity is a lambda rather
than a feature:

```python
from lumen import SwiGLU

def layer(index: int) -> Block:
    # layer 3 trades its state for a plain feed-forward
    mixer = SwiGLU(512, 1024) if index == 3 else GatedDeltaNet(config)
    return Block(512, mixer, norm_eps=1e-5, d_mlp=0)
```

A sub-layer that holds no state needs no support from `Block` — statefulness is
read off the object rather than declared, so a stateless module in the mixer
slot simply has a `None` in its state.

A factory rather than a pre-built sequence because `n_layers` has to exist
*before* any block does: the depth-scaled initialisation needs it, and under a
sequence the caller would have to know it too.

---

## Streaming

The same contract the mixers hold — `forward(x, *, state=, return_state=)`,
`init_state(batch)`, `step(x_t, state)` — at both levels, threading each block's
state:

```python
prefix, state = trunk(x[:, :n], return_state=True)   # prefill in one pass
y, state = trunk.step(x[:, n:n+1], state)            # then one position at a time
```

`init_state(batch)` takes a real batch size, and that is not only a decode
convenience: stepping many independent streams in lockstep is a first-class path.

```python
state = trunk.init_state(256)                        # 256 independent streams
y, state = trunk.step(x, state)                      # x is (256, 1, d_model)
```

`BlockState` and `StackState` are frozen dataclasses and `step` returns a
*successor* rather than mutating. Forking a stream cannot leave two branches
sharing a buffer, which matters wherever a prefix is continued down more than one
future.

---

## Initialisation

`Stack` initialises itself completely in `__init__` and exposes
`reset_parameters()`. Two passes: every `nn.Linear` weight at `init_std`, then
the projections that write to the residual stream rescaled by
`1/sqrt(2·n_layers)`.

It finds those projections by **asking**:

```python
mixer.residual_out_projections()     # -> (o_proj,)
```

rather than by matching parameter names. A caller sweeping `named_parameters()`
for a suffix works until an attribute is renamed, at which point it initialises
one tensor fewer and produces a model that trains and is wrong. A custom
sub-layer must implement this method or `Block` will refuse to guess.

Three consequences worth knowing before you build one:

- **The stack draws its own weights and nothing else's.** Your embedding and
  head are yours to initialise.
- **Construct the trunk first.** It then occupies a stable prefix of the RNG
  draw sequence whatever is built around it, which is what makes "rebuild this
  trunk from its seed" well defined. This cannot be enforced, only stated.
- **Biases are never touched**, so a sub-layer that deliberately sets one keeps
  it.

**The depth pass runs last and therefore wins**, including over a sub-layer that
zero-initialised its own output projection — `UndertowConfig(zero_init=True)`
being the case that exists. Splicing a zero-initialised layer into an
*already-trained* trunk does not construct a fresh `Stack` and never meets this;
building a fresh stack out of such layers does. The obvious alternative — skip
projections that are currently all-zero — is refused, because it would make
initialisation depend on parameter *values* rather than only on the seed.

---

## Checkpoint keys

```
blocks.{i}.norm.weight          blocks.{i}.mixer.*
blocks.{i}.norm_local.weight    blocks.{i}.local.*
blocks.{i}.norm2.weight         blocks.{i}.mlp.{up,gate,down_proj}.weight
norm_f.weight
```

These are inherited verbatim from the implementation this was extracted from,
including the inconsistency among the norm names. That is deliberate: the keys
are the interface, a transplant between bodies is a `state_dict` operation, and
that operation does not degrade gracefully under a naming difference — it either
raises or partially matches and quietly loads a subset. Every tidying is a
permanent remap entry against archived weights, bought with nothing measurable.

The practical consequence is that **a stack built here loads weights trained by
an implementation that never imported this library**, provided that
implementation used the same names. Two projects can interoperate at the
checkpoint level without either adopting the other's classes.

---

## Taking one and not the other

`Block` is usable without `Stack`, the mixers are usable without either, and
declining all of it costs nothing. `Stack` is a convenience for consumers that
want the higher-level object; its one irreducible job is the depth-scaled
initialisation, which cannot live lower down because the scale depends on how
many blocks write to the stream.

A project whose purpose is experimenting *at* block level is expected to keep its
own block. What it should not change are the keys.
