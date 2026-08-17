# States as pytrees

Every streaming state in this package is a registered pytree node, so the
`torch` transforms that have to flatten a structure can walk one.

`GatedDeltaNetState`, `UndertowState`, `BlockState` and `StackState` are frozen
dataclasses of tensors. Until they are declared, `torch` sees each of them as a
single opaque leaf, and a leaf that is not a tensor is refused:

```
ValueError: Got in_dim=0 for an input but the input is of type
<class 'lumen.stack.StackState'>. We cannot vmap over non-Tensor arguments
```

Registration happens when you `import lumen`. There is nothing to call and no
opt-in — but the call is exported for anyone who wants to be explicit, and it is
idempotent.

For *why* it is shaped this way — why import time, why one of the four cannot use
the public entry point, and what `UndertowState.seen` has to do with any of it —
see the module docstring in `lumen/pytree.py` and
[`latticedynamics/lumen#12`](https://github.com/latticedynamics/lumen/issues/12).

---

## What this buys

`tree_map` and friends over a state:

```python
import torch.utils._pytree as pytree
from lumen import Stack

state = trunk.init_state(batch=8)
state = pytree.tree_map(lambda t: t.to("cuda"), state)     # move a whole stream
```

`torch.compile` across a boundary that a state crosses, and `functional_call`
with a state in the keyword arguments.

And the case it exists for: **many distinct parameter sets through one module
definition, in a single batched call.**

## Many parameter sets at once

`stack_module_state` plus `vmap(functional_call(...))` is the sanctioned way to
evaluate `G` different weight draws without writing a second, batched
implementation of the trunk — a population-based search being the consumer that
needs this, since there the weights are the thing being searched.

```python
import torch
import torch.utils._pytree as pytree
from torch.func import functional_call, stack_module_state, vmap

from lumen import Block, GatedDeltaNet, Stack

SETS, BATCH, SEQ, D = 16, 4, 128, 256

def layer(index: int) -> Block:
    return Block(D, GatedDeltaNet(config), norm_eps=1e-5, d_mlp=4 * D)

models = [Stack(D, 8, layer, norm_eps=1e-5) for _ in range(SETS)]
params, buffers = stack_module_state(models)

# A definition to call, carrying no storage of its own.
base = Stack(D, 8, layer, norm_eps=1e-5).to("meta")

def one_set(parameters, buffered, x, state):
    return functional_call(
        base, (parameters, buffered), (x,), {"state": state, "return_state": True}
    )

x = torch.randn(SETS, BATCH, SEQ, D)

# One state per parameter set. init_state gives the per-set shape; the leading
# axis is what vmap maps over, so it has to be added to every leaf -- which is
# the traversal that needs the registration in the first place.
state = pytree.tree_map(
    lambda leaf: leaf.unsqueeze(0).expand(SETS, *leaf.shape).contiguous(),
    models[0].init_state(BATCH),
)

y, state = vmap(one_set)(params, buffers, x, state)
# y     (SETS, BATCH, SEQ, D)
# state carries both axes, and feeds straight back in for the next chunk
```

The returned state is a real state, not merely a surviving shape: pass it back
as `state=` on the next chunk and the stream continues, per parameter set and
per stream independently.

**The base module holds no storage.** `.to("meta")` is what makes it a
definition rather than a seventeenth model — all the weights come from `params`.

## Checking what registered

```python
import lumen

lumen.REGISTERED
# ('GatedDeltaNetState', 'BlockState', 'StackState', 'UndertowState')
```

Four names is the whole set. A shorter tuple means `torch` on this machine did
not expose an entry point one of the states needs — the library is unaffected
and every path documented elsewhere behaves identically, but a `torch.func`
transform will still refuse the states that are missing. Reading it is cheaper
than rediscovering the same fact from a `vmap` error inside a recurrence.

`register_state_pytrees()` returns the same tuple and can be called at any time;
after import it has nothing left to do.

## What it does not change

Nothing. Registration mutates a registry inside `torch`, not anything here — no
tensor changes value, no checkpoint key moves, and no existing call path behaves
differently. A trunk built from a given seed has the same weights and produces
the same outputs with the registration as without it.
