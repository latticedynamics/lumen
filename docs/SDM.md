# Sparse Delta Memory

A gated delta rule over a large, sparsely addressed table. Each head holds a
table of `N` slots; each position writes `W` of them and reads `R`, chosen by
product keys. Per-token work is `O((W+R)·d_v)` whatever `N` is, and a slot
nobody writes to is left exactly as it was — decay included — so **state size
is decoupled from parameter count and from compute.**

The design rationale — the diagonal-decay derivation, which defaults are earned
and which are not, what is deliberately excluded — is in
[the design record](design/SPARSE_DELTA_MEMORY.md). This page is how to use it.

```python
from lumen.sdm import SparseDeltaMemory, SparseDeltaMemoryConfig

mixer = SparseDeltaMemory(SparseDeltaMemoryConfig(
    d_model=512, n_heads=2, n_slots=128**2, initial_memory="learned",
))
y = mixer(x)                                   # (B, T, 512) -> (B, T, 512)
```

`d_model`, `n_heads`, `n_slots` and `initial_memory` are required. Everything
else has a default, and every default is listed in the design record with
whether anyone has compared it (§3.9 there — mostly, nobody has).

---

## Choosing the initial memory

`initial_memory` has **no default**, because its two values are two different
models:

| | what a stream starts from | what it adds |
|---|---|---|
| `"zero"` | an empty table | nothing; the large state is the whole proposition |
| `"learned"` | a trained table, `(n_heads, n_slots, d_v)` | `n_slots · d_model` parameters — a memory layer sharing the table with the fast-weight memory |

Because unwritten slots are frozen, what a learned table holds survives a
context rather than being decayed out of it; in-context writes overwrite it
slot by slot, as the write gate decides. The paper this layer comes from
reports that the two buy different things — the large state buys long-context
recall, the learned table buys most of the modelling loss (design record §3.5).

**A learned table starts at zero**, so at construction a `"learned"` layer *is*
the `"zero"` layer, bit for bit, and it only becomes a different model as
training moves it. It still trains from zero: every read sends a gradient to
the slots it lands on.

## Sizing the state

| | per layer, per stream |
|---|---|
| state | `n_slots · d_model` values (`d_v = d_model / n_heads` per head) |
| compute per token | `O((n_writes + n_reads) · d_v)` per head — independent of `n_slots` |
| address parameters | `2 · d_model · n_heads · 2√n_slots` — grows as `√n_slots` |
| learned table | `n_slots · d_model` parameters, under `"learned"` only |

`n_slots` is the knob that grows the state without growing compute — in
training as well as in the recurrence, because the kernel writes the table in
place and carries its gradient sparsely (design record §3.11). It is per head
and must be a perfect square, because product keys address it as
`√N × √N`. There is deliberately no `expand_v`: value width multiplies state
*and* compute, and `n_slots` already buys the first without the second.

At `d_model = 512`, `n_slots = 128²` is 8.4M values per stream per layer —
32 MiB in fp32. `512²` slots is 134M values, 512 MiB. Size a batch accordingly.

## Interchangeable with the other mixers

Same three methods as Gated DeltaNet and Undertow — `forward`, `init_state`,
`step` — and the same output side: per-head RMSNorm, SiLU gate, projection. A
`Block` can hold any of the three without knowing which, and a `Stack` of them
initialises correctly: the base draw redraws projection weights, leaves the gate
biases alone, and never touches a bare table.

## Generation

Prefill in one parallel pass, then step:

```python
y, state = mixer(prompt, return_state=True)    # (B, T_prompt, d)
token = sample(y[:, -1])

for _ in range(max_new_tokens):
    y, state = mixer.step(embed(token), state)  # (B, 1, d)
    token = sample(y[:, -1])
```

`init_state(batch)` hands out the initial table as a **broadcast view** — one
table, stride 0 along the batch — so no stream pays for a copy until its first
write. Under `"learned"` it is a view of the parameter, so a stream trained
through its state carries a gradient back to the table.

**`step` returns a successor and never mutates the state it was given**, the
same guarantee every mixer here makes, so branching a stream cannot leave two
branches sharing a buffer. For this layer that guarantee is not free: the
successor is a full table, `O(n_slots · d_v)` per head per step against
`O((W+R) · d_v)` of actual work. At small tables it does not matter; at very
large ones it dominates decode. An in-place decode path is an open question,
not a feature (design record §3.6).

## Reading without writing

`read(x, state)` is a step with the write switched off: the read address is
formed exactly as `step` would form it, applied to the table as it stands.
Nothing is written, nothing decays, and the state is unchanged. It is what a
probe token reads.

A re-read of a whole pass with fresh queries — Gated DeltaNet's `reread` — is
not offered: it would have to retain the table entering every chunk.

## Sequence length

Any. A sequence that does not fill its last chunk is padded internally with
positions that write nothing and decay nothing, which is exact: every real
output, and every slot of the final table, is what it would have been.

## Other options

| field | default | what it is |
|---|---|---|
| `n_writes`, `n_reads` | 64 | slots written and read per position. Untested: the paper states 64, its released configs use 128 |
| `key_norm` | `"softmax"` | how the selected scores become weights. `"l2"` normalises them to a unit vector instead — the delta rule at full strength, and the experiment for whether this layer is a delta rule at all (design record §4.2) |
| `decay_weighting` | `"write_set"` | every written slot decays by the full `α`. `"key"` scales each slot's decay by its write weight, removing the jump at the top-`W` boundary; untested, and refused with `key_norm="l2"` |
| `beta_max` | 2.0 | write-strength ceiling. The paper's is 1; past 2 is refused, because the chunkwise solve stops being stable there |
| `chunk_size` | 32 | numerically inert, and a speed and memory dial: training keeps `O(T · chunk_size · (W+R))` per sequence. Measured on one machine, 32 was fastest at every table size; smaller chunks use less memory. Where the optimum sits depends on the device (design record §3.9) |
| `norm_eps`, `dropout` | 1e-5, 0.0 | as in Gated DeltaNet |

## Optimising a learned table

`initial_memory_parameters()` returns the table (or nothing, under `"zero"`),
so an optimiser can treat it differently without matching parameter names. The
case for doing so: ordinary weight decay shrinks **every** slot every step,
read or not, so what a learned table holds and no batch reads decays by
optimiser step even though the context never touches it.

```python
from lumen.sdm import SparseDeltaMemory

tables = [
    p for m in model.modules() if isinstance(m, SparseDeltaMemory)
    for p in m.initial_memory_parameters()
]
chosen = {id(p) for p in tables}
optimizer = torch.optim.AdamW([
    {"params": [p for p in model.parameters() if id(p) not in chosen]},
    {"params": tables, "weight_decay": 0.0},
], weight_decay=0.1)
```

That is how to separate it, not a recommendation of what to do with it. What
the table *should* be optimised with is open.

## Many parameter sets at once

The layer runs under `torch.func.vmap` over stacked parameter sets, and its
state is a registered pytree — see [States as pytrees](PYTREE.md). That was a
design constraint on the kernel, not a given: every shape in it depends on the
configuration, never on the data. A learned table is a parameter like any
other, so it is stacked per set with the rest.

Under a transform the kernel holds its table functionally — each chunk writes
a new one — because the in-place path's gradient bookkeeping is invisible to
`torch.func`. It is chosen automatically and gives the same outputs; the cost
is a table copy per chunk, which grows with `n_slots` as nothing else here
does.

## Subclassing

Reuse is by subclassing. Four methods are the seams:

| method | override to |
|---|---|
| `_address` | change how slots are selected and weighted |
| `_features` | change anything else the kernel takes: values, write strength, decay |
| `_scan` | swap in a different kernel — or call this one with `in_place=False` to differentiate twice |
| `_out` | change the output path |

**An override of `_address` must keep write indices distinct within each
position.** The kernels depend on it and can check it, but the layer turns the
check off: product keys guarantee it by construction, and the check costs a
host sync per call and breaks `vmap`. `_out` carries the interchangeability
commitment, as in Gated DeltaNet.

## Precision and hardware

The reference path is fp32 and plain PyTorch — a sort and a binary search,
gathers, index writes, batched matmuls and one triangular solve per chunk. There is no compute-capability
floor, and it runs on a Pascal card as it does on anything newer. The forward
has no atomics, so it is deterministic on a GPU without asking.

Training memory has two terms beyond the table itself: the rows each chunk
gathers, `O(T · (W+R) · d_v)` per head, and the pairwise terms,
`O(T · chunk_size · (W+R))`. The table is not among them: training writes it in
place and never retains it. For long sequences, `Stack.recompute = True` trades
a second forward for the first.

The backward adds each gathered row's gradient back into the table's with a
scatter-add, so on a GPU its sums can reorder from run to run. Under
`torch.use_deterministic_algorithms(True)` they do not.

The in-place path cannot be differentiated twice; a second derivative through
it is refused. The kernel's `in_place=False` can be, at the cost of a table
copy per chunk.

Run the GPU-marked tests locally; CI is CPU-only:

```bash
pytest -m gpu tests/test_sdm.py
```

## Verifying it yourself

The suite holds the layer to the acceptance conditions in the design record:
the chunkwise path reproduces a plain sequential implementation of the
recurrence to round-off in fp64 at every chunk size; with every slot selected,
it *is* Gated DeltaNet, checked against Lumen's own GDN oracle; unwritten slots
come out bit-identical; gradients match the oracle; `step` reproduces `forward`.

```bash
pytest tests/test_sdm.py -q
```

The sequential implementation stays in `lumen.sdm.reference` permanently. It is
the specification, and the fast path is answerable to it.
