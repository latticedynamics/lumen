# Gated DeltaNet

A fixed-size associative memory with a delta-rule write. Each state is a matrix
that a key addresses and a value is written into; the write *replaces* what was
at that address rather than accumulating on top of it, and a decay forgets
everything uniformly. Cost is linear in sequence length and the state does not
grow, which is what makes generation constant-memory.

The design rationale — what the layout means, which defaults are earned and
which are not, what is deliberately excluded — is in
[the design record](design/GATED_DELTANET.md). This page is how to use it.

```python
from lumen.gdn import GatedDeltaNet, GatedDeltaNetConfig, HeadLayout

mixer = GatedDeltaNet(GatedDeltaNetConfig(d_model=512, layout=HeadLayout.shared_key(8)))
y = mixer(x)                                   # (B, T, 512) -> (B, T, 512)
```

`d_model` and `layout` are the only required arguments. Everything else has a
default, and the two that matter are discussed below.

**There is no `n_heads` argument, and that is deliberate.** It is a read-only
property derived from the layout. A head count does not determine a head
*arrangement* — `shared_key(8)`, `diagonal(8)` and `crossed(2, 4)` are all eight
heads and all different models — so a config taking `n_heads=8` has to pick one
of them on your behalf. This one makes you say which.

---

## Interchangeable with Undertow

Both mixers implement the same three methods and the same output shape, so a
block can hold either without knowing which:

```python
class Block(nn.Module):
    def __init__(self, mixer):
        self.mixer = mixer                     # GatedDeltaNet or UndertowAttention
        ...

    def forward(self, x, *, state=None, return_state=False):
        return self.mixer(self.norm(x), state=state, return_state=return_state)
```

That is a deliberate commitment rather than a coincidence: the output side of
each layer — per-head RMSNorm, SiLU gate, projection — is shaped to match the
other. There is a test that holds both to it.

## Generation

`init_state` / `step` / `forward(..., state=, return_state=)`. Prefill the
prompt in one parallel pass, then step:

```python
y, state = mixer(prompt, return_state=True)    # (B, T_prompt, d)
token = sample(y[:, -1])

for _ in range(max_new_tokens):
    y, state = mixer.step(embed(token), state)  # (B, 1, d)
    token = sample(y[:, -1])
```

Device and dtype follow the module, so `mixer.cuda().init_state(batch)` needs no
second argument.

The state is **frozen**: `step` returns a successor rather than mutating what it
was given. Branching a stream — beam search, speculative decode — cannot leave
two branches sharing a buffer.

```python
state = mixer.init_state(batch)
_, branch_a = mixer.step(x, state)             # `state` is untouched
_, branch_b = mixer.step(other, state)         # and reusable
```

`forward` also accepts a state, which makes chunked prefill a one-liner. Feeding
a long prompt in pieces and feeding it whole give the same answer:

```python
state = None
for piece in chunks(prompt, 512):
    y, state = mixer(piece, state=state, return_state=True)
```

**The state has two parts and both matter** — the memory and the short
convolution's cache. Keep the `GatedDeltaNetState` object; do not rebuild one
from `state.memory` alone, or the first few positions after a resume will be
wrong in a way that is easy to miss.

## Reading without writing

Every read the layer performs in `forward` and `step` is paired with a write:
position `t` writes `k_t, v_t` and then reads with `q_t`. Two methods read
without writing.

**Re-read a whole pass with fresh queries.** `forward(..., return_cache=True)`
hands back the pass's write trajectory, and `reread` reads it again:

```python
y, cache = mixer(x, return_cache=True)         # x wrote the memory
y2 = mixer.reread(probe, cache)                # probe asks it; nothing is written
```

`y2` is exactly what `forward` would have produced had `probe` been the query
side while the keys, values, write strengths and decays stayed those of `x`.
With `probe = x` it is `y`, bit for bit. The cost is the query projection,
three matmuls and the output path; the scan — the sequential part — is skipped,
because the readout is linear in the query once the trajectory is known.
`return_state` and `return_cache` compose: both together return
`(y, state, cache)`.

**Read one position against a stream.** `read(x, state)` is `step` with the
write switched off — the query `step` would form from `x`, applied to the
memory as it stands, and no successor state:

```python
y, state = mixer.step(token, state)            # writes, then reads
peek = mixer.read(probe, state)                # reads what `probe` would see; `state` untouched
```

Both form the query through the layer's own path — projection, short conv,
SiLU, centring, normalisation — because that is the address space the memory
was written under. The short conv runs against the context the original query
had: for `reread`, the cache carries the conv state the pass started from; for
`read`, the stream's conv cache is read and not advanced. Keep the
`GatedDeltaNetReadCache` object whole for the same reason you keep the state
whole.

`backend="fla"` cannot build a read cache — its kernel returns only the final
state — and `return_cache=True` on such a layer raises rather than quietly
producing one from the reference. Use the reference backend where re-reads are
needed. `read` needs no cache and works on every backend.

## Sequence length

Any length. A sequence that does not fill its last chunk is padded internally
with no write and no decay, so the state and every real output are exactly what
they would have been. `chunk_size` is a performance dial, not a constraint on
your data.

## Head layout

A head is a `(key group, value group)` pair. Four named layouts cover the cases
anyone runs, and `shared_key` is the one to reach for if you are unsure:

```python
from lumen.gdn import HeadLayout

HeadLayout.shared_key(8)      # the ordinary choice: one address space, 8 payloads
HeadLayout.diagonal(8)        # one key and one value per head
HeadLayout.crossed(2, 4)      # every (key group, value group) pair
HeadLayout.shared_value(8)    # 8 address spaces, one payload
```

The layout is the only place the head count lives:

```python
config = GatedDeltaNetConfig(d_model=512, layout=HeadLayout.crossed(2, 4))
config.n_heads       # 8, derived -- there is no second field to disagree with
```

`n_heads` was a separate argument until 0.3, cross-checked against the layout at
construction. Deriving it makes disagreement *unrepresentable* rather than
*detected*, which is the stronger guarantee: there is no state in which the two
are both present and wrong.

Two things worth knowing before choosing one. `β` lives on the **key group**, and
so does `α` by default — so the layout also decides how many independent
forgetting timescales the layer has, and `shared_key` has one. That coupling is
breakable: `decay="state"` gives every state its own timescale without touching
the addresses, which is the dial to reach for if what you wanted from a layout
was horizon diversity. And `diagonal` is the ordinary one-key-one-value
arrangement, which is there so this layer has something outside itself to be
compared against.

## Widths

`expand_k` and `expand_v` set the total key and value width as multiples of
`d_model`; per-state `d_k = expand_k · d_model / n_heads`, and likewise for
`d_v`. The default is a **wide key side and a narrow value one**
(`expand_k=2.0`, `expand_v=1.0`), which inverts the more common ratio.

The reasoning is in the design record; the short version is that
`rank(M) ≤ min(d_k, d_v)`, and cross-talk between l2-normalised keys falls as
`1/√d_k`, so the key axis is the one that buys distinguishable addresses. The
default is backed by a controlled comparison rather than inherited, but it is
one training scale on one corpus — if your setting differs, measure.

## Other options

| option | default | what it does |
|---|---|---|
| `chunk_size` | 64 | Parallel block size. A power of two. **Performance only** — it does not enter the answer, and it does not bound what the layer can represent. See below. |
| `conv_size` | 4 | Short causal depthwise convolution on q/k/v. Local mixing, not a positional code — it carries relative offsets only. `0` disables it. |
| `beta_max` | 2.0 | Write-strength ceiling. Above 1 the update becomes a reflection rather than only a contraction. **2 is also the maximum**, and it is refused above that — see below. |
| `centre` | `False` | Learned per-head key/query centres, subtracted before the l2 norm. Zero-initialised, so switching it on is an exact no-op until training moves it. |
| `decay` | `"key_group"` | Where the forgetting timescale lives. `"state"` gives every state its own, via a zero-initialised offset — so switching it on is an identity at init and diverges under training. `"state_gated"` gives every state its own *and* its own input modulation, by widening `a_proj`. Off by default because it is an experiment nobody has run, not because it lost one. |

### Why `beta_max` stops at 2

Not taste. The triangular solve inside the chunkwise form advances by
`(I − β k kᵀ)`, whose spectral norm is `max(1, |1 − β|)` — exactly 1 for every
`β` in `[0, 2]`, and greater than 1 past it. At 2 the update is a reflection and
norm-preserving; beyond, it expands, and the inverse grows *geometrically in
`chunk_size`*. On one machine, near-identical unit keys in fp64, the largest
entry of that inverse is flat at 1.98 from `C = 16` to `C = 256` at `β_max = 2`,
and runs 5.8 → 1.7e8 over the same range at 2.1.

The failure is worse than it looks because it hides: the decay damps the solve,
so a too-large `beta_max` works while the model forgets quickly and breaks once
it learns to remember. It is refused on construction instead.

### Per-state decay

`decay="state"` adds one learned scalar per state inside the softplus:

```python
config = GatedDeltaNetConfig(
    d_model=512, layout=HeadLayout.shared_key(8), expand_k=2.0, decay="state"
)
```

It costs `H` scalars — not a wider projection — and the `O(C³)` triangular solve
is still built once per key group, so the arrangement does not undo what the head
layout buys. A **fixed geometric band of timescales** is an init of that same
parameter rather than a separate option:

```python
with torch.no_grad():
    layer.a_offset.copy_(torch.linspace(-2.0, 2.0, m).view(g_k, m))
```

Checkpoints do not cross the boundary silently: `a_offset` is a new parameter, so
loading between the two arrangements raises rather than quietly dropping it.

### Per-state decay that is also gated

`decay="state_gated"` gives every state its own timescale *and* its own
modulation by the input, by widening `a_proj` from one output per key group to
one per state. It is the more expensive of the two — `d_model × (H − G_k)`
weights against `state`'s `H` scalars — and it is a different model, not a
second spelling of one.

Its init guarantee is deliberately the weaker of the two, and the difference is
worth reading before comparing runs. `state` is an **exact identity** with
`key_group` at init, because it adds zero to the same projection. `state_gated`
reproduces the **arrangement** only — every state in a key group starts on one
gate, so the layer has to earn its way out of `key_group` — but a wider
`nn.Linear` draws a different number of values, so its rates are not the rates a
`key_group` layer built from the same seed would have had.

That arrangement is re-established by `apply_init_structure()` rather than by the
constructor alone, because a trunk-level init redraws the layer's projections
afterwards. See [BLOCK.md](./BLOCK.md#structure-among-a-sub-layers-weights); if
you write your own trunk that redraws `nn.Linear` weights, call it.
| `norm_eps` | 1e-5 | Output RMSNorm epsilon. |
| `dropout` | 0.0 | Applied after the output projection. |
| `backend` | `"reference"` | `"reference"` or `"fla"` — see [backends](#backends). Opt-in, never auto-detected. |

### Why `chunk_size` is only performance

Worth stating explicitly, because for a while it was not true here.

The chunkwise form absorbs the decay by writing the state as `M_t = γ_t M̃_t`.
Everything that reaches an output is then the ratio `γ_t / γ_i` for `i ≤ t`,
which lies in `(0, 1]`. The direct way to compute that is `exp(g_t − g_i)` for
`g = log γ`; the tempting way is `γ_t · (1/γ_i)`, which reaches the same answer
through a factor that grows without bound. In fp32, `1/γ` passes the largest
representable value once accumulated decay within a chunk exceeds ≈88.7.

A layer computing it the second way needs a ceiling on accumulated decay to stay
in range — and that ceiling is a bound on `Σ log α` *within one chunk*, so it
implies a shortest expressible half-life of

```
    chunk_size · ln 2 / ceiling
```

which is **linear in the chunk size**. A parameter chosen to tile the sequence
for the GPU then decides how fast a head is permitted to forget, and raising it
for throughput silently lengthens the shortest memory the model can express.

This implementation forms the relative decay directly, so no ceiling exists and
no such coupling exists. If you are comparing against an implementation that has
a `max_chunk_decay`, `decay_bound` or similar, that is what it is for — and its
`chunk_size` is not a free parameter in the way this one is.

## Positional encoding

There is none, and none is wanted. The recurrence *is* the order. The short
convolution carries relative offsets within its window and nothing absolute.

## Subclassing

Reuse is by subclassing. Three methods are the seams:

| method | override to |
|---|---|
| `_features` | change how q/k/v/β/α are produced |
| `_scan` | swap in a different kernel |
| `_out` | change the output path |
| `_reread_query` | give reads-without-writes their own projection, or skip the unused k/v/β/α work |

`_reread_query`'s default runs `_features` and keeps `q`, so a subclass that
changes how `q` is produced gets re-reads in its own address space without
overriding anything else. It is derived that way rather than factored out of
`_features` precisely so that cannot go wrong silently; the price is the unused
projections, which is why the seam exists.

`_out` carries an interface commitment: replacing it is what breaks
interchangeability with Undertow, and the per-head normalisation it performs is
relied on elsewhere. Know that before you replace it.

## Backends

The reference path is fp32, depends on nothing but `torch`, and is the default.
An accelerated path is available beside it and is **opt-in**:

```python
config = GatedDeltaNetConfig(
    d_model=2048, layout=HeadLayout.diagonal(32),
    expand_k=1.0, expand_v=4.0, backend="fla",
)
```

It dispatches to [`flash-linear-attention`][fla]'s Triton chunk kernels. Install
it yourself and pin it — it is someone else's research code and moves without
warning.

[fla]: https://github.com/fla-org/flash-linear-attention

It is called `"fla"` rather than `"triton"` on purpose. Undertow's
`backend="triton"` means *Lumen's own* Triton kernels; this is a third-party
dependency. One word cannot name both, or an archived config no longer says
which code ran — and it leaves `"triton"` free here for a kernel Lumen writes.

Not auto-detected, and that is deliberate rather than cautious. A fast path that
switches itself on whenever a package happens to be importable means two
installations of this layer are no longer running the same object, and a
difference in their numbers stops being a difference in their experiment. Ask
for it, measure it on your hardware, and record that you did.

### What it needs

Refused at construction, not at the first forward:

| requirement | why |
|---|---|
| `HeadLayout.diagonal(...)` | fla implements one key per head; `crossed`, `shared_key` and `shared_value` have no kernel |
| `decay` is `"key_group"` or `"state"` | on a diagonal layout these are the same object (`G_k = H`, `m = 1`). `"state_gated"` widens `a_proj` to `H` independent signals and is a different model |
| `fla` importable | requesting a backend that cannot be built is an error, not a silent downgrade |

A CUDA-only path meeting a CPU or fp64 tensor falls back quietly instead —
that is a device or dtype gap, not a missing capability, and an fp64 oracle run
taking the reference is the right answer rather than a limitation of it.

**`return_cache=True` is refused under `"fla"`**, on every device. Its kernel
returns only the final state, not the per-chunk states a re-read needs. The
refusal is unconditional rather than a fallback because fla already falls back
to the reference on CPU, and a capability present on a CPU box and absent on
the GPU the same config trains on would be a trap.

**`chunk_size` is inert under `"fla"`.** fla chooses its own chunking and this
value cannot reach it. That is numerically safe because the answer does not
depend on the chunk size at all (see [below](#why-chunk_size-is-only-performance)),
but it is worth knowing before you read a value back out of an archived config.

### Measured, on one machine

One layer, `d_model=2048`, `diagonal(32)`, `expand_k=1.0`, `expand_v=4.0`,
`B=4`, `T=1024`, forward+backward, on an A100-SXM4-40GB. Both sides
58,900,800 parameters:

| path | ms | vs reference |
|---|---|---|
| reference, fp32 | 116.6 | 1.00× |
| fla, fp32 | 87.3 | 1.33× |
| reference, bf16 autocast | 36.5 | 3.20× |
| **fla, bf16 autocast** | **11.4** | **10.3×** |

Evidence, not specification — and the mechanism does not generalise the way
Undertow's does. Undertow's Triton win is bandwidth and launch overhead, which
is why it holds on a card without tensor cores. This one is largely
*arithmetic*: the chunk inner loop is matmul-dense, so expect it to shrink
sharply without tensor cores.

How sharply depends on the configuration, not just the card. On a Tesla P40
(SM 6.1, no tensor cores), fp32, `d_model=512`, `diagonal(8)`, `expand_v=4.0`,
`T=512`: fla takes the forward and **loses the backward**, netting roughly
1.2× end to end. At the crossed layout `lumen.bench`'s docstring records, the
same card gives the reference ~1.6× overall. Same hardware, opposite verdict,
because the two configurations put different amounts of work in the part fla
accelerates.

> Those P40 figures were taken while the card was running an unrelated job at
> ~50% utilisation, so treat the ratio as indicative and the *direction* as the
> claim. The A100 table above was measured on an idle device.

So: none of these numbers is a property of the implementations. Measure the
configuration you actually train, on the card you actually train it on, and
report the backward separately — an accelerated forward attached to a slower
backward is how a kernel wins every microbenchmark and loses the run.

### It computes the same function

Not a claim about speed, and the one that licenses switching at all. Against
Lumen's own fp64 sequential oracle:

| | relative error |
|---|---|
| reference chunkwise vs oracle | 2.07e-7 |
| fla vs oracle | 2.24e-7 |
| gradients, fla vs reference | 2.4e-7 … 5.3e-7, all five inputs |

fla lands the same distance from the oracle as the chunkwise path does — one
algorithm reached by two routes, rather than two algorithms for one function.
`tests/test_gdn_backend.py` holds this as a standing test, and asserts fla is
no worse than the path already trusted rather than merely under a threshold.

Switching backend mid-experiment still changes your numbers at fp32 round-off.
That is small, and it is not zero.

## Precision and hardware

Run the GPU-marked tests locally before relying on a GPU path; CI is CPU-only:

```bash
pytest -m "gpu or triton"
```

## Verifying it yourself

The suite holds the layer to the acceptance conditions in the design record: the
chunkwise path reproduces a plain sequential implementation of the recurrence to
1e-9 in fp64 at every layout, `step` reproduces `forward`, chunked prefill
reproduces one pass, and zero-initialised centring is bit-identical to no
centring at all.

```bash
pytest tests/test_gdn.py -q
```

The sequential implementation is kept in `lumen.gdn.reference` permanently. It
is not dead code; it is the specification, and the fast path is answerable to
it.
