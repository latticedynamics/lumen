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
from lumen.gdn import GatedDeltaNet, GatedDeltaNetConfig

mixer = GatedDeltaNet(GatedDeltaNetConfig(d_model=512, n_heads=8))
y = mixer(x)                                   # (B, T, 512) -> (B, T, 512)
```

`d_model` and `n_heads` are the only required arguments. Everything else has a
default, and the two that matter are discussed below.

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

## Sequence length

Any length. A sequence that does not fill its last chunk is padded internally
with no write and no decay, so the state and every real output are exactly what
they would have been. `chunk_size` is a performance dial, not a constraint on
your data.

## Head layout

A head is a `(key group, value group)` pair. Four named layouts cover the cases
anyone runs, and the default is `shared_key`:

```python
from lumen.gdn import HeadLayout

HeadLayout.shared_key(8)      # default: one address space, 8 payloads
HeadLayout.diagonal(8)        # one key and one value per head
HeadLayout.crossed(2, 4)      # every (key group, value group) pair
HeadLayout.shared_value(8)    # 8 address spaces, one payload
```

Pass one explicitly when you want it, and `n_heads` is cross-checked against it:

```python
GatedDeltaNetConfig(d_model=512, n_heads=8, layout=HeadLayout.crossed(2, 4))
```

A layout that disagrees with `n_heads` is an error at construction rather than a
quietly different model.

Two things worth knowing before choosing one. `α` and `β` live on the **key
group**, so the layout also decides how many independent forgetting timescales
the layer has — `shared_key` has one. And `diagonal` is the ordinary
one-key-one-value arrangement, which is there so this layer has something
outside itself to be compared against.

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
| `chunk_size` | 64 | Parallel block size. A power of two. Performance only. |
| `conv_size` | 4 | Short causal depthwise convolution on q/k/v. Local mixing, not a positional code — it carries relative offsets only. `0` disables it. |
| `beta_max` | 2.0 | Write-strength ceiling. Above 1 the update becomes a reflection rather than only a contraction. |
| `max_chunk_decay` | 8.0 | Bounds accumulated decay within a chunk, which keeps the chunkwise form's `1/γ` in range. |
| `centre` | `False` | Learned per-head key/query centres, subtracted before the l2 norm. Zero-initialised, so switching it on is an exact no-op until training moves it. |
| `norm_eps` | 1e-5 | Output RMSNorm epsilon. |
| `dropout` | 0.0 | Applied after the output projection. |

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

`_out` carries an interface commitment: replacing it is what breaks
interchangeability with Undertow, and the per-head normalisation it performs is
relied on elsewhere. Know that before you replace it.

## Precision and hardware

The reference path is fp32 and depends on nothing but `torch`, because that is
the path that runs everywhere. There is no accelerated kernel in this version —
one ships when it has been measured to beat the reference here, the way
Undertow's was.

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
