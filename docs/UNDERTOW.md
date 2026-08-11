# Undertow

Fixed-window causal attention with no positional encoding and an optional
graded window boundary.

Each query attends to at most `window` past positions and nothing else. There is
no RoPE, no learned position embedding, and no distance term in the score — what
ordering information the layer has comes from the causal window itself. An
optional `plateau` grades the boundary: full strength out to that distance, then
a cosine ramp toward the window edge, entering as an additive log-space bias
before the softmax.

For *why* it is shaped this way — the placement equivalence, the masking
decision, the backward structure — see [the design record](design/UNDERTOW.md).

---

## Quick start

```python
import torch
from lumen import UndertowAttention, UndertowConfig

attn = UndertowAttention(UndertowConfig(
    d_model=384,
    n_heads=8,
    window=32,      # required: each query reaches 32 positions, itself included
    plateau=24,     # full strength to distance 24, cosine ramp over the last 8
))

x = torch.randn(2, 256, 384)
y = attn(x)                      # (2, 256, 384)
```

`plateau=None` (the default) gives a hard window — ordinary sliding-window
attention, no ramp.

## Configuration

| field | default | meaning |
|---|---|---|
| `d_model` | — | residual width; must divide evenly into `n_heads` |
| `n_heads` | — | attention heads |
| `window` | — | positions a query may reach, itself included |
| `plateau` | `None` | distance held at full strength before the ramp; `None` = hard window |
| `dropout` | `0.0` | applied to the layer output |
| `zero_init` | `False` | start `o_proj` at zero — see [splicing](#splicing-into-a-trained-stack) |
| `eps` | `1e-6` | per-head RMSNorm epsilon |
| `backend` | `"reference"` | `"reference"` or `"triton"` — see [backends](#backends) |

`window` has no default on purpose. Useful values are task-dependent, and this
library does not canonise a number nobody has compared.

Everything is validated when the config is constructed, not when the layer first
runs, so a mistake surfaces at the line that made it.

A sequence shorter than the window is fine — it clamps rather than raising. A
short batch is not a user error.

## Streaming

The layer offers Lumen's standard streaming interface, so a block can hold this
or Gated DeltaNet without knowing which one it has:

```python
state = attn.init_state(batch=2)

for t in range(n_tokens):
    y_t, state = attn.step(x[:, t:t + 1], state)    # (B, 1, d_model)
```

The state is a ring buffer of the last `window - 1` keys and values. It is
**constant in generated length** — that is what a fixed window buys, and it is
enforced by a test rather than claimed here.

Prefill a prompt with the parallel path, then continue token by token:

```python
prompt_out, state = attn(prompt, return_state=True)

for _ in range(n_new):
    y_t, state = attn.step(next_token_embedding, state)
```

`forward` also accepts an incoming state, so a long input can be processed in
chunks without changing the result:

```python
out_a, state = attn(x[:, :512], return_state=True)
out_b, state = attn(x[:, 512:], state=state, return_state=True)
# torch.cat([out_a, out_b], dim=1) equals attn(x)
```

`UndertowState` is frozen and `step` returns a successor rather than mutating in
place, so branching a stream — beam search, speculative decoding — cannot leave
two branches sharing a buffer.

## Backends

The reference path is fp32 and depends only on `torch`. A Triton path is
available and is **opt-in**:

```python
config = UndertowConfig(d_model=512, n_heads=8, window=32, backend="triton")
```

It is not auto-detected, and that is deliberate rather than cautious. A fast
path that switches itself on whenever a package happens to be importable means
two installations of this layer are no longer running the same object — and a
difference in their numbers stops being a difference in their experiment. Ask
for it, measure it on your hardware, and record that you did.

Requesting `"triton"` when Triton will not import raises at construction rather
than silently downgrading. A CUDA-only path meeting a CPU tensor falls back
quietly, because that is a device gap and not a missing capability.

Measured on one machine — a Tesla P40 (Pascal, SM 6.1, no tensor cores), fp32,
`d_model=512`, 8 heads, batch 4:

| window | forward | forward + backward |
|---|---|---|
| 8 | 1.9–2.0× | 1.55× |
| 32 | 3.5–4.1× | 2.2–2.4× |

Those are evidence from one card, not a specification. What generalises is the
mechanism: the fused kernel removes `window` kernel launches and `window` passes
over K and V, so the advantage grows with the window and comes from bandwidth
rather than arithmetic — which is why it does not depend on tensor cores.

Measure it yourself with [`lumen.bench`](../README.md), which reports the
backward separately for exactly this reason.

## Training convention: skip the context prefix

Not part of the layer — but if you train with a window, you probably want this.

The first `window` positions of a sequence do not have a full window behind
them. Position 3 sees four tokens; position 200 sees the whole window. Training
on both averages together a difficulty gap that is an artifact of where the
sequence happens to start, not something you want the model to learn.

Compute the loss only on positions from `window` onward:

```python
logits = model(tokens)
loss = F.cross_entropy(
    logits[:, window:].reshape(-1, vocab_size),
    targets[:, window:].reshape(-1),
)
```

Every position contributing to the loss then saw exactly `window` positions of
direct context. The cost is `window / seq_len` of your tokens — 12.5% at
`seq_len=256, window=32`, about 3% at `seq_len=1024` — so longer sequences
amortise it.

This is three lines and it stays in your training loop. The layer has never
heard of logits or targets, and coupling it to them is how a layer stops being
reusable.

## Splicing into a trained stack

`zero_init=True` starts the output projection at zero, making the layer an exact
identity no-op:

```python
attn = UndertowAttention(UndertowConfig(..., zero_init=True))
assert torch.equal(attn(x), torch.zeros_like(x))
```

At step 0 a model with this layer spliced in is bit-identical to the checkpoint
it came from. Nothing is destroyed, and the layer earns its contribution from
zero instead of injecting noise into a converged residual stream.

## Testing

```bash
pytest                          # everything available on this machine
pytest -m "not gpu and not triton"   # what CI runs
pytest -m "gpu or triton"            # needs a CUDA device
```

The suite keeps a dense `O(T²)` implementation permanently and checks the
windowed path against it at every position — including the partial-window
prefix, which is comparable precisely because absent keys are masked rather than
zero-padded. The obvious implementation is not dead code; it is the
specification.
