# Hardware Probe

`lumen.probe` answers one question: **what does this machine actually do?**

Not what its support flags claim. Not what the compute-capability table in a
blog post says. What it does, measured, in achieved TFLOP/s.

```bash
lumen-probe --all
lumen-probe --device 0 --json
lumen-probe --skip-triton        # skip kernel compilation
```

## Why a consolidation library ships a hardware probe

Merging four implementations into one is only worth doing if the survivor runs
on all the machines the four did. That makes "what is this machine capable of"
a question the library has to answer honestly rather than infer — and the usual
way to infer it, reading a compute capability and consulting a table, produces
confident wrong answers.

So capability here is **measured**, and every component's reference path is the
one that needs no capability at all.

## The problem it solves

Support flags describe what an API will *accept*, not what the silicon will
*do*. Where those diverge, nothing in the type system, the return value, or the
docs will tell you — the code runs, produces correct numbers, and is quietly
slower than the fallback it replaced.

`torch.cuda.is_bf16_supported()` returns `True` on a Tesla P40. The P40 has no
bf16 hardware whatsoever — torch quietly widens to fp32 and narrows back, and
the result is **1.6x slower** than simply asking for fp32. You find out by
training something for six hours and wondering why a 24 GB card is slower than
it should be.

That is not a Pascal quirk. The gap regenerates at every frontier, wherever an
API accepts a dtype ahead of the silicon that runs it natively; only the dtype's
name changes. Pascal is just where it is easiest to demonstrate, which is why
the examples below all come from **one Tesla P40 (SM 6.1)**. They are evidence
for the method, not a specification of the hardware you should expect.

| the claim | the reality on SM 6.1 |
|---|---|
| `is_bf16_supported() == True` | emulated; 0.62x fp32 |
| fp16 is faster than fp32 | measures 1.10x, but only from halved memory traffic — no fp16 compute path exists |
| Triton requires SM 7.0+ | **false** — compiles and runs correctly on SM 6.1 |
| Triton is fast because it's Triton | its autotune space mostly cannot fit in 48 KB of shared memory |

## What it reports

The report is deliberately two-column: **claimed** versus **measured**.

```
  CLAIMED (from compute capability)
    tensor cores       no
    bf16 native        no
    fp16 fast path     no
    tf32               no

  MEASURED (achieved TFLOP/s, 2048^3 matmul)
    dtype          TFLOP/s   vs fp32   status
    float32           7.70     1.00x   ok
    float16           9.75     1.10x   ok
    bfloat16          5.54     0.62x   EMULATED (slower than fp32)
```

Anything reduced-precision that measures *slower* than fp32 is flagged
`EMULATED`, because there is no other explanation: hardware that exists is not
slower than the fallback it replaces.

## The recommendation is not just "the fastest"

An early version of this module ranked dtypes by raw throughput and recommended
`float16` on the P40, on the strength of a 1.10x matmul win.

That was wrong, and the way it was wrong is instructive. The P40 has no fp16
compute path at all; cuBLAS widens to fp32 internally. The entire 10% comes
from moving half as many bytes. It buys none of bf16's exponent range, and fp16
training then requires loss scaling — machinery that costs more than 10%.

So a reduced-precision dtype now has to clear two bars:

1. the hardware must have a real path for it (`fp16_fast`, `bf16_native`)
2. the measured win must exceed `_WORTH_THE_NUMERICS` (1.15x)

Otherwise the answer is `float32`, with the reasoning printed in the notes.
A microbenchmark is not a training run.

## Triton: measured, not assumed

The probe never consults the compute capability to decide whether Triton works.
It compiles and runs an elementwise kernel and a `tl.dot`, and walks block sizes
upward until one fails. On a P40 that yields:

```
    version            3.5.1
    elementwise        ok
    tl.dot             ok up to block 64
    cold compile       1.13s
    est. autotune tax  ~45s per kernel per (shape, dtype)
```

`tl.dot` works to block 64 and fails at 128 — that is the 48 KB shared-memory
ceiling, found empirically rather than derived.

The **autotune tax** is the number that actually matters. Upstream chunk kernels
carry autotune spaces of roughly 40 configs, each compiled and benchmarked on
first use, keyed on shape *and* dtype. At 1.13s per cold compile that is ~45
seconds per kernel per shape — and one real measurement of an upstream kernel
took **185 seconds** for a single autotune sweep.

Most of those configs specify block sizes that cannot fit in 48 KB. You pay
full benchmarking cost to discover they don't work.

The mitigation is unglamorous and nobody upstream will do it: pin the configs
that fit, and persist `TRITON_CACHE_DIR` across runs.

## API

```python
from lumen import probe, format_report, arch_facts

result = probe(index=0)
print(format_report(result))

result.recommended_dtype        # "float32"
result.device.arch.tensor_cores # False
result.triton.max_dot_block     # 64
result.triton.autotune_tax_s    # ~45.0
result.to_dict()                # JSON-serialisable

arch_facts((6, 1)).generation   # "Pascal" -- pure, no device needed
```

`arch_facts` takes a compute capability and returns claims with no device
access, so the derivation logic is testable on a CPU-only box and in CI.

Every probe function captures failures into its result rather than raising.
"This dtype errors on this card" is a row in the table, not a crash.
