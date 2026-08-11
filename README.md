# Lumen

**One place for the components worth keeping.**

> Over the years, I've accumulated some real oddities in my research, and I need a single place to keep them organized, optimized, and well-documented. I had a habit of rebuilding the same components and techniques over and over again for each new project, so this is my attempt to centralize those in a single location. I have found value in these approaches, and I hope that by sharing these ideas, others can build upon them. — [Kenneth Grace](https://github.com/kennethgrace)

## The problem this solves

Rebuilding the same layer for each new project costs more than the typing. It
costs twice, and the second cost is the one that hurts.

**Findings get stranded in whichever copy discovered them.** Gated DeltaNet is
the case that made this concrete: four implementations across four projects, not
copies of each other — each had drifted, and three had grown something the
others lacked. One had measured that uncentred keys were costing it most of its
addressable directions, and fixed it. The other three never learned. A separate
copy was meanwhile spending real parameters widening its key axis to buy back
directions the first had recovered for free. Neither result could see the other.

**And the projects stop being able to talk.** Two of them can be training what
is, in theory, the same layer, at the same time, and have no way to put their
numbers side by side — nothing is shared except the intent. Same architecture,
two implementations, no common surface. Incommensurable.

A shared core does not make every result comparable. A model trained on one
corpus and a model trained on another still are not, and no amount of shared
code fixes that. What it removes is the axis that was never supposed to be a
variable: when the layer is *the same object*, a difference in the numbers is a
difference in the experiment.

That is what this repository is for.

## Status

Honest version: **the infrastructure is here, the components are not yet.**

Consolidating a layer means proving the merged version reproduces what it
replaces, on hardware that is not the machine it was written on. Both of those
need instruments, so the instruments were built first.

### Shipped

| | |
|---|---|
| [`lumen.probe`](./docs/HARDWARE_PROBE.md) | What a machine *actually does*, measured — dtype reality-checking, Triton viability, autotune cost |
| `lumen.bench` | Layer benchmarking that reports the **backward pass separately** — see below |

### Being consolidated

Every component gets a design record before it gets code — what the existing
implementations uniquely contribute, what was decided and on what evidence, and
what is still open. That record is the thing that stops the next rebuild.

A component is listed here once its record lands, and not before. A roadmap of
things that do not exist yet is a promise, not a description.

## Install

```bash
pip install -e .            # torch is the only hard dependency
pip install -e '.[dev]'     # + pytest
```

## Start here

```bash
lumen-probe --all           # what does this machine actually do?
lumen-probe --json          # same, machine-readable
```

## The discipline

Consolidated code only pays off if it still runs later, and elsewhere. In August
2026, reproducing a linear-attention paper published that May failed at three
separate walls:

1. A transitive dependency had deleted six capability flags the code imports.
2. A `git+https://…` dependency, pinned by nothing, required a snapshot that
   exists at **no public commit**.
3. A hard-coded CUDA arch gate in an unrelated import path.

None of the three had anything to do with the GPU. A brand-new datacentre card
hits all three exactly as hard as a 2016 Tesla P40 does.

So Lumen **pins what moves, probes what it cannot pin, and degrades cleanly**
instead of asserting version floors that are wrong. Research dependencies get
pinned, because that is what moves without warning; `torch` does not, because
the user already has a build matched to their driver. `triton` is not a
dependency at all — it is a runtime capability that gets *measured*.

The reference path for every component is fp32 and dependency-free. Faster paths
are welcome behind the same interface, but they have to beat the reference on
the machine in front of you, and that is a measurement, not an assumption.

## Worked example: a Tesla P40

The P40 is the development bench for this repository, and a useful worst case:
it is where assumptions fail hardest and therefore where the instruments get
tested. All numbers below are from that one card (SM 6.1) — they are evidence,
not specifications.

```
  MEASURED (achieved TFLOP/s, 2048^3 matmul)
    dtype          TFLOP/s   vs fp32   status
    float32           7.70     1.00x   ok
    float16           9.75     1.10x   ok
    bfloat16          5.54     0.62x   EMULATED (slower than fp32)

  RECOMMENDED DTYPE    float32
```

`torch.cuda.is_bf16_supported()` returns `True` on this card. It has no bf16
hardware whatsoever. The probe reports what the silicon *does*, not what the API
*accepts* — and when the two disagree, the disagreement is the finding.

That gap is not a Pascal quirk; it regenerates at every frontier, wherever an
API accepts a dtype ahead of the silicon that runs it natively. Which is why the
probe measures rather than consulting a compute-capability table.

## Why the backward pass gets its own column

Most kernel benchmarks report the forward pass. Training does not care about the
forward pass in isolation, and the two can point in opposite directions. Same
card, hand-rolled fp32 torch versus a production Triton kernel at matched
parameter count:

| T | forward | backward | end-to-end |
|---|---|---|---|
| 512 | triton **1.21x** | torch **1.98x** | torch **1.58x** |
| 1024 | triton **1.35x** | torch **2.01x** | torch **1.60x** |
| 2048 | triton **1.46x** | torch **2.06x** | torch **1.62x** |
| 4096 | triton **1.51x** | torch **2.07x** | torch **1.62x** |

Triton wins every forward and loses every backward, and the backward is the
larger term. A forward-only benchmark reports the exact opposite conclusion,
with a straight face.

## License

MIT. See [LICENSE](./LICENSE).
