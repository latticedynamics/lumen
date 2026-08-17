# Lumen

**One place for the components worth keeping.**

[![tests](https://github.com/latticedynamics/lumen/actions/workflows/tests.yml/badge.svg)](https://github.com/latticedynamics/lumen/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.10%20–%203.13-blue)](https://www.python.org/)
[![torch](https://img.shields.io/badge/torch-%E2%89%A5%202.0-ee4c2c)](https://pytorch.org/)
[![license](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

Lumen is a consolidation library for PyTorch research components: sequence
mixers, hardware instruments, and the design record for each one. Components
that get rebuilt once per project are merged here a single time — organised,
measured, documented, and held to one interface, so that two projects using the
same layer are demonstrably using *the same layer*.

> Over the years, I've accumulated some real oddities in my research, and I need
> a single place to keep them organized, optimized, and well-documented. I had a
> habit of rebuilding the same components and techniques over and over again for
> each new project, so this is my attempt to centralize those in a single
> location. I have found value in these approaches, and I hope that by sharing
> these ideas, others can build upon them.
> — [Kenneth Grace](https://github.com/kennethgrace)

---

## Install

```bash
pip install -e .            # torch is the only hard dependency
pip install -e '.[dev]'     # + pytest
```

Python 3.10–3.13. `triton` is deliberately **not** a dependency — it is a
runtime capability that gets probed, and every component runs without it.

## What does this machine actually do?

```bash
lumen-probe --all           # human-readable report per device
lumen-probe --json          # same, machine-readable
```

## Quick start

Every mixer in Lumen exposes the same three methods — `forward`, `init_state`,
`step` — with the same shapes, so a transformer block can hold either one
without knowing which it has.

```python
import torch
from lumen import (
    GatedDeltaNet, GatedDeltaNetConfig, HeadLayout,
    UndertowAttention, UndertowConfig,
)

d_model, batch, seq = 256, 2, 128

mixers = {
    "gdn": GatedDeltaNet(GatedDeltaNetConfig(
        d_model=d_model, layout=HeadLayout.shared_key(4), expand_k=2.0,
    )),
    "undertow": UndertowAttention(UndertowConfig(
        d_model=d_model, n_heads=4, window=32, plateau=24,
    )),
}

x = torch.randn(batch, seq, d_model)
for name, mixer in mixers.items():
    print(name, tuple(mixer(x).shape))
# gdn (2, 128, 256)
# undertow (2, 128, 256)
```

Both stream in constant memory, and the recurrence agrees with the parallel
path — so the check is written once and run against either:

```python
for name, mixer in mixers.items():
    mixer.eval()
    with torch.no_grad():
        full = mixer(x)
        # step() takes one position, (B, 1, d_model), and returns
        # (output, successor_state) -- thread the state through.
        state, outs = mixer.init_state(batch), []
        for t in range(seq):
            y, state = mixer.step(x[:, t : t + 1], state)
            outs.append(y)
        streamed = torch.cat(outs, dim=1)
    print(name, (full - streamed).abs().max().item())
# gdn       ~8e-07   <- fp32 round-off, not a modelling difference
# undertow  ~8e-07
```

## What's in it

**Instruments** — built first, because consolidating a layer means proving the
merged version reproduces what it replaces, on hardware that is not the machine
it was written on.

| | |
|---|---|
| [`lumen.probe`](./docs/HARDWARE_PROBE.md) | What a device *actually does*, measured — dtype reality-checking, Triton viability, autotune cost. CLI: `lumen-probe` |
| `lumen.bench` | Layer benchmarking that reports the **backward pass separately**, because a forward-only measurement can rank two kernels backwards — `benchmark_layer`, `compare`, `format_table` |

**Components** — each listed here only once its design record lands.

| | |
|---|---|
| [`lumen.undertow`](./docs/UNDERTOW.md) | Fixed-window causal attention, no positional encoding, optional graded boundary. Streaming with constant memory, opt-in Triton path. [Design record](./docs/design/UNDERTOW.md) |
| [`lumen.gdn`](./docs/GDN.md) | Gated DeltaNet — a fixed-size associative memory with a delta-rule write. Linear in sequence length, constant-memory generation, configurable head layout. [Design record](./docs/design/GATED_DELTANET.md) |
| [`lumen.block`](./docs/BLOCK.md) | A residual `Block` and a `Stack` of them — the level a result is actually read off. Modality-free, sub-layer instances rather than a config, and the depth-scaled init that cannot live any lower. [Design record](./docs/design/BLOCK.md) |

A roadmap of things that do not exist yet is a promise, not a description, so
there isn't one here.

---

## Why it exists

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
numbers side by side — nothing shared except the intent. Same architecture, two
implementations, no common surface. Incommensurable.

A shared core does not make every result comparable. A model trained on one
corpus and a model trained on another still are not, and no amount of shared
code fixes that. What it removes is the axis that was never supposed to be a
variable: when the layer is *the same object*, a difference in the numbers is a
difference in the experiment.

That is what this repository is for.

### The design record is the unit of value

Every component gets a design record **before** it gets code: what each existing
implementation uniquely contributed, what was decided and on what evidence, and
what is still open. Where two lineages disagreed and nobody has run the
comparison, the record says so and the experiment goes on the list —
consolidating is not a licence to pick.

That record, not the code, is the thing that stops the next rebuild.

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
instead of asserting version floors that are wrong.

- Research dependencies get pinned — that is what moves without warning.
- `torch` does not, because you already have a build matched to your driver, and
  pinning it is how a library becomes uninstallable on the hardware it claims to
  support.
- `triton` is not a dependency at all. It is a runtime capability that gets
  *measured*.
- The reference path for every component is fp32 and dependency-free. Faster
  paths are welcome behind the same interface, but they have to beat the
  reference on the machine in front of you — and that is a measurement, not an
  assumption.

---

## Documentation

| | |
|---|---|
| [`docs/HARDWARE_PROBE.md`](./docs/HARDWARE_PROBE.md) | What the probe reports and why it measures rather than asks |
| [`docs/UNDERTOW.md`](./docs/UNDERTOW.md) | Using Undertow — configuration, streaming, splicing into a trained stack |
| [`docs/BLOCK.md`](./docs/BLOCK.md) | Using Block and Stack — composition, streaming, initialisation, checkpoint keys |
| [`docs/GDN.md`](./docs/GDN.md) | Using Gated DeltaNet — head layout, widths, generation, subclassing |
| [`docs/design/UNDERTOW.md`](./docs/design/UNDERTOW.md) | Undertow design record — lineages, decisions, evidence, open questions |
| [`docs/design/GATED_DELTANET.md`](./docs/design/GATED_DELTANET.md) | Gated DeltaNet design record — same |

## Development

```bash
pip install -e '.[dev]'
pytest -m "not gpu and not triton"   # what CI runs: CPU-only, no Triton
pytest -m "gpu or triton"            # run locally before tagging
```

CI is CPU-only by decision rather than by limitation. What a hosted runner can
verify — the reference path, windowed-versus-dense equivalence,
step-versus-forward equivalence, and every piece of pure logic — is a large
share of the risk, and is why pure logic is kept separate from device access.
Anything CI cannot check is a reason to keep the reference path exact, not a
reason to lower the bar.

Reuse happens by **subclassing the layer**, not by importing the kernels. The
modules under `reference.py` are seams for subclasses and the substrate the test
suite checks against; they are deliberately not exported.

## Status

`0.3.0` — alpha, and versioned honestly. The *interface* is the thing this
library exists to protect, so it changes slowly and visibly; a merged component
is required to reproduce the outputs of what it replaces, to a stated tolerance,
before anything switches to it. Internals are fair game.

## License

MIT. See [LICENSE](./LICENSE).
