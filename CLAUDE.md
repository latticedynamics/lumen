# Lumen — working agreement

## What this repository is

A **consolidation library**. Components that have been rebuilt across several
projects get merged here once, organized, optimized, and documented.

It exists to buy two things, and the second is the one that is easy to forget
while writing code:

1. A finding discovered in one copy stops being stranded there.
2. **The projects can talk to each other.** Lumen is the common surface — when
   two projects share the same layer object, a difference in their numbers is a
   difference in their experiment, not an artefact of two implementations that
   drifted apart.

The second is why the interface and the numerics are load-bearing in a way that
internal cleverness is not. A faster implementation that changes the surface, or
the outputs, has spent the thing it was supposed to protect.

The unit of value is not the code. It is the **design record**: what each
existing implementation uniquely contributed, what was decided and on what
evidence, and what is still open. Code without that record just becomes the next
thing someone rebuilds.

Consequences that are easy to get wrong:

- **A component gets a design record before it gets code.** It is drafted in
  `notes/drafts/`, sanitised, and promoted to `docs/design/` when the component
  lands — see below.
- **Consolidation must be numerically transparent.** A merged component
  reproduces the outputs of what it replaces to a stated tolerance *before* any
  project switches to it. Silently changing numerics invalidates every
  experimental record downstream. Merge → verify → switch → evolve.
- **Do not canonise an untested default.** If two lineages disagree on a value
  and nobody has run the comparison, the design record says so and the
  experiment goes on the list. Consolidating is not a licence to pick.
- **Instruments stay separable from what they measure.** Diagnostic code does
  not get merged into the layer it inspects.

## docs/ versus notes/

Two audiences, and mixing them is what made the original survey unshippable.

**`notes/` is gitignored.** Nothing in it ships.

A document belongs in **`notes/`** if it contains any of:

- absolute paths into a machine (`~/Development/...`)
- unpublished results belonging to another repository
- second person addressed to a specific person ("open questions for Ken")
- work not yet done — sequencing, checklists, TODOs

A document belongs in **`docs/`** if a stranger who cloned the repo would be
worse off without it.

- `docs/` — how to use what is here
- `docs/design/` — why a component is shaped the way it is; the decisions and
  their evidence

**Documents get split, not filed.** A cross-repo survey almost always contains
both a public design rationale and a private work log. Extract the first, keep
the second.

### `notes/drafts/` — the staging area

A document does not go straight from a survey into `docs/`. It is drafted in
**`notes/drafts/`**, where it may freely carry machine paths, other projects'
results, and whatever else made it useful to write. It is sanitised and promoted
to `docs/` only when the component it describes actually lands.

```
survey  →  notes/drafts/THING.md  →  (sanitise)  →  docs/design/THING.md
                     ↑
        lives here as long as it needs to
```

Two things this buys, and the second is the one that is easy to learn the hard
way:

- A design record can be written *early*, while the reasoning is fresh, without
  that being a decision to publish it.
- **Un-publishing is not symmetric with publishing.** A repository that goes
  public exposes its whole history, not its tip — so a document that was
  committed and later removed is still there. Staging in a gitignored directory
  means the promotion is the only commit that ever happens.

When in doubt it goes in `notes/`. Promoting a draft is always available;
un-publishing one is not.

## Writing about hardware

The development bench is a Tesla P40 (SM 6.1, Pascal). That is a *useful worst
case* — it is where assumptions fail hardest — and it is **not** the audience.

- State the general principle first; the P40 is evidence for it, never the
  subject of it.
- Label measured numbers as coming from one machine. They are evidence, not
  specifications.
- Never write a capability check that consults a compute capability where a
  measurement is possible. `lumen.probe` exists precisely because the
  conventional SM floors in circulation are wrong.
- Reference paths are fp32 and dependency-free, because that is what runs
  everywhere. Faster paths are welcome behind the same interface; they must beat
  the reference on the target machine, measured.

## Dependencies

- **Pin research code.** Anything imported from someone else's research
  repository moves without warning. Pin it.
- **Do not pin `torch`.** The user has a build matched to their driver; pinning
  it is how a library becomes uninstallable on the hardware it claims to
  support.
- **`triton` is not a dependency.** It is a runtime capability that gets probed,
  and everything degrades cleanly without it.

## Code conventions

- Type hints on every parameter and return. `from __future__ import annotations`.
- `argparse` for CLIs, `pathlib` for paths, frozen dataclasses for config,
  validated on construction.
- **Failures are data, not crashes.** Probe and benchmark functions capture
  exceptions into their result so one broken implementation is a row in the
  table rather than an aborted run. Keep this — it is load-bearing for
  comparison tables.
- Pure logic separates from device access, so it stays testable on a CPU-only
  box and in CI (`arch_facts` is the pattern).
- Comments explain *why*, especially where a value was chosen against an
  obvious-looking alternative.
