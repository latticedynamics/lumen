"""
Lumen -- one place for the components worth keeping.

A consolidation library.  Unconventional architectures and training techniques
that have been rebuilt once per project get merged here a single time,
organised, optimised and documented.

Rebuilding the same layer is not primarily a waste of typing; it is a way of
**stranding findings in whichever copy happened to discover them.**  One copy
of a layer measures that its uncentred keys are costing it most of its
addressable directions and fixes it; the others never learn, and a sibling
copy goes on spending parameters to buy back the same directions a different
way.  Consolidation is what lets those two facts meet.

That only pays off if the merged code still runs later, and elsewhere, so it
is held to one discipline: **pin what moves, probe what cannot be pinned, and
degrade cleanly rather than assert a version floor.**  Reproducing a
linear-attention paper published in May 2026 was attempted that August and
failed at three separate walls -- deleted capability flags in a transitive
dependency, an unpinned git dependency whose required snapshot exists at no
public commit, and a hard-coded CUDA arch gate.  None of the three had
anything to do with the GPU.  A brand-new datacentre card hits all three
exactly as hard as a 2016 Tesla P40 does.

``triton`` is accordingly not a dependency; it is a runtime capability that
gets measured (see :mod:`lumen.probe`).  Every component's reference path is
fp32 and dependency-free, because that is the path that runs everywhere.

Start here::

    lumen-probe --all          # what does this machine actually do?

or from Python::

    from lumen import probe, format_report
    print(format_report(probe(index=0)))
"""

from __future__ import annotations

from lumen.bench import BenchResult, benchmark_layer, compare, format_table
from lumen.block import Block, BlockState
from lumen.gdn import (
    GatedDeltaNet,
    GatedDeltaNetConfig,
    GatedDeltaNetState,
    HeadLayout,
)
from lumen.nn import RMSNorm, SwiGLU, rms_norm
from lumen.undertow import UndertowAttention, UndertowConfig, UndertowState
from lumen.stack import Stack, StackState

# Imported for its side effect as much as its surface: the state containers are
# declared as pytree nodes here, so `torch.func` can traverse them.  Additive
# and numerically inert -- see :mod:`lumen.pytree`.
from lumen.pytree import REGISTERED, register_state_pytrees
from lumen.probe import (
    ArchFacts,
    DeviceInfo,
    DtypeReport,
    ProbeResult,
    TritonReport,
    arch_facts,
    device_info,
    format_report,
    measure_dtype,
    probe,
    probe_triton,
)

__version__ = "0.4.1"

__all__ = [
    "__version__",
    # probe
    "ArchFacts",
    "DeviceInfo",
    "DtypeReport",
    "ProbeResult",
    "TritonReport",
    "arch_facts",
    "device_info",
    "format_report",
    "measure_dtype",
    "probe",
    "probe_triton",
    # bench
    "BenchResult",
    "benchmark_layer",
    "compare",
    "format_table",
    # gdn
    "GatedDeltaNet",
    "GatedDeltaNetConfig",
    "GatedDeltaNetState",
    "HeadLayout",
    # nn
    "RMSNorm",
    "SwiGLU",
    "rms_norm",
    # block
    "Block",
    "BlockState",
    # stack
    "Stack",
    "StackState",
    # undertow
    "UndertowAttention",
    "UndertowConfig",
    "UndertowState",
    # pytree
    "REGISTERED",
    "register_state_pytrees",
]
