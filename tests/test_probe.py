"""Tests for the capability probe.

The pure logic -- deriving architectural claims from a compute capability,
and deciding when a measurement contradicts them -- is deliberately separated
from device access so it can be tested anywhere, including on a CPU-only
machine and in CI.
"""

from __future__ import annotations

import pytest
import torch

from lumen.probe import (
    ArchFacts,
    DeviceInfo,
    DtypeReport,
    ProbeResult,
    TritonReport,
    arch_facts,
    format_report,
    probe,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


@pytest.mark.parametrize(
    ("capability", "generation", "tensor_cores", "bf16"),
    [
        ((6, 1), "Pascal", False, False),        # Tesla P40 -- the reference case
        ((7, 0), "Volta/Turing", True, False),
        ((7, 5), "Volta/Turing", True, False),
        ((8, 0), "Ampere/Ada", True, True),
        ((8, 9), "Ampere/Ada", True, True),
        ((9, 0), "Hopper", True, True),
    ],
)
def test_arch_facts(capability, generation, tensor_cores, bf16):
    facts = arch_facts(capability)
    assert facts.generation == generation
    assert facts.tensor_cores is tensor_cores
    assert facts.bf16_native is bf16
    assert facts.compute_capability == capability


def test_sm_is_numeric():
    assert arch_facts((6, 1)).sm == pytest.approx(6.1)
    assert arch_facts((9, 0)).sm == pytest.approx(9.0)


def test_pascal_claims_no_modern_features():
    """The P40 case that motivated this module."""
    facts = arch_facts((6, 1))
    assert not facts.tensor_cores
    assert not facts.bf16_native
    assert not facts.fp16_fast
    assert not facts.tf32_available


def test_emulated_detection():
    """Slower than fp32 means there is no hardware behind the API."""
    emulated = DtypeReport(dtype="bfloat16", runs=True, tflops=3.0, relative_to_fp32=0.33)
    native = DtypeReport(dtype="bfloat16", runs=True, tflops=40.0, relative_to_fp32=4.0)
    baseline = DtypeReport(dtype="float32", runs=True, tflops=10.0, relative_to_fp32=1.0)
    assert emulated.emulated
    assert not native.emulated
    assert not baseline.emulated


def test_emulated_is_false_without_measurement():
    assert not DtypeReport(dtype="bfloat16", runs=False, error="boom").emulated


def test_autotune_tax_scales_with_cold_compile():
    fast = TritonReport(available=True, cold_compile_s=0.5)
    slow = TritonReport(available=True, cold_compile_s=5.0)
    assert fast.autotune_tax_s is not None
    assert slow.autotune_tax_s == pytest.approx(10 * fast.autotune_tax_s)
    assert TritonReport(available=False).autotune_tax_s is None


@pytest.mark.parametrize(
    ("kb", "expected_fragment"),
    [(228, "large"), (164, "ample"), (100, "moderate"), (48, "small")],
)
def test_shared_memory_verdict(kb, expected_fragment):
    info = DeviceInfo(
        index=0, name="test", uuid="-", total_memory_mb=1024,
        shared_memory_per_block_kb=kb, multiprocessors=1, arch=arch_facts((6, 1)),
    )
    assert expected_fragment in info.shared_memory_verdict


def test_recommended_dtype_picks_fastest_measured():
    result = ProbeResult(
        device=DeviceInfo(0, "test", "-", 1024, 48, 1, arch_facts((6, 1))),
        dtypes=[
            DtypeReport("float32", True, tflops=11.8, relative_to_fp32=1.0),
            DtypeReport("float16", True, tflops=0.2, relative_to_fp32=0.02),
            DtypeReport("bfloat16", True, tflops=3.9, relative_to_fp32=0.33),
        ],
        triton=TritonReport(available=False, error="not installed"),
        torch_version="2.9.1",
        python_version="3.13.0",
    )
    assert result.recommended_dtype == "float32"


def test_marginally_faster_fp16_is_rejected_without_hardware():
    """The P40 case: fp16 measures ~8% faster because cuBLAS widens to fp32.

    Ranking on raw throughput would recommend it. It should not be recommended --
    the win is memory traffic and it does not survive fp16 loss scaling.
    """
    result = ProbeResult(
        device=DeviceInfo(0, "Tesla P40", "-", 22905, 48, 30, arch_facts((6, 1))),
        dtypes=[
            DtypeReport("float32", True, tflops=7.70, relative_to_fp32=1.00),
            DtypeReport("float16", True, tflops=8.32, relative_to_fp32=1.08),
            DtypeReport("bfloat16", True, tflops=4.78, relative_to_fp32=0.62),
        ],
        triton=TritonReport(available=True, version="3.5.1"),
        torch_version="2.9.1",
        python_version="3.13.1",
    )
    assert result.recommended_dtype == "float32"


def test_genuinely_faster_bf16_is_recommended_on_ampere():
    result = ProbeResult(
        device=DeviceInfo(0, "A100", "-", 40000, 164, 108, arch_facts((8, 0))),
        dtypes=[
            DtypeReport("float32", True, tflops=19.0, relative_to_fp32=1.00),
            DtypeReport("bfloat16", True, tflops=290.0, relative_to_fp32=15.3),
        ],
        triton=TritonReport(available=True, version="3.5.1"),
        torch_version="2.9.1",
        python_version="3.13.1",
    )
    assert result.recommended_dtype == "bfloat16"


def test_recommended_dtype_unknown_when_nothing_ran():
    result = ProbeResult(
        device=DeviceInfo(0, "test", "-", 1024, 48, 1, arch_facts((6, 1))),
        dtypes=[DtypeReport("float32", False, error="boom")],
        triton=TritonReport(available=False),
        torch_version="2.9.1",
        python_version="3.13.0",
    )
    assert result.recommended_dtype == "unknown"


def test_to_dict_is_json_serialisable():
    import json

    result = ProbeResult(
        device=DeviceInfo(0, "test", "-", 1024, 48, 1, arch_facts((6, 1))),
        dtypes=[DtypeReport("float32", True, tflops=11.8, relative_to_fp32=1.0)],
        triton=TritonReport(available=True, version="3.5.1", cold_compile_s=0.4),
        torch_version="2.9.1",
        python_version="3.13.0",
        notes=["a note"],
    )
    assert json.loads(json.dumps(result.to_dict()))["recommended_dtype"] == "float32"


@requires_cuda
def test_probe_runs_on_real_device():
    result = probe(index=0, skip_triton=True)
    assert result.device.name
    assert any(d.runs for d in result.dtypes)
    assert result.recommended_dtype != "unknown"
    assert format_report(result).startswith("=")


@requires_cuda
def test_probe_notes_are_actionable():
    result = probe(index=0, skip_triton=True)
    assert result.notes, "a probe with no notes has told the user nothing"
