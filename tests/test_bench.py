"""Tests for the benchmarking harness."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from lumen.bench import BenchResult, benchmark_layer, compare, format_table

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _result(**kwargs) -> BenchResult:
    base = dict(label="x", dtype="float32", seq_len=1024, batch=4,
                params=1_000_000, fwd_ms=10.0, fwd_bwd_ms=30.0)
    base.update(kwargs)
    return BenchResult(**base)


def test_backward_is_the_difference():
    assert _result(fwd_ms=8.7, fwd_bwd_ms=24.5).bwd_ms == pytest.approx(15.8)


def test_tokens_per_s_uses_end_to_end_time():
    # 4 x 1024 tokens in 24.5 ms
    assert _result(fwd_ms=8.7, fwd_bwd_ms=24.5).tokens_per_s == pytest.approx(4 * 1024 / 0.0245)


def test_tokens_per_s_is_zero_when_timing_failed():
    assert _result(fwd_bwd_ms=0.0).tokens_per_s == 0.0


def test_format_table_ranks_against_fastest():
    rows = format_table([
        _result(label="torch-fp32", fwd_ms=8.7, fwd_bwd_ms=24.5),
        _result(label="triton-fp32", fwd_ms=5.7, fwd_bwd_ms=39.0),
    ])
    assert "torch-fp32" in rows and "triton-fp32" in rows
    assert "1.00x" in rows           # the winner
    assert "1.59x" in rows           # 39.0 / 24.5


def test_format_table_reports_failures_without_crashing():
    rows = format_table([
        _result(label="ok"),
        _result(label="broken", error="RuntimeError: no kernel"),
    ])
    assert "FAILED" in rows and "broken" in rows


def test_format_table_handles_empty():
    assert format_table([]) == "(no results)"


def test_format_table_groups_by_seq_len():
    rows = format_table([_result(seq_len=512), _result(seq_len=2048)])
    assert "512" in rows and "2048" in rows


def _mlp() -> nn.Module:
    return nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 64))


# The integration tests below run on the CPU.  Only the two that compare a
# device against itself need a card; the rest were gated behind `requires_cuda`
# because the default device made them unrunnable, not because they test
# anything about CUDA -- and gating the failures-are-data and
# fresh-layer-per-cell contracts on hardware meant CI never checked either.
#
# `device="cpu"` also asserts different things on different machines, which is
# the point: on a CPU-only box it proves the harness works without CUDA, and on
# a box with a card it proves a CPU cell takes the CPU path anyway.


def test_benchmark_layer_runs_on_cpu():
    """The instrument must work on the one device CI is guaranteed to have.

    It used to return an error row here -- `_time` synchronised the CUDA device
    unconditionally, and `ValueError: Expected a cuda device` was captured as
    data rather than raised.  A quiet failure in a measurement tool reads
    exactly like a measurement.
    """
    result = benchmark_layer(_mlp(), label="mlp", batch=2, seq_len=128, d_model=64,
                             device="cpu", warmup=1, reps=3)
    assert result.error is None, f"CPU benchmark failed: {result.error}"
    assert result.fwd_ms > 0
    assert result.fwd_bwd_ms >= result.fwd_ms
    assert result.tokens_per_s > 0


def test_compare_produces_a_real_table_on_cpu():
    """No cell may be a FAILED row, and the table must carry real numbers."""
    results = compare({"mlp": _mlp}, seq_lens=(64, 128), d_model=64,
                      batch=1, device="cpu", warmup=1, reps=2)
    assert all(r.error is None for r in results), [r.error for r in results]
    table = format_table(results)
    assert "FAILED" not in table
    assert "1.00x" in table          # a ratio implies at least one timed cell


def test_benchmark_captures_failure_as_a_row():
    """A layer with the wrong input width must not abort the table."""
    result = benchmark_layer(nn.Linear(999, 8), label="mismatched", batch=1, seq_len=8,
                            d_model=64, device="cpu", warmup=1, reps=1)
    assert result.error is not None


def test_compare_builds_a_fresh_layer_per_cell():
    seen: list[int] = []

    def build() -> nn.Module:
        seen.append(1)
        return nn.Linear(32, 32)

    results = compare({"linear": build}, seq_lens=(16, 32), d_model=32,
                      batch=1, device="cpu", warmup=1, reps=2)
    assert len(results) == 2
    assert len(seen) == 2, "compare() must construct a new layer per (seq_len, dtype) cell"


@requires_cuda
def test_benchmark_layer_on_a_real_module():
    result = benchmark_layer(_mlp(), label="mlp", batch=2, seq_len=128, d_model=64,
                             warmup=1, reps=3)
    assert result.error is None
    assert result.fwd_ms > 0
    assert result.fwd_bwd_ms >= result.fwd_ms
    assert result.tokens_per_s > 0
