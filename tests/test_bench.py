"""Tests for the benchmarking harness."""

from __future__ import annotations

import re

import pytest
import torch
import torch.nn as nn

from lumen import bench
from lumen.bench import BenchResult, benchmark_layer, compare, format_table

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
requires_mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an MPS device")


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

    It also used to assert `fwd_bwd_ms >= fwd_ms` and flake at ~10% (#11).
    Raising `warmup` was measured to fix it -- 0/40 at `warmup=5` against 1/40
    at `warmup=1` -- but that buys a lower failure rate, not a sound assertion:
    the ordering of two independent timings remains a fact about the machine.
    So the flaky claim is dropped here rather than made rarer, and the
    deterministic one it was standing in for is tested directly instead.
    """
    result = benchmark_layer(_mlp(), label="mlp", batch=2, seq_len=128, d_model=64,
                             device="cpu", warmup=1, reps=3)
    assert result.error is None, f"CPU benchmark failed: {result.error}"
    assert result.fwd_ms > 0
    assert result.fwd_bwd_ms > 0
    assert result.tokens_per_s > 0
    # Same reasoning as the CUDA sibling: the ordering of two independent
    # timings is not assertable, only the promise that an inversion is never
    # rendered as a number.
    assert not re.search(r"-\d+\.\d", format_table([result]))


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
    """The measured fields are real; the ordering of the two is not asserted.

    This used to assert `fwd_bwd_ms >= fwd_ms` and flaked on a shared card.
    That is not a warmup problem and no amount of tuning fixes it: the two are
    independent timings of a sub-millisecond workload, so their ordering is a
    property of what else the device was doing, not of the layer.  A
    measurement tool whose own tests assert unresolvable orderings is arguing
    against itself.

    What *is* deterministic is that the result must never present the inversion
    as a number -- see `test_an_unresolved_backward_is_not_rendered_as_a_time`.
    """
    result = benchmark_layer(_mlp(), label="mlp", batch=2, seq_len=128, d_model=64,
                             warmup=1, reps=3)
    assert result.error is None
    assert result.fwd_ms > 0
    assert result.fwd_bwd_ms > 0
    assert result.tokens_per_s > 0
    # Whatever the ordering came out as, the table must not contain a negative
    # duration.  Matched by pattern rather than by cell index, so this keeps
    # meaning the same thing if a column is ever added.
    assert not re.search(r"-\d+\.\d", format_table([result]))


# ── waiting on the device that was asked ──────────────────────────────────
#
# Whether a timing waited for the device is not visible in the timing.  An
# unsynchronised cell returns a positive number of milliseconds and no error,
# so neither of the two tests below asserts on a duration: they record which
# device was waited on, which is deterministic and needs no hardware -- the
# wait is replaced by a counter, so this runs in CI on a box with no GPU.


@pytest.fixture
def waits(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every device wait `lumen.bench` issues, in order, with the wait itself removed."""
    seen: list[str] = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: seen.append("cuda"))
    monkeypatch.setattr(torch.mps, "synchronize", lambda *args, **kwargs: seen.append("mps"))
    return seen


@pytest.mark.parametrize(
    ("device", "expected"),
    [("cpu", []), ("cuda:0", ["cuda"]), ("mps", ["mps"])],
)
def test_sync_waits_on_the_device_being_measured(
    waits: list[str], device: str, expected: list[str]
) -> None:
    """Each queueing device has its own wait, and the CPU has none.

    The Metal row is the one this exists for.  It used to take the CPU branch,
    so a cell on `mps` timed the dispatch and reported it as the kernel -- an
    order of magnitude low on a real layer, and indistinguishable from a
    measurement.
    """
    bench._sync(torch.device(device))
    assert waits == expected


def test_every_timed_sample_is_synchronised(waits: list[str]) -> None:
    """One wait after warmup, then one per sample -- or the sample is of nothing.

    A single wait at the end would still be "synchronised" and would still be
    wrong: every sample but the last would close before its work had finished,
    and the median would be of dispatch times with one honest outlier.
    """
    bench._time(lambda: None, warmup=2, reps=5, device=torch.device("mps"))
    assert waits == ["mps"] * 6


@pytest.mark.mps
@requires_mps
def test_benchmark_layer_runs_on_mps():
    """The harness produces a real row on Metal, and not an error one.

    Deliberately says nothing about how long it took.  That the samples were
    waited for is held above, without a device; what needs the device is that
    the cell runs at all -- `.to("mps")`, the synchronise, and the cache
    release are each a call that only exists here.
    """
    result = benchmark_layer(_mlp(), label="mlp", batch=2, seq_len=128, d_model=64,
                             device="mps", warmup=1, reps=3)
    assert result.error is None, f"MPS benchmark failed: {result.error}"
    assert result.fwd_ms > 0
    assert result.fwd_bwd_ms > 0
    assert not re.search(r"-\d+\.\d", format_table([result]))


def test_an_unresolved_backward_is_not_rendered_as_a_time():
    """A backward under the noise floor renders as absent, not as a negative number.

    Constructed rather than measured, because the condition being tested is one
    a healthy machine will not reliably produce -- and a test that can only
    fail on a busy device is the flake this replaces.
    """
    inverted = BenchResult(label="mlp", dtype="float32", seq_len=128, batch=2,
                           params=8320, fwd_ms=5.06, fwd_bwd_ms=1.49)

    assert inverted.bwd_ms < 0, "the difference itself stays raw, not clamped"
    assert not inverted.bwd_resolved

    table = format_table([inverted])
    assert "<noise" in table
    assert "-3.6" not in table

    healthy = BenchResult(label="mlp", dtype="float32", seq_len=128, batch=2,
                          params=8320, fwd_ms=1.0, fwd_bwd_ms=3.0)
    assert healthy.bwd_resolved
    assert "2.0" in format_table([healthy])


def test_an_errored_row_never_claims_a_resolved_backward():
    """Failures are data, and a failed row's derived fields are not measurements."""
    failed = BenchResult(label="mlp", dtype="float32", seq_len=128, batch=2,
                         params=0, fwd_ms=float("nan"), fwd_bwd_ms=float("nan"),
                         error="RuntimeError: out of memory")
    assert not failed.bwd_resolved
