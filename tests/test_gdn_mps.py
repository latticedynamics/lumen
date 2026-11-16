"""On Metal, the reference is still the reference.

There is no MPS kernel in this package and this file is why that is a finding
rather than an omission.  The chunkwise fp32 path in
:mod:`lumen.gdn.reference` runs on an Apple-silicon GPU as it stands -- every
op it issues has a native Metal implementation -- and what is held here is that
it computes **the same function** there: scored against the fp64 sequential
oracle, forward and backward, it lands where the CPU does.

Two things about this device shape the tests.

* **There is no fp64 on Metal.**  The oracle cannot run where the path under
  test does, so every comparison is one draw made in fp64 on the CPU and cast
  down for each route.  The 1e-9 gate of the design record's §3.7 is
  unreachable here by construction; what applies is its second claim, that the
  fp32 path is the fp64 path evaluated in fp32.

* **A missing op need not raise.**  With ``PYTORCH_ENABLE_MPS_FALLBACK=1`` in
  the environment -- which a machine that has ever met a missing op tends to
  have set, permanently -- an operator Metal lacks is run on the CPU instead,
  with one warning and the right answer.  A change that introduced such an op
  would pass every numerical test in this file while moving part of the layer
  off the GPU.  So the absence of a fallback is tested on its own, in a child
  process with the variable removed.

The second half holds ``torch.compile``.  It is not a backend and Lumen has no
code for it: it is torch's own compilation of the reference, which unrolls the
chunk loop into one graph and fuses the elementwise work around the matmuls.
That makes it one algorithm reached by two routes in the strictest sense
available, and it is held to the same bar ``backend="fla"`` is -- no further
from the oracle than the path already trusted.

Marked ``mps`` and run locally before tagging; CI has no such device.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from collections.abc import Callable, Iterator

import pytest
import torch

from lumen.gdn import GatedDeltaNet, GatedDeltaNetConfig, HeadLayout
from lumen.gdn.reference import chunk_gated_delta, sequential_gated_delta

pytestmark = [
    pytest.mark.mps,
    pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="needs an MPS device"
    ),
]

# fp32 round-off on these shapes -- the same bound, for the same reason, as
# `tests/test_gdn_backend.py`.  Not tighter: the routes reassociate one sum, and
# a tolerance the arithmetic cannot meet is worse than a loose one (§3.7).
FP32_ROUNDOFF = 5e-6

NAMES = ("q", "k", "v", "beta", "log_alpha")

# (key groups, states per key group, per-state decay).  The diagonal case, a
# key shared across states, and that same sharing with every state on its own
# timescale -- which sends the kernel down solve-then-damp, the other of its
# two routes through the triangular solve.
ARRANGEMENTS = [
    pytest.param(4, 1, False, id="diagonal"),
    pytest.param(2, 4, False, id="shared-key"),
    pytest.param(2, 4, True, id="shared-key-per-state-decay"),
]

Kernel = Callable[..., tuple[torch.Tensor, torch.Tensor]]
Run = tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]


def _draw(groups: int, per_group: int, per_state_decay: bool) -> list[torch.Tensor]:
    """Kernel inputs in fp64 on the CPU: `B=2`, `T=256`, `d_k=64`, `d_v=128`.

    One draw, cast down for every route, so the routes are answering one
    question.  Drawing separately per device would compare two samples.
    """
    torch.manual_seed(0)
    batch, seq, d_k, d_v = 2, 256, 64, 128
    kind = dict(dtype=torch.float64)
    q = torch.nn.functional.normalize(
        torch.randn(batch, groups, per_group, seq, d_k, **kind), dim=-1
    )
    k = torch.nn.functional.normalize(
        torch.randn(batch, groups, 1, seq, d_k, **kind), dim=-1
    )
    v = torch.randn(batch, groups, per_group, seq, d_v, **kind)
    # beta spans [0, 2]: past 1 the update is a reflection, which is where
    # `beta_max = 2` puts a trained model and where two routes most easily part.
    beta = 2.0 * torch.rand(batch, groups, 1, seq, **kind)
    decay_axis = per_group if per_state_decay else 1
    log_alpha = -0.1 * torch.rand(batch, groups, decay_axis, seq, **kind)
    return [q, k, v, beta, log_alpha]


def _run(kernel: Kernel, inputs: list[torch.Tensor], seed: torch.Tensor) -> Run:
    """Outputs, final state, and the gradient of every input under one upstream signal."""
    leaves = [tensor.detach().clone().requires_grad_(True) for tensor in inputs]
    out, state = kernel(*leaves)
    grads = torch.autograd.grad(out, leaves, seed.to(out.device, out.dtype))
    return out, state, grads


def _distance(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.detach().cpu().double(), b.detach().cpu().double()
    return ((a - b).norm() / b.norm()).item()


def _chunked(*inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return chunk_gated_delta(*inputs, 64)


def _routes(
    groups: int, per_group: int, per_state_decay: bool, on_mps: Kernel
) -> tuple[Run, Run, Run]:
    """The fp64 oracle, the CPU fp32 chunkwise path, and `on_mps` on Metal."""
    base = _draw(groups, per_group, per_state_decay)
    torch.manual_seed(1)
    seed = torch.randn(2, groups, per_group, 256, 128, dtype=torch.float64)

    single = [tensor.float() for tensor in base]
    oracle = _run(sequential_gated_delta, base, seed)
    cpu = _run(_chunked, single, seed)
    mps = _run(on_mps, [tensor.to("mps") for tensor in single], seed)
    return oracle, cpu, mps


def _assert_no_worse(got: Run, trusted: Run, oracle: Run, what: str) -> None:
    """`got` reaches the oracle, and is no further from it than `trusted` is.

    The second half is the claim.  A bare threshold would still pass a route
    ten times further out than the one already relied on, if the threshold
    happened to be loose -- and it is deliberately loose.
    """
    for name, index in (("output", 0), ("final state", 1)):
        error = _distance(got[index], oracle[index])
        assert error < FP32_ROUNDOFF, f"{what}: {name} is {error:.3e} from the oracle"
        assert error < _distance(trusted[index], oracle[index]) * 10, (
            f"{what}: {name} is further from the oracle than the trusted path"
        )
    for name, grad, trusted_grad, oracle_grad in zip(
        NAMES, got[2], trusted[2], oracle[2]
    ):
        error = _distance(grad, oracle_grad)
        assert error < FP32_ROUNDOFF, f"{what}: d{name} is {error:.3e} from the oracle"
        assert error < _distance(trusted_grad, oracle_grad) * 10, (
            f"{what}: d{name} is further from the oracle than the trusted path"
        )


# ── the reference, on Metal ───────────────────────────────────────────────


@pytest.mark.parametrize(("groups", "per_group", "per_state_decay"), ARRANGEMENTS)
def test_mps_is_as_close_to_the_oracle_as_the_cpu_is(
    groups: int, per_group: int, per_state_decay: bool
) -> None:
    """Forward, final state, and all five gradients, against the fp64 oracle.

    The gradients are scored against the *oracle's* gradients rather than the
    CPU chunkwise path's.  That is available here and not in the fla test --
    the oracle is on the CPU either way -- and it is the stronger statement:
    not "the two devices agree" but "both differentiate the recurrence".
    """
    oracle, cpu, mps = _routes(groups, per_group, per_state_decay, _chunked)
    _assert_no_worse(mps, cpu, oracle, "mps chunkwise")


def _layer(**overrides: object) -> GatedDeltaNet:
    config: dict[str, object] = dict(
        d_model=128, layout=HeadLayout.shared_key(4), chunk_size=16
    )
    config.update(overrides)
    torch.manual_seed(0)
    return GatedDeltaNet(GatedDeltaNetConfig(**config))  # type: ignore[arg-type]


def test_the_layer_agrees_with_itself_across_devices() -> None:
    """One set of weights, both devices, through `forward` and through `step`.

    The kernel test above feeds the kernel directly.  This one goes through
    everything around it -- the projections, the grouped short convolution, the
    per-head norm -- each a separate Metal implementation that produces
    plausible output when wrong.  The length is not a multiple of the chunk, so
    the padded path is the one exercised.
    """
    on_cpu = _layer()
    on_mps = _layer().to("mps")
    on_mps.load_state_dict(on_cpu.state_dict())
    x = torch.randn(2, 72, 128)

    with torch.no_grad():
        want, want_state = on_cpu(x, return_state=True)
        got, got_state = on_mps(x.to("mps"), return_state=True)
        assert _distance(got, want) < FP32_ROUNDOFF
        assert _distance(got_state.memory, want_state.memory) < FP32_ROUNDOFF

        for position in range(4):
            token = torch.randn(2, 1, 128)
            want, want_state = on_cpu.step(token, want_state)
            got, got_state = on_mps.step(token.to("mps"), got_state)
            assert _distance(got, want) < FP32_ROUNDOFF, f"step {position} disagrees"


def test_nothing_needs_the_cpu_fallback() -> None:
    """Every op the layer issues has a Metal implementation -- none is borrowed.

    Run in a child process with ``PYTORCH_ENABLE_MPS_FALLBACK`` removed, because
    that is the only arrangement in which a missing op raises.  With it set, the
    op runs on the CPU, the numbers come out right, and every other test in
    this file passes over a layer that has partly left the GPU.  The variable is
    read once per process, so it cannot be cleared from inside this one.

    The child covers what reaches distinct ops: every layout, all three decay
    arrangements, centring, a padded length, the backward, the read cache and
    its re-read, and the decode step.
    """
    script = textwrap.dedent(
        """
        import torch
        from lumen.gdn import GatedDeltaNet, GatedDeltaNetConfig, HeadLayout

        configs = [
            dict(layout=HeadLayout.diagonal(4)),
            dict(layout=HeadLayout.shared_key(4)),
            dict(layout=HeadLayout.shared_value(4)),
            dict(layout=HeadLayout.crossed(2, 2), centre=True),
            dict(layout=HeadLayout.shared_key(4), decay="state"),
            dict(layout=HeadLayout.shared_key(4), decay="state_gated"),
            dict(layout=HeadLayout.diagonal(4), conv_size=0),
        ]
        for config in configs:
            layer = GatedDeltaNet(
                GatedDeltaNetConfig(d_model=128, chunk_size=16, **config)
            ).to("mps")
            x = torch.randn(2, 72, 128, device="mps", requires_grad=True)
            y, state, cache = layer(x, return_state=True, return_cache=True)
            (y.pow(2).mean() + layer.reread(x, cache).pow(2).mean()).backward()
            layer.read(x[:, :1], state)
            layer.step(x[:, :1], state)
        torch.mps.synchronize()
        """
    )
    environment = {
        name: value
        for name, value in os.environ.items()
        if name != "PYTORCH_ENABLE_MPS_FALLBACK"
    }
    child = subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert child.returncode == 0, (
        "the layer does not run on MPS without the CPU fallback:\n"
        + child.stderr[-2000:]
    )


# ── torch.compile, which is the reference by another route ────────────────


@pytest.fixture
def compiler() -> Iterator[None]:
    """A clean compile cache, and a recompile limit that raises instead of yielding.

    The second is not hygiene.  Past ``recompile_limit`` compilations of one
    function -- eight, by default, and every distinct sequence length is one --
    Dynamo stops compiling it and runs it eagerly, with a warning and no error.
    A test that compared a "compiled" route to an eager one would then be
    comparing eager to eager, agreeing to zero, and passing.  The limit is
    process-wide, so an unrelated earlier test could be what spends it.

    Skips when this torch build cannot compile for Metal at all: that is a gap
    in the build rather than a defect in the layer.
    """
    import torch._dynamo

    def trivial(x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x) * 2.0 + 1.0

    torch.compiler.reset()
    try:
        torch.compile(trivial, fullgraph=True)(torch.ones(4, device="mps"))
    except Exception as exc:  # noqa: BLE001 -- any refusal is the same skip
        torch.compiler.reset()
        pytest.skip(
            f"torch.compile cannot target MPS on torch {torch.__version__}: "
            f"{type(exc).__name__}"
        )

    config = torch._dynamo.config
    previous = config.fail_on_recompile_limit_hit
    config.fail_on_recompile_limit_hit = True
    try:
        yield
    finally:
        config.fail_on_recompile_limit_hit = previous
        torch.compiler.reset()


@pytest.mark.parametrize(("groups", "per_group", "per_state_decay"), ARRANGEMENTS)
def test_compiled_is_as_close_to_the_oracle_as_eager_is(
    compiler: None, groups: int, per_group: int, per_state_decay: bool
) -> None:
    """The compiled kernel against the fp64 oracle, with eager as the yardstick.

    ``fullgraph=True`` is what makes this a test of the compiled route: it
    raises on a graph break, so the kernel either ran as one compiled graph or
    the test failed.  Without it Dynamo may split the function and run the
    pieces between its graphs in the interpreter, and "compiled" would name
    whatever fraction happened to compile.
    """
    compiled = torch.compile(_chunked, fullgraph=True)
    oracle, _, got = _routes(groups, per_group, per_state_decay, compiled)

    base = _draw(groups, per_group, per_state_decay)
    torch.manual_seed(1)
    seed = torch.randn(2, groups, per_group, 256, 128, dtype=torch.float64)
    eager = _run(_chunked, [tensor.float().to("mps") for tensor in base], seed)

    _assert_no_worse(got, eager, oracle, "mps compiled")


def test_a_compiled_layer_is_the_layer(compiler: None) -> None:
    """Compiled and eager agree on everything `forward` can return.

    The state and the read cache come back as the dataclasses they are, not as
    tuples, because both are registered pytree nodes (`lumen.pytree`) -- and a
    cache built by the compiled pass can be re-read by the eager layer, which
    is the property that lets the two be mixed in one program.
    """
    layer = _layer().to("mps")
    compiled = torch.compile(layer, fullgraph=True)
    x = torch.randn(2, 80, 128, device="mps")

    with torch.no_grad():
        want, want_state, want_cache = layer(x, return_state=True, return_cache=True)
        got, got_state, got_cache = compiled(x, return_state=True, return_cache=True)

        assert type(got_state) is type(want_state)
        assert type(got_cache) is type(want_cache)
        assert _distance(got, want) < FP32_ROUNDOFF
        assert _distance(got_state.memory, want_state.memory) < FP32_ROUNDOFF
        assert _distance(layer.reread(x, got_cache), want) < FP32_ROUNDOFF

        # A second chunk, from the state the first one handed back.
        tail = torch.randn(2, 80, 128, device="mps")
        want_tail = layer(tail, state=want_state)
        got_tail = compiled(tail, state=got_state)
        assert _distance(got_tail, want_tail) < FP32_ROUNDOFF


def test_a_compiled_step_is_the_step(compiler: None) -> None:
    """Decode, compiled: the same outputs and the same successor memory."""
    layer = _layer().to("mps")
    step = torch.compile(layer.step, fullgraph=True)

    with torch.no_grad():
        want_state = got_state = layer.init_state(2)
        for position in range(6):
            token = torch.randn(2, 1, 128, device="mps")
            want, want_state = layer.step(token, want_state)
            got, got_state = step(token, got_state)
            assert _distance(got, want) < FP32_ROUNDOFF, f"step {position} disagrees"
        assert _distance(got_state.memory, want_state.memory) < FP32_ROUNDOFF
