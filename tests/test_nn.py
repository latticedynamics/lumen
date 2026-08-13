"""Tests for the shared building blocks."""

from __future__ import annotations

import pytest
import torch

from lumen.gdn import GatedDeltaNet, GatedDeltaNetConfig
from lumen.nn import RMSNorm, rms_norm
from lumen.undertow import UndertowAttention, UndertowConfig


def test_rms_norm_gives_unit_root_mean_square():
    x = torch.randn(4, 16, 32) * 7.0
    out = rms_norm(x, torch.ones(32), eps=0.0)
    assert out.pow(2).mean(-1).allclose(torch.ones(4, 16), atol=1e-5)


def test_rms_norm_does_not_subtract_the_mean():
    """The whole difference from layer norm.  A constant input stays constant."""
    out = rms_norm(torch.full((2, 8), 3.0), torch.ones(8), eps=0.0)
    assert out.allclose(torch.ones(2, 8))


def test_rms_norm_is_scale_invariant_up_to_sign():
    """``rms_norm(c·v) = sign(c)·rms_norm(v)``.

    Undertow's `_out` docstring leans on this: it is what makes the two decay
    profile placements equivalent.  Exact only at ``eps=0``, which is why the
    test says so -- ``eps`` sits inside the square root next to ``mean(v²)``,
    so it stops being negligible once ``c²·mean(v²)`` approaches it.
    """
    v = torch.randn(3, 12)
    weight = torch.randn(12)
    base = rms_norm(v, weight, eps=0.0)
    for c in (4.0, 0.05, -2.5):
        scaled = rms_norm(c * v, weight, eps=0.0)
        expected = base if c > 0 else -base
        assert scaled.allclose(expected, atol=1e-5), f"failed at c={c}"


def test_rms_norm_weight_broadcasts_over_heads():
    """One `(d_head,)` vector shared across heads is the per-head layer's case."""
    x = torch.randn(2, 16, 4, 8)          # (B, T, H, D)
    out = rms_norm(x, torch.randn(8), eps=1e-5)
    assert out.shape == x.shape


def test_module_matches_the_function_in_fp32():
    x = torch.randn(2, 16, 32)
    norm = RMSNorm(32, eps=1e-5)
    with torch.no_grad():
        norm.weight.normal_()
    assert torch.equal(norm(x), rms_norm(x, norm.weight, 1e-5))


def test_module_computes_in_fp32_and_restores_the_input_dtype():
    """The reduction promotes; the layer stays transparent to its caller."""
    norm = RMSNorm(32)
    x = torch.randn(2, 16, 32, dtype=torch.float16)
    out = norm(x)
    assert out.dtype == torch.float16
    # Bit-identical to promoting by hand, which is what "computed in fp32" means.
    expected = rms_norm(x.float(), norm.weight, norm.eps).to(torch.float16)
    assert torch.equal(out, expected)


LAYERS = [
    lambda: GatedDeltaNet(GatedDeltaNetConfig(d_model=128, n_heads=4, expand_k=2.0)),
    lambda: UndertowAttention(UndertowConfig(d_model=128, n_heads=4, window=8)),
]


@pytest.mark.parametrize("build", LAYERS, ids=["gdn", "undertow"])
def test_out_path_promotes_and_never_demotes(build):
    """An fp64 caller must not be silently capped at fp32 by the output path.

    ``.float()`` casts to *exactly* fp32: a promotion for fp16 and bf16, a
    demotion for fp64.  The probe is two inputs that fp32 cannot tell apart --
    they differ by 1e-10 at unit scale, well under fp32's 1.2e-7 epsilon -- so
    an output path that rounds through fp32 returns bit-identical results for
    both, and one that promotes does not.

    Perturbing a single element rather than scaling: RMS normalisation is
    scale-invariant, so a uniform rescale would cancel and the probe would pass
    for the wrong reason.
    """
    torch.manual_seed(0)
    layer = build().double().eval()
    x = torch.randn(1, 8, 128, dtype=torch.float64)

    a = torch.randn(1, 8, 128, dtype=torch.float64)
    b = a.clone()
    b[0, 0, 0] += 1e-10

    assert torch.equal(a.float(), b.float()), "probe is invalid: fp32 can see the difference"

    with torch.no_grad():
        ya, yb = layer(a + x * 0), layer(b + x * 0)
    ya = ya[0] if isinstance(ya, tuple) else ya
    yb = yb[0] if isinstance(yb, tuple) else yb
    assert not torch.equal(ya, yb), "output path rounded an fp64 difference away"


@pytest.mark.parametrize("build", LAYERS, ids=["gdn", "undertow"])
def test_head_norm_stays_a_flat_parameter(build):
    """``head_norm``, not ``head_norm.weight`` -- a checkpoint compatibility test.

    The mixers share :func:`rms_norm` rather than holding an :class:`RMSNorm`,
    because a submodule prefixes its parameters and would rename this key.
    Every archived checkpoint, and Luminous's legacy remap, address it flat.
    Tidying the parameter into a submodule is the obvious-looking change this
    test exists to fail.
    """
    keys = build().state_dict()
    assert "head_norm" in keys
    assert "head_norm.weight" not in keys
