"""Tests for the shared building blocks."""

from __future__ import annotations

import pytest
import torch

from lumen.gdn import GatedDeltaNet, GatedDeltaNetConfig, HeadLayout
from lumen.nn import RMSNorm, SwiGLU, rms_norm
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


def test_swiglu_is_the_arrangement_it_claims_to_be():
    """``down(up(x) * silu(gate(x)))``, spelled out.

    Worth pinning because the class exists to be *the same arrangement* the
    mixers already carry on their output path. If the two drift apart, the
    claim that a block adding one of these is duplicating a function the mixer
    already has stops being true, and that claim is load-bearing — it is why
    ``d_mlp`` has no default.
    """
    torch.manual_seed(0)
    mlp = SwiGLU(32, 64)
    x = torch.randn(2, 8, 32)
    expected = mlp.down_proj(mlp.up(x) * torch.nn.functional.silu(mlp.gate(x)))
    assert torch.equal(mlp(x), expected)


def test_swiglu_keys_match_the_lineage():
    """``up`` / ``gate`` / ``down_proj`` are checkpoint keys, not style.

    The asymmetry is inherited on purpose: renaming ``down_proj`` to match its
    two bare siblings would invalidate every archived state dict for a
    cosmetic gain.
    """
    assert set(SwiGLU(32, 64).state_dict()) == {
        "up.weight",
        "gate.weight",
        "down_proj.weight",
    }


def test_swiglu_declares_its_residual_write():
    mlp = SwiGLU(32, 64)
    assert mlp.residual_out_projections() == (mlp.down_proj,)


def test_swiglu_rejects_a_degenerate_width():
    with pytest.raises(ValueError):
        SwiGLU(32, 0)


LAYERS = [
    lambda: GatedDeltaNet(GatedDeltaNetConfig(d_model=128, layout=HeadLayout.shared_key(4), expand_k=2.0)),
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
    Every archived checkpoint, and every downstream legacy remap, addresses it
    flat.
    Tidying the parameter into a submodule is the obvious-looking change this
    test exists to fail.
    """
    keys = build().state_dict()
    assert "head_norm" in keys
    assert "head_norm.weight" not in keys


@pytest.mark.parametrize("build", LAYERS, ids=["gdn", "undertow"])
def test_residual_out_projections_are_the_layer_output(build):
    """The claim, checked against the code rather than against a name.

    ``residual_out_projections()`` says *this is where the layer's contribution
    leaves it*.  If that is true, zeroing what it returns must zero the layer's
    output exactly -- these projections are bias-free, so nothing downstream of
    them can put a value back.

    This is the test that survives a refactor.  Asserting the method returns
    something called ``o_proj`` would pass for a layer whose output path had
    moved on without it; this cannot.
    """
    torch.manual_seed(0)
    layer = build().eval()
    x = torch.randn(2, 16, 128)

    with torch.no_grad():
        assert layer(x).abs().max() > 0, "probe is invalid: layer was already silent"
        for projection in layer.residual_out_projections():
            for parameter in projection.parameters():
                parameter.zero_()
        assert torch.equal(layer(x), torch.zeros_like(x))


@pytest.mark.parametrize("build", LAYERS, ids=["gdn", "undertow"])
def test_residual_out_projections_match_the_name_sweep(build):
    """The bridge: asking returns exactly what grepping used to find.

    A depth-scaled initialiser conventionally sweeps ``named_parameters()`` for
    an ``o_proj.weight`` suffix.  That works until somebody renames an
    attribute, at which point it silently initialises one tensor fewer.  This
    asserts the two agree *today*, which is what lets a caller switch from the
    fragile mechanism to the checkable one without changing any weights.

    Delete this test when no caller sweeps names any more.  Until then it is the
    only thing tying the new answer to the old behaviour.
    """
    layer = build()
    declared = {
        id(parameter)
        for projection in layer.residual_out_projections()
        for parameter in projection.parameters()
    }
    swept = {
        id(parameter)
        for name, parameter in layer.named_parameters()
        if name.endswith("o_proj.weight")
    }
    assert declared and declared == swept


@pytest.mark.parametrize("build", LAYERS, ids=["gdn", "undertow"])
def test_residual_out_projections_adds_no_checkpoint_key(build):
    """Additive means additive: a method is not a parameter.

    Phase 1 of this release exists to land ahead of the block, and the reason it
    is allowed to land alone is that it cannot move a checkpoint.  A fresh layer
    must load an identically-configured layer's state dict under ``strict``.
    """
    source, destination = build(), build()
    destination.load_state_dict(source.state_dict(), strict=True)
