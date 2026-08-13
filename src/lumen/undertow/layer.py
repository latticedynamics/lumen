"""Undertow — fixed-window causal attention, no positional encoding.

Each query attends to at most ``window`` past positions and nothing else.
Position is not encoded anywhere: there is no RoPE, no learned embedding, and
no distance term in the score.  What ordering information the layer has comes
from the causal window itself.

An optional ``plateau`` grades the window's boundary — full attention strength
out to that distance, then a cosine ramp toward the edge — entering as an
additive log-space bias before the softmax, which places it in the same family
as ALiBi and T5's relative position bias.  With ``plateau=None`` the boundary is
hard and the layer is ordinary sliding-window attention.

The design record is ``docs/design/UNDERTOW.md``; it carries the formal
definition, the decisions, and what is deliberately excluded.  Two points from
it are worth repeating where the code lives:

* **Absent keys are masked to ``-inf``, never zero-padded.**  A zeroed key still
  scores ``⟨q, 0⟩ = 0``, and ``exp(0) = 1`` — padding draws real attention
  weight.  Masking properly also makes this path agree with the dense oracle at
  *every* position, including the partial-window prefix, so no region has to be
  excluded from the equivalence test.
* **The profile enters in log space, and that is not an arbitrary choice
  between two options.**  Multiplying the profile in after the softmax produces
  weights that differ from these only by a positive per-(batch, head, position)
  scalar — which the per-head RMSNorm below then cancels exactly.  The two
  forms are the same layer.  The log-space form is preferred because it shares
  one masking mechanism with the validity mask and cannot drive the normaliser
  toward zero.  ``tests/test_undertow.py`` holds the layer to that equivalence.

Streaming works the way the rest of Lumen's components do — ``init_state`` /
``step`` / ``forward(..., state=, return_state=)`` — so a block can hold this or
Gated DeltaNet without knowing which.  The state here is a ring buffer of the
last ``W-1`` keys and values, constant in generated length.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from lumen.nn import rms_norm
from lumen.undertow import triton_kernels
from lumen.undertow.reference import (
    extend,
    log_decay_profile,
    window_validity,
    windowed_aggregate,
    windowed_scores,
)

BACKENDS = ("reference", "triton")


@dataclass(frozen=True)
class UndertowState:
    """Everything needed to continue a stream — a ring buffer and a count.

    ``keys`` and ``values`` are `(B, H, W-1, D)`: the most recent positions
    still within reach of the *next* query.  Constant in generated length,
    which is the property a fixed window exists to buy.

    Unfilled slots sit at the **front** (oldest end) and hold zeros.  They are
    masked before the softmax and their contents never reach an output; the
    count is what distinguishes them.

    Frozen — :meth:`UndertowAttention.step` returns a new state rather than
    mutating one, so a caller can branch a stream (beam search, speculative
    decode) without two branches quietly sharing a buffer.
    """

    keys: torch.Tensor
    values: torch.Tensor
    seen: int

    @property
    def filled(self) -> int:
        """How many buffer slots hold real positions."""
        return min(self.seen, self.keys.shape[2])


@dataclass(frozen=True)
class UndertowConfig:
    """Configuration for :class:`UndertowAttention`, validated on construction.

    Args:
        d_model:  Residual stream width.  Must divide evenly into ``n_heads``.
        n_heads:  Number of attention heads.
        window:   How many positions a query may reach, itself included.
                  Required — there is no defensible default, and no controlled
                  comparison exists to justify inventing one.
        plateau:  Distance out to which attention is at full strength, before
                  the cosine ramp begins.  ``None`` (the default) means a hard
                  window with no ramp — safe to default to precisely because it
                  is the case where the profile is identically 1.
        dropout:  Applied to the layer output, after the output projection.
        zero_init: Start ``o_proj`` at zero, making the layer an exact identity
                  no-op.  This is what makes it safe to splice a fresh Undertow
                  layer into an already-trained stack: at step 0 the spliced
                  model is bit-identical to the checkpoint it came from, so
                  nothing is destroyed and the layer earns its contribution from
                  zero rather than injecting noise into a converged residual
                  stream.
        eps:      Per-head RMSNorm epsilon.
        backend:  ``"reference"`` (default) or ``"triton"``.  **Opt-in, and
                  deliberately not auto-detected.**  A fast path that switches
                  itself on whenever a package happens to be importable would
                  mean two projects sharing this layer are not running the same
                  object — and then a difference in their numbers stops being a
                  difference in their experiment, which is the one thing this
                  library exists to prevent.  Ask for it, measure it, keep it.
    """

    d_model: int
    n_heads: int
    window: int
    plateau: int | None = None
    dropout: float = 0.0
    zero_init: bool = False
    eps: float = 1e-6
    backend: str = "reference"

    def __post_init__(self) -> None:
        if self.d_model < 1:
            raise ValueError(f"d_model must be >= 1, got {self.d_model}")
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be >= 1, got {self.n_heads}")
        if self.d_model % self.n_heads:
            raise ValueError(
                f"d_model ({self.d_model}) must divide evenly into "
                f"n_heads ({self.n_heads})"
            )
        if self.window < 1:
            raise ValueError(f"window must be >= 1, got {self.window}")
        if self.plateau is not None and not 0 <= self.plateau < self.window:
            raise ValueError(
                f"plateau must satisfy 0 <= plateau < window, got "
                f"plateau={self.plateau}, window={self.window}"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {self.eps}")
        # Validate the *name* here and availability in the layer: this dataclass
        # is pure logic and stays testable on a machine with no GPU.
        if self.backend not in BACKENDS:
            raise ValueError(
                f"backend must be one of {BACKENDS}, got {self.backend!r}"
            )

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads


class UndertowAttention(nn.Module):
    """Fixed-window causal attention with an optional graded boundary.

    Example::

        config = UndertowConfig(d_model=384, n_heads=8, window=32, plateau=24)
        attn = UndertowAttention(config)
        y = attn(x)                       # (B, T, d_model) -> (B, T, d_model)

    The output side — per-head RMSNorm, then a SiLU gate, then a projection —
    mirrors Lumen's Gated DeltaNet, so a block can hold either mixer without
    knowing which one it has.  That is a deliberate interface commitment, not
    incidental structure: the profile-placement equivalence documented above
    depends on the attention output passing through a scale-invariant per-head
    normalisation.  A subclass that replaces :meth:`_out` gives that up.

    Reuse is by subclassing.  :meth:`_window_scores`, :meth:`_window_weights`,
    :meth:`_window_aggregate` and :meth:`_out` are the seams.
    """

    def __init__(self, config: UndertowConfig) -> None:
        super().__init__()

        # An explicit request that cannot be honoured is an error, not a silent
        # downgrade.  Degrading cleanly is for capabilities nobody asked for.
        if config.backend == "triton" and not triton_kernels.HAS_TRITON:
            raise RuntimeError(
                'backend="triton" was requested but triton did not import. '
                "Install it, or use the reference backend."
            )

        self.config = config

        d_model = config.d_model
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.g_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.head_norm = nn.Parameter(torch.ones(config.d_head))
        self.dropout = nn.Dropout(config.dropout)
        self.scale = math.sqrt(config.d_head)

        # Not persistent: it is a pure function of (window, plateau), both of
        # which live in the config, so serialising it would only create a way
        # for a checkpoint to disagree with its own configuration.
        self.register_buffer(
            "log_profile",
            log_decay_profile(config.window, config.plateau),
            persistent=False,
        )

        if config.zero_init:
            nn.init.zeros_(self.o_proj.weight)

    # ── seams ─────────────────────────────────────────────────────────────

    def _project(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """`(B, T, d)` → q, k, v as `(B, H, T, D)` plus the gate as `(B, T, d)`."""
        batch, seq_len, _ = x.shape
        shape = (batch, seq_len, self.config.n_heads, self.config.d_head)

        def heads(projected: torch.Tensor) -> torch.Tensor:
            return projected.view(shape).transpose(1, 2)

        return (
            heads(self.q_proj(x)),
            heads(self.k_proj(x)),
            heads(self.v_proj(x)),
            self.g_proj(x),
        )

    def _window_scores(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        window: int,
        *,
        prefix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`(B, H, T, W)` raw scores, on whichever backend is configured.

        The Triton path takes the extended key array rather than a prefix: the
        extension is a differentiable cat or pad, so autograd routes the
        gradient back to the history and the chunk on its own, and the kernel's
        inner loop needs no bounds check.
        """
        if self.config.backend == "triton" and triton_kernels.usable(q):
            return triton_kernels.windowed_scores(
                q, extend(k, window, prefix), window, self.scale
            )
        return windowed_scores(q, k, window, self.scale, prefix=prefix)

    def _window_aggregate(
        self,
        weights: torch.Tensor,
        v: torch.Tensor,
        window: int,
        *,
        prefix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`(B, H, T, D)` weighted sum, on whichever backend is configured.

        Falls back to the reference on non-CUDA tensors — that is a device
        without such a path, not a missing capability, so it is not an error.
        """
        if self.config.backend == "triton" and triton_kernels.usable(v):
            return triton_kernels.windowed_aggregate(
                weights, extend(v, window, prefix), window
            )
        return windowed_aggregate(weights, v, window, prefix=prefix)

    def _window_weights(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        window: int,
        *,
        prefix: torch.Tensor | None = None,
        n_prefix: int = 0,
    ) -> torch.Tensor:
        """`(B, H, T, W)` attention weights — scores, bias, mask, softmax.

        Both the decay profile and the structural validity mask are additive
        pre-softmax terms, so there is one masking path here and not two.
        """
        scores = self._window_scores(q, k, window, prefix=prefix)

        # When T < window the window is clamped, but the profile is *sliced*,
        # never rebuilt: p(δ) is a property of the configured window and must
        # not change shape because a batch happened to be short.  The last
        # `window` entries carry distances W-1 … 0, which is exactly the range
        # a clamped window spans.
        scores = scores + self.log_profile[-window:].to(scores.dtype)

        valid = window_validity(
            q.shape[2], window, n_prefix=n_prefix, device=q.device
        )
        scores = scores.masked_fill(~valid, float("-inf"))

        return torch.softmax(scores, dim=-1)

    def _out(self, o: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """`(B, H, T, D)` → per-head RMSNorm → SiLU gate → projection.

        The RMSNorm is scale-invariant, which is what makes the two profile
        placements equivalent (see the module docstring).  Changing this method
        changes that guarantee.
        """
        batch, _, seq_len, _ = o.shape
        o = o.transpose(1, 2).reshape(
            batch, seq_len, self.config.n_heads, self.config.d_head
        ).float()
        o = rms_norm(o, self.head_norm, self.config.eps)
        o = o.reshape(batch, seq_len, self.config.d_model).to(gate.dtype)
        return self.dropout(self.o_proj(o * F.silu(gate)))

    # ── streaming ─────────────────────────────────────────────────────────

    def init_state(
        self,
        batch: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> UndertowState:
        """An empty stream — a zeroed `(B, H, W-1, D)` ring buffer, ``seen=0``.

        Device follows the layer's own parameters unless overridden, so
        ``layer.to("cuda").init_state(batch)`` needs no second argument and
        cannot silently produce a state on the wrong device.  It used not to,
        and the failure mode was the bad one: correct on a CPU-only box and in
        CI, a device mismatch on the first decode of a GPU run.

        Dtype does **not** follow the module, and that asymmetry with
        :meth:`GatedDeltaNet.init_state` is deliberate: the buffer is fp32
        because the kernels are, whatever the surrounding module has been cast
        to.  ``dtype`` is offered for callers who have measured that something
        else works on their hardware.
        """
        config = self.config
        shape = (batch, config.n_heads, config.window - 1, config.d_head)
        device = self.q_proj.weight.device if device is None else device
        zeros = torch.zeros(shape, device=device, dtype=dtype or torch.float32)
        return UndertowState(keys=zeros, values=zeros.clone(), seen=0)

    def _next_state(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        seen: int,
    ) -> UndertowState:
        """Keep the newest ``W-1`` positions, front-padding if there are fewer.

        Front-padding matches :meth:`init_state`'s layout — unfilled slots at
        the oldest end — so ``seen`` alone always says which slots are real,
        whether the state came from a fresh start, a prefill, or a step.
        """
        buffer_len = self.config.window - 1
        available = keys.shape[2]

        if buffer_len == 0:
            return UndertowState(keys[:, :, :0], values[:, :, :0], seen)
        if available >= buffer_len:
            return UndertowState(
                keys[:, :, -buffer_len:].contiguous(),
                values[:, :, -buffer_len:].contiguous(),
                seen,
            )
        pad = buffer_len - available
        return UndertowState(
            F.pad(keys, (0, 0, pad, 0)), F.pad(values, (0, 0, pad, 0)), seen
        )

    def step(
        self, x: torch.Tensor, state: UndertowState
    ) -> tuple[torch.Tensor, UndertowState]:
        """One position — `(B, 1, d_model)` → output and the successor state.

        Scores against the whole `W`-slot buffer in a single matmul rather than
        going through the windowed kernels: at `T = 1` the offset loop would be
        `W` launches to do one small product.

        Slots the stream has not reached yet sit at the front and are masked
        by *count*, where :meth:`forward` masks by positional validity.  Two
        different mechanisms reaching the same answer is exactly why the
        agreement between them is worth a test rather than an assertion.
        """
        if x.shape[1] != 1:
            raise ValueError(
                f"step() consumes one position at a time, got {x.shape[1]}; "
                f"use forward(x, state=..., return_state=True) for a chunk"
            )

        window = self.config.window
        q, k, v, gate = self._project(x)

        keys = torch.cat([state.keys, k.float()], dim=2)
        values = torch.cat([state.values, v.float()], dim=2)

        scores = torch.matmul(q.float(), keys.transpose(-2, -1)) / self.scale
        scores = scores + self.log_profile.to(scores.dtype)

        n_valid = min(state.seen + 1, window)
        if n_valid < window:
            invalid = torch.zeros(window, dtype=torch.bool, device=x.device)
            invalid[: window - n_valid] = True
            scores = scores.masked_fill(invalid, float("-inf"))

        o = torch.matmul(torch.softmax(scores, dim=-1), values)

        return self._out(o, gate), self._next_state(
            keys, values, state.seen + 1
        )

    # ── forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        *,
        state: UndertowState | None = None,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, UndertowState]:
        """`(B, T, d_model)` → `(B, T, d_model)`, optionally continuing a stream.

        Args:
            state: history to attend into. ``None`` starts at the beginning of
                a sequence. Passing one makes this a *chunked* prefill —
                queries near the start of ``x`` reach back into the buffer
                exactly as far as the window allows.
            return_state: also return the state after consuming ``x``, so a
                prompt can be prefilled in one parallel pass and generation
                continued with :meth:`step`.

        A sequence shorter than the window clamps rather than raising: a
        windowed path has no reason to reject a short batch, and slicing the
        profile keeps every distance meaning what it meant. With a ``state``
        there is nothing to clamp — the history supplies the reach.
        """
        seq_len = x.shape[1]
        q, k, v, gate = self._project(x)
        q, k, v = q.float(), k.float(), v.float()

        if state is None:
            window = min(self.config.window, seq_len)
            prefix_k = prefix_v = None
            n_prefix = 0
            seen = 0
        else:
            window = self.config.window
            prefix_k, prefix_v = state.keys, state.values
            n_prefix = state.filled
            seen = state.seen

        weights = self._window_weights(
            q, k, window, prefix=prefix_k, n_prefix=n_prefix
        )
        o = self._window_aggregate(weights, v, window, prefix=prefix_v)
        y = self._out(o, gate)

        if not return_state:
            return y

        keys = k if prefix_k is None else torch.cat([prefix_k, k], dim=2)
        values = v if prefix_v is None else torch.cat([prefix_v, v], dim=2)
        return y, self._next_state(keys, values, seen + seq_len)

    def extra_repr(self) -> str:
        config = self.config
        return (
            f"d_model={config.d_model}, n_heads={config.n_heads}, "
            f"window={config.window}, plateau={config.plateau}"
        )
