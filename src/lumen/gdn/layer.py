"""Gated DeltaNet — a fixed-size associative memory with a delta-rule write.

Each state is a matrix `M ∈ ℝ^(d_k × d_v)`, read as `o_tᵀ = q_tᵀ M_t` and
updated by::

    M_t  =  α_t (I − β_t k_t k_tᵀ) M_{t−1}  +  β_t k_t v_tᵀ

The bracketed factor removes whatever is currently stored at address `k_t`
before writing the new value there — that is the delta rule, and it is what
distinguishes this from a linear attention that only accumulates.  `α_t` forgets
everything uniformly.  With ``beta_max = 2`` the factor `1 − β` reaches negative
values, so the update is a reflection rather than only a contraction.

The design record is ``docs/design/GATED_DELTANET.md``.  Three points from it
are worth repeating where the code lives:

* **A head is a `(key group, value group)` pair**, not a row of a fixed grid.
  See :mod:`lumen.gdn.layout` — the short version is that a layer hard-wired to
  the crossed layout cannot express the ordinary one-key-one-value arrangement
  at all, which would leave it with no published reference point to check
  itself against.
* **`expand_k` has no default.**  The comparison that would justify one — key
  width against key centring, which are the same claim bought at very different
  prices — has not been run to conclusion.  Consolidating is not a licence to
  pick.
* **Centring is zero-initialised**, so a checkpoint from an uncentred model
  loads and behaves identically by construction.  There is a test for that;
  a construction claim deserves better than a comment.

Streaming works the way the rest of Lumen's components do — ``init_state`` /
``step`` / ``forward(..., state=, return_state=)`` — so a block can hold this or
Undertow without knowing which.  The state here is the memory itself plus the
short-conv cache, constant in generated length.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from lumen.gdn.layout import HeadLayout
from lumen.nn import rms_norm
from lumen.gdn.reference import (
    assign_values,
    chunk_gated_delta,
    recurrent_gated_delta,
)


@dataclass(frozen=True)
class GatedDeltaNetState:
    """Everything needed to continue a stream.

    ``memory`` is `(B, G_k, m, d_k, d_v)` — the associative memory itself, and
    the reason this layer is constant in generated length.

    ``conv`` holds the short causal convolution's left context, one cache per
    projection, or ``None`` when the layer has no convolution.  It is small and
    easy to forget, and forgetting it corrupts only the first few positions
    after a resume — which is exactly the kind of bug that survives a cursory
    test.

    Frozen — :meth:`GatedDeltaNet.step` returns a successor rather than mutating
    in place, so branching a stream (beam search, speculative decode) cannot
    leave two branches quietly sharing a buffer.
    """

    memory: torch.Tensor
    conv: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None


@dataclass(frozen=True)
class GatedDeltaNetConfig:
    """Configuration for :class:`GatedDeltaNet`, validated on construction.

    Args:
        d_model:  Residual stream width.
        layout:   Which key and value each state uses.  **The source of truth
                  for the head count** — ``n_heads`` is derived from it, so the
                  two cannot disagree.  See :mod:`lumen.gdn.layout`.
        expand_k: Total key width as a multiple of ``d_model``; per-state
                  ``d_k = expand_k · d_model / H``.  **Required.**  Since
                  ``rank(M) ≤ min(d_k, d_v)``, this is the dial that buys
                  non-interfering addresses — and it is also the dial that
                  ``centre`` may make unnecessary, which is why it gets no
                  default until that comparison is settled.
        expand_v: Total value width as a multiple of ``d_model``.  Buys
                  embedding room in front of the output gate rather than more
                  addresses.
        chunk_size: `C`, the chunkwise-parallel block.  A power of two dividing
                  the sequence length.
        conv_size: Width of the short causal depthwise convolution on q/k/v.
                  ``0`` disables it.  Local mixing, **not** a positional code —
                  it carries relative offsets within the window and nothing
                  absolute.
        beta_max: Write strength ceiling.  At 2 the update reaches reflections;
                  all contributing lineages agree here.
        max_chunk_decay: Ceiling on `−Σ log α` accumulated within one chunk,
                  which bounds `1/γ` and keeps the decay absorption in range.
        centre:   Subtract a learned per-head centre from `q` and `k` before
                  the l2 norm.  Zero-initialised, so turning it on is an exact
                  no-op until training moves it.
        norm_eps: Per-head output RMSNorm epsilon.
        dropout:  Applied to the layer output, after the output projection.
    """

    d_model: int
    n_heads: int
    expand_k: float = 2.0
    expand_v: float = 1.0
    layout: HeadLayout | None = None
    chunk_size: int = 64
    conv_size: int = 4
    beta_max: float = 2.0
    max_chunk_decay: float = 8.0
    centre: bool = False
    norm_eps: float = 1e-5
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.d_model < 1:
            raise ValueError(f"d_model must be >= 1, got {self.d_model}")
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be >= 1, got {self.n_heads}")

        # `layout` and `n_heads` over-determine each other on purpose: the
        # default is derived, and an explicit one is cross-checked.  That turns
        # "changed n_heads, forgot the layout" from a silently different model
        # into an error at construction.
        if self.layout is None:
            object.__setattr__(self, "layout", HeadLayout.shared_key(self.n_heads))
        elif self.layout.n_heads != self.n_heads:
            raise ValueError(
                f"layout has {self.layout.n_heads} heads but n_heads={self.n_heads}; "
                f"pass one or the other, or make them agree"
            )

        if self.chunk_size < 1 or self.chunk_size & (self.chunk_size - 1):
            raise ValueError(f"chunk_size must be a power of two, got {self.chunk_size}")
        if self.conv_size < 0:
            raise ValueError(f"conv_size must be >= 0, got {self.conv_size}")
        if self.beta_max <= 0:
            raise ValueError(f"beta_max must be > 0, got {self.beta_max}")
        if self.max_chunk_decay <= 0:
            raise ValueError(
                f"max_chunk_decay must be > 0, got {self.max_chunk_decay}"
            )
        if self.norm_eps <= 0:
            raise ValueError(f"norm_eps must be > 0, got {self.norm_eps}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

        for name, expand in (("expand_k", self.expand_k), ("expand_v", self.expand_v)):
            if expand <= 0:
                raise ValueError(f"{name} must be > 0, got {expand}")
            width = int(expand * self.d_model)
            if width % self.n_heads:
                raise ValueError(
                    f"{name} · d_model = {width} must divide evenly into "
                    f"{self.n_heads} heads"
                )
            if width // self.n_heads < 1:
                raise ValueError(
                    f"{name} = {expand} is too small for {self.n_heads} heads "
                    f"at d_model = {self.d_model}"
                )

    @property
    def d_k(self) -> int:
        """Per-state key width."""
        return int(self.expand_k * self.d_model) // self.n_heads

    @property
    def d_v(self) -> int:
        """Per-state value width."""
        return int(self.expand_v * self.d_model) // self.n_heads


class ShortConv(nn.Module):
    """Depthwise causal convolution over time, with its cache as the left pad.

    One code path for training and decode: the cache *is* the causal left
    padding, zeros on the first step.  No ``padding=`` argument, so there is no
    off-by-one to get wrong.
    """

    def __init__(self, channels: int, size: int) -> None:
        super().__init__()
        self.size = size
        self.conv = nn.Conv1d(channels, channels, size, groups=channels, bias=False)

    def forward(
        self, x: torch.Tensor, cache: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, channels = x.shape
        u = x.transpose(1, 2)
        if cache is None:
            cache = u.new_zeros(batch, channels, self.size - 1)
        u = torch.cat([cache, u], dim=-1)

        if u.shape[-1] == self.size:
            # Decode.  A depthwise convolution over exactly one output position
            # is a weighted sum of `size` values, and calling `conv1d` for it
            # costs the same as calling it for a whole training sequence --
            # the time is dispatch, not arithmetic, and grouped convolution
            # dispatch scales with the channel count.  Measured on one machine:
            # ~450x, and it is 95% of a decode step.
            #
            # This is arithmetically the same sum, but not bit-identical to
            # cuDNN/oneDNN's accumulation order -- order 1e-15 in fp64, six
            # orders inside the acceptance gate.  It is confined to decode on
            # purpose: `forward` keeps the library call, so training numerics
            # are untouched and every experimental record stays valid.
            y = (u * self.conv.weight.squeeze(1)).sum(-1, keepdim=True)
        else:
            y = F.conv1d(u, self.conv.weight, groups=channels)

        new_cache = u[..., -(self.size - 1):] if self.size > 1 else cache
        return y.transpose(1, 2), new_cache


class GatedDeltaNet(nn.Module):
    """A gated delta-rule associative memory, in the house streaming shape.

    Example::

        layout = HeadLayout.shared_key(8)
        config = GatedDeltaNetConfig(d_model=512, layout=layout, expand_k=2.0)
        mixer = GatedDeltaNet(config)
        y = mixer(x)                      # (B, T, d_model) -> (B, T, d_model)

    The output side — per-head RMSNorm, then a SiLU gate, then a projection —
    matches Undertow's, so a block can hold either mixer without knowing which
    one it has.

    Reuse is by subclassing.  :meth:`_features`, :meth:`_scan` and :meth:`_out`
    are the seams.
    """

    def __init__(self, config: GatedDeltaNetConfig) -> None:
        super().__init__()
        self.config = config

        d_model = config.d_model
        layout = config.layout
        n_heads, d_k, d_v = config.n_heads, config.d_k, config.d_v
        n_key_groups, n_value_groups = layout.n_key_groups, layout.n_value_groups

        self.q_proj = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.k_proj = nn.Linear(d_model, n_key_groups * d_k, bias=False)
        self.v_proj = nn.Linear(d_model, n_value_groups * d_v, bias=False)
        self.a_proj = nn.Linear(d_model, n_key_groups, bias=True)
        self.b_proj = nn.Linear(d_model, n_key_groups, bias=True)
        self.g_proj = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.o_proj = nn.Linear(n_heads * d_v, d_model, bias=False)

        self.head_norm = nn.Parameter(torch.ones(d_v))
        self.dropout = nn.Dropout(config.dropout)

        if config.conv_size > 0:
            self.q_conv = ShortConv(n_heads * d_k, config.conv_size)
            self.k_conv = ShortConv(n_key_groups * d_k, config.conv_size)
            self.v_conv = ShortConv(n_value_groups * d_v, config.conv_size)
        else:
            self.q_conv = self.k_conv = self.v_conv = None

        # Not created at all when off, so a layer without centring has exactly
        # the parameters it had before centring existed — and `torch.zeros`
        # draws no randomness, so the init RNG stream is identical either way.
        if config.centre:
            m = layout.states_per_key_group
            self.q_centre = nn.Parameter(torch.zeros(n_key_groups, m, d_k))
            self.k_centre = nn.Parameter(torch.zeros(n_key_groups, 1, d_k))
        else:
            self.q_centre = self.k_centre = None

        # Start with slow forgetting: softplus(bias) small => alpha near 1.
        nn.init.constant_(self.a_proj.bias, -3.0)
        nn.init.zeros_(self.b_proj.bias)

    # ── seams ─────────────────────────────────────────────────────────────

    def _features(
        self,
        x: torch.Tensor,
        conv_cache: tuple[torch.Tensor, ...] | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, ...] | None,
    ]:
        """`(B, T, d)` → `q, k, v, beta, log_alpha` in kernel shape, plus cache."""
        config = self.config
        layout = config.layout
        batch, seq_len, _ = x.shape
        n_key_groups = layout.n_key_groups
        m = layout.states_per_key_group
        d_k, d_v = config.d_k, config.d_v

        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        new_cache = None
        if self.q_conv is not None:
            cache = conv_cache or (None, None, None)
            q, cache_q = self.q_conv(q, cache[0])
            k, cache_k = self.k_conv(k, cache[1])
            v, cache_v = self.v_conv(v, cache[2])
            new_cache = (cache_q, cache_k, cache_v)

        # Heads are key-group-major (the layout guarantees it), so this is a
        # reshape and never a gather.
        q = F.silu(q).view(batch, seq_len, n_key_groups, m, d_k)
        k = F.silu(k).view(batch, seq_len, n_key_groups, 1, d_k)
        v = F.silu(v).view(batch, seq_len, layout.n_value_groups, d_v)

        # Recentre before normalising, never after: once the vectors are on the
        # sphere the cone they occupy is already fixed, and the cone is the
        # whole problem.
        if self.q_centre is not None:
            q = q - self.q_centre
            k = k - self.k_centre

        # l2 norm on the address side only.  Magnitude lives in beta, where
        # erase and write cannot disagree about it.
        q = F.normalize(q, dim=-1).permute(0, 2, 3, 1, 4)
        k = F.normalize(k, dim=-1).permute(0, 2, 3, 1, 4)
        v = assign_values(v, layout)

        beta = config.beta_max * torch.sigmoid(self.b_proj(x))
        beta = beta.permute(0, 2, 1).unsqueeze(2)

        log_alpha = -F.softplus(self.a_proj(x))
        log_alpha = log_alpha.permute(0, 2, 1).unsqueeze(2)
        # Bound 1/gamma within a chunk; the decay absorption divides by it.
        log_alpha = log_alpha.clamp(min=-config.max_chunk_decay / config.chunk_size)

        return q, k, v, beta, log_alpha, new_cache

    def _scan(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        log_alpha: torch.Tensor,
        memory: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The recurrence.  Override to swap in an accelerated kernel."""
        return chunk_gated_delta(
            q, k, v, beta, log_alpha, self.config.chunk_size, memory
        )

    def _out(self, o: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """`(B, G_k, m, T, d_v)` → per-head RMSNorm → SiLU gate → projection."""
        config = self.config
        batch, _, _, seq_len, _ = o.shape

        # Promote to *at least* fp32, rather than `.float()`, which casts to
        # exactly fp32 -- a promotion for fp16 and bf16 and a silent demotion
        # for fp64.  The reduction inside the norm genuinely needs the headroom
        # (a mean of squares underflows in fp16), but nothing here wants an fp64
        # caller quietly capped at fp32: it floors a whole-layer fp64 comparison
        # around 6e-8, which is fp32 epsilon and eight orders off what the
        # kernel's own oracle checks at.  Every dtype below fp64 is unaffected.
        #
        # The promotion stays at the call site rather than inside `rms_norm`
        # because it covers this whole output path, not the norm alone.
        o = o.permute(0, 3, 1, 2, 4).to(torch.promote_types(o.dtype, torch.float32))
        o = rms_norm(o, self.head_norm, config.norm_eps)
        o = o.reshape(batch, seq_len, config.n_heads * config.d_v).to(x.dtype)
        return self.dropout(self.o_proj(o * F.silu(self.g_proj(x))))

    # ── streaming ─────────────────────────────────────────────────────────

    def init_state(
        self,
        batch: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> GatedDeltaNetState:
        """An empty stream — a zeroed memory and a zeroed convolution cache.

        Device and dtype follow the layer's own parameters unless overridden,
        so ``layer.to("cuda").init_state(batch)`` needs no second argument and
        cannot silently produce a state on the wrong device.
        """
        config = self.config
        layout = config.layout
        reference = self.q_proj.weight
        device = reference.device if device is None else device
        dtype = reference.dtype if dtype is None else dtype

        memory = torch.zeros(
            batch,
            layout.n_key_groups,
            layout.states_per_key_group,
            config.d_k,
            config.d_v,
            device=device,
            dtype=dtype,
        )

        conv = None
        if config.conv_size > 0:
            widths = (
                config.n_heads * config.d_k,
                layout.n_key_groups * config.d_k,
                layout.n_value_groups * config.d_v,
            )
            conv = tuple(
                torch.zeros(
                    batch, width, config.conv_size - 1, device=device, dtype=dtype
                )
                for width in widths
            )

        return GatedDeltaNetState(memory=memory, conv=conv)

    def step(
        self, x: torch.Tensor, state: GatedDeltaNetState
    ) -> tuple[torch.Tensor, GatedDeltaNetState]:
        """One position — `(B, 1, d_model)` → output and the successor state."""
        if x.shape[1] != 1:
            raise ValueError(
                f"step() consumes one position at a time, got {x.shape[1]}; "
                f"use forward(x, state=..., return_state=True) for a chunk"
            )

        q, k, v, beta, log_alpha, conv = self._features(x, state.conv)
        o, memory = recurrent_gated_delta(
            q[..., 0, :],
            k[..., 0, :],
            v[..., 0, :],
            beta[..., 0],
            log_alpha[..., 0].exp(),
            state.memory,
        )
        return self._out(o.unsqueeze(-2), x), GatedDeltaNetState(
            memory=memory, conv=conv
        )

    # ── forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        *,
        state: GatedDeltaNetState | None = None,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, GatedDeltaNetState]:
        """`(B, T, d_model)` → `(B, T, d_model)`, optionally continuing a stream.

        Args:
            state: memory to continue from.  ``None`` starts from a zeroed
                memory, which is the beginning of a sequence.
            return_state: also return the state after consuming ``x``, so a
                prompt can be prefilled in one parallel pass and generation
                continued with :meth:`step`.
        """
        q, k, v, beta, log_alpha, conv = self._features(
            x, state.conv if state is not None else None
        )
        o, memory = self._scan(
            q, k, v, beta, log_alpha, state.memory if state is not None else None
        )
        y = self._out(o, x)

        if not return_state:
            return y
        return y, GatedDeltaNetState(memory=memory, conv=conv)

    def extra_repr(self) -> str:
        config = self.config
        return (
            f"d_model={config.d_model}, layout={config.layout.describe()}, "
            f"d_k={config.d_k}, d_v={config.d_v}, "
            f"expand_k={config.expand_k}, expand_v={config.expand_v}, "
            f"centre={config.centre}"
        )
