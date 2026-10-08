"""Sparse Delta Memory — a gated delta rule over a large, sparsely addressed table.

Per head, a table of `N` slots, read as `y_t = M_tᵀ q_t` and updated by::

    M_t  =  (I − β_t k_t k_tᵀ) Λ_t M_{t−1}  +  β_t k_t v_tᵀ

where `k_t` and `q_t` are nonzero only on the `W` slots position `t` writes and
the `R` it reads, chosen by product keys, and `Λ_t` decays the written slots and
nothing else.  Per-token arithmetic is `O((W+R)·d_v)` whatever `N` is, and the
only parameters that grow with `N` are the two address projections, as `√N`.
**State size is decoupled from parameter count and from compute** — which is the
reason to have this layer.  One cost is not: :meth:`SparseDeltaMemory.step`
returns a new table per stream, `O(B·N·d_v)` a step, unless its state is
donated — see :class:`SparseDeltaMemoryState`.

The design record is drafted ahead of promotion.  Points from it worth
repeating where the code lives:

* **A slot nobody writes to is frozen, decay included.**  Forgetting is
  triggered by writes, not by time passing, so a slot's half-life is counted in
  writes to it.  The kernels hold this bit for bit, and there is a test.
* **`initial_memory` is required.**  ``"zero"`` starts every stream from an
  empty table; ``"learned"`` starts it from a trained one, and because the
  table is frozen wherever it is not written, what was learned survives the
  context rather than being decayed out of it.  A learned table read through
  product keys is a memory layer, so the second option is a memory layer and a
  fast-weight memory sharing one table — a different model, not a better
  setting of the first.  Two models behind one field is exactly what GDN's
  ``layout`` argument refuses to default, for the same reason.
* **The learned table is zero-initialised**, so at construction the two options
  are the same layer, exactly.  It still trains from zero: `∂y/∂M₀[n] = q_n`
  wherever a read lands.
* **The decay is carried relatively, per slot**, because it is diagonal and
  does not factor out of the chunkwise solve the way GDN's scalar decay does.
  See :mod:`lumen.sdm.reference`.

Streaming works the way the rest of Lumen's components do — ``init_state`` /
``step`` / ``forward(..., state=, return_state=)`` — so a block can hold this,
Gated DeltaNet or Undertow without knowing which.  The state is the table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from lumen.nn import rms_norm
from lumen.sdm import triton_kernels
from lumen.sdm.reference import (
    _transforms_active,
    chunk_sparse_delta,
    read_sparse_delta,
    recurrent_sparse_delta,
)

INITIAL_MEMORY = ("zero", "learned")
KEY_NORMS = ("softmax", "l2")
DECAY_WEIGHTINGS = ("write_set", "key")
BACKENDS = ("reference", "triton")


@dataclass(frozen=True)
class SparseDeltaMemoryState:
    """Everything needed to continue a stream: the table.

    ``memory`` is `(B, H, N, d_v)`.  Straight out of
    :meth:`SparseDeltaMemory.init_state` it is a **broadcast view** of one
    `(H, N, d_v)` table — no stream costs a copy until its first write, and that
    write is the copy the successor rule below makes anyway.

    Frozen — :meth:`SparseDeltaMemory.step` returns a successor rather than
    mutating in place, so branching a stream cannot leave two branches quietly
    sharing a buffer.  **Here that guarantee has a price** that it does not have
    for Gated DeltaNet: the successor is a full table per stream, `O(B·N·d_v)`
    per head per step against `O(B·(W+R)·d_v)` of actual work.  At `B = 1` the
    step is launch-bound and the copy hides; at a large batch it is most of the
    step, at tables nobody would call large.

    ``step(x, state, donate=True)`` waives it: the successor is written into
    the donated table, and the donated state must not be read again.  The
    default keeps the guarantee.  Design record §3.6 has the measurements.
    """

    memory: torch.Tensor


def _writable_in_place(memory: torch.Tensor) -> bool:
    """May a donated table be written where it stands?

    Donation says the caller is done with the state, not that its table is the
    stream's own.  It is not when it is a view: ``init_state`` hands out a
    broadcast, and under ``"learned"`` that broadcast shares the parameter's
    storage -- at `B = 1` it does not even overlap, so a write would go through
    and change the learned table.  Nor under a ``torch.func`` transform, or
    while autograd records the table, where an in-place write is refused or
    breaks the gradient.  Each of those steps by copy instead.
    """
    if _transforms_active():
        return False
    if torch.is_grad_enabled() and memory.requires_grad:
        return False
    return memory._base is None


@dataclass(frozen=True)
class SparseDeltaMemoryConfig:
    """Configuration for :class:`SparseDeltaMemory`, validated on construction.

    Args:
        d_model:  Residual stream width.
        n_heads:  Independent tables, each with its own address space.
                  ``d_v = d_model / n_heads``.
        n_slots:  `N`, slots **per head**, a perfect square — product keys
                  address it as `√N × √N`.  The knob that grows the state
                  without growing compute: per-token arithmetic does not
                  depend on it, and the address projections grow as `√N`.
                  Decode's successor copy does, `O(B·N·d_v)` a step, unless
                  the state is donated — see :class:`SparseDeltaMemoryState`.
        initial_memory: ``"zero"`` or ``"learned"``, and **required** — see the
                  module docstring.  ``"learned"`` adds `H · N · d_v`
                  parameters, reported like any others; the paper this layer
                  comes from leaves them out of its parameter counts, and an
                  iso-parameter comparison that does so is an iso-*active*-
                  parameter comparison.
        n_writes: `W`, slots written per position.
        n_reads:  `R`, slots read per position.  Neither default has been
                  compared by anyone: the paper states 64, its released configs
                  use 128.
        key_norm: How the selected scores become weights, on both sides.

                  * ``"softmax"`` — the paper's.  Weights on the simplex, so
                    ``‖k‖² = Σp²`` and the delta correction along `k` is
                    `β‖k‖²` — at the access statistics the paper reports, an
                    order of magnitude weaker than Gated DeltaNet's.
                  * ``"l2"`` — the selected scores normalised to a unit vector:
                    signed weights, ``‖k‖ = 1``, and the delta rule at full
                    strength.  **The experiment for whether this layer is a
                    delta rule at all**, or mostly a gated overwrite; shipped
                    for that reason, untested.
        decay_weighting: How a write's decay spreads over its `W` slots.

                  * ``"write_set"`` — every written slot decays by the full
                    `α`.  The paper's.
                  * ``"key"`` — slot `n` decays by `α^{k_n}`, so a slot selected
                    at weight 0.001 barely decays.  Removes the discontinuity at
                    the top-`W` boundary; untested.  **Refused under**
                    ``key_norm="l2"``: a negative weight would make a
                    positive log-decay, which is growth.
        beta_max: Write strength ceiling, **defaulting to 2 and untested**; the
                  paper's is 1.  Past 2 is refused: the UT/WY solve advances by
                  `(I − β k kᵀ)`, norm-preserving for `β‖k‖² ∈ [0, 2]`, and every
                  key here has `‖k‖ ≤ 1`.
        chunk_size: `C`, the chunkwise block.  Numerically inert — there is a
                  test — and a memory dial: training keeps `O(T·C·(W+R))` per
                  sequence.  The default is a guess to be measured.
        norm_eps: Per-head output RMSNorm epsilon.
        dropout:  Applied to the layer output, after the output projection.
        backend:  ``"reference"`` (default) or ``"triton"``.  **Opt-in, and
                  not auto-detected**, for the reason Undertow's
                  ``docs/design/UNDERTOW.md`` §3.4 gives: two projects sharing
                  this layer must be running the same code.  ``"triton"`` is
                  Lumen's own kernels (:mod:`lumen.sdm.triton_kernels`) for
                  the table-free terms and the walk over the table: the same
                  arithmetic to fp32 round-off, not bit-identical to the
                  reference, and its backward is bit-identical run to run
                  without deterministic algorithms (no atomics).  Where they
                  cannot run -- a CPU tensor, fp64, a ``torch.func``
                  transform with gradients, a ``chunk_size`` above 64 -- the
                  reference runs instead.
    """

    d_model: int
    n_heads: int
    n_slots: int
    initial_memory: Literal["zero", "learned"]
    n_writes: int = 64
    n_reads: int = 64
    key_norm: Literal["softmax", "l2"] = "softmax"
    decay_weighting: Literal["write_set", "key"] = "write_set"
    beta_max: float = 2.0
    chunk_size: int = 32
    norm_eps: float = 1e-5
    dropout: float = 0.0
    backend: Literal["reference", "triton"] = "reference"

    def __post_init__(self) -> None:
        if self.d_model < 1:
            raise ValueError(f"d_model must be >= 1, got {self.d_model}")
        if self.n_heads < 1 or self.d_model % self.n_heads:
            raise ValueError(
                f"n_heads must be a positive divisor of d_model = {self.d_model}, "
                f"got {self.n_heads}"
            )
        root = math.isqrt(max(self.n_slots, 0))
        if self.n_slots < 1 or root * root != self.n_slots:
            raise ValueError(
                f"n_slots must be a perfect square (product keys address it as "
                f"sqrt(N) x sqrt(N)), got {self.n_slots}"
            )
        if self.initial_memory not in INITIAL_MEMORY:
            raise ValueError(
                f"initial_memory must be one of {INITIAL_MEMORY}, "
                f"got {self.initial_memory!r}"
            )
        for name, count in (("n_writes", self.n_writes), ("n_reads", self.n_reads)):
            if not 1 <= count <= self.n_slots:
                raise ValueError(
                    f"{name} must be in [1, n_slots = {self.n_slots}], got {count}"
                )
        if self.key_norm not in KEY_NORMS:
            raise ValueError(f"key_norm must be one of {KEY_NORMS}, got {self.key_norm!r}")
        if self.decay_weighting not in DECAY_WEIGHTINGS:
            raise ValueError(
                f"decay_weighting must be one of {DECAY_WEIGHTINGS}, "
                f"got {self.decay_weighting!r}"
            )
        if self.decay_weighting == "key" and self.key_norm == "l2":
            raise ValueError(
                'decay_weighting="key" is defined for softmax keys only: an l2 key '
                "has signed weights, and a negative weight would turn the decay "
                "into growth"
            )
        if not 0.0 < self.beta_max <= 2.0:
            raise ValueError(
                f"beta_max must be in (0, 2], got {self.beta_max}; past 2 the "
                f"delta update expands rather than reflects and the UT/WY inverse "
                f"grows geometrically in chunk_size"
            )
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {self.chunk_size}")
        if self.norm_eps <= 0:
            raise ValueError(f"norm_eps must be > 0, got {self.norm_eps}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {self.backend!r}")

    @property
    def d_v(self) -> int:
        """Per-head value width."""
        return self.d_model // self.n_heads

    @property
    def sub_keys(self) -> int:
        """`√N` — the length of each of the two product-key score vectors."""
        return math.isqrt(self.n_slots)


class SparseDeltaMemory(nn.Module):
    """A gated delta rule over a large table, in the house streaming shape.

    Example::

        config = SparseDeltaMemoryConfig(
            d_model=512, n_heads=2, n_slots=128**2, initial_memory="learned"
        )
        mixer = SparseDeltaMemory(config)
        y = mixer(x)                      # (B, T, d_model) -> (B, T, d_model)

    The output side — per-head RMSNorm, then a SiLU gate, then a projection —
    matches Gated DeltaNet's and Undertow's, so a block can hold any of the
    three without knowing which one it has.

    Reuse is by subclassing.  :meth:`_address`, :meth:`_features`, :meth:`_scan`
    and :meth:`_out` are the seams.
    """

    def __init__(self, config: SparseDeltaMemoryConfig) -> None:
        super().__init__()
        if config.backend == "triton" and not triton_kernels.HAS_TRITON:
            raise RuntimeError(
                'backend="triton" was requested but triton did not import. '
                "Install it, or use the reference backend."
            )
        self.config = config

        d_model, n_heads, d_v = config.d_model, config.n_heads, config.d_v
        scores = n_heads * 2 * config.sub_keys

        self.write_proj = nn.Linear(d_model, scores, bias=False)
        self.read_proj = nn.Linear(d_model, scores, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.a_proj = nn.Linear(d_model, n_heads, bias=True)
        self.b_proj = nn.Linear(d_model, n_heads, bias=True)
        self.g_proj = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.o_proj = nn.Linear(n_heads * d_v, d_model, bias=False)

        self.head_norm = nn.Parameter(torch.ones(d_v))
        self.dropout = nn.Dropout(config.dropout)

        # Zero, and created after every draw above: `torch.zeros` consumes no
        # randomness, so a "learned" layer and a "zero" layer built from one seed
        # have identical projections AND an identical initial table -- the same
        # layer, exactly, until training moves this.  Not created at all under
        # "zero", so that layer has exactly the parameters it needs.
        if config.initial_memory == "learned":
            self.initial_memory = nn.Parameter(
                torch.zeros(n_heads, config.n_slots, d_v)
            )
        else:
            self.initial_memory = None

        # Gated DeltaNet's gate, not the paper's `-exp(A_log)·softplus(a + dt)`:
        # one convention for a block holding either mixer.  At -3 the initial
        # `alpha = exp(-softplus(-3)) ≈ 0.953`, which happens to be where the
        # paper's trained gate settles -- and under write-set decay it only ever
        # applies to slots being written.
        nn.init.constant_(self.a_proj.bias, -3.0)
        nn.init.zeros_(self.b_proj.bias)

        #: Rebuild the kernel's pairwise terms in the backward rather than keep
        #: them, while training: a memory dial, not part of the function, and
        #: an attribute for the reason ``Stack.recompute`` is one -- it is not
        #: in the ``state_dict`` and draws nothing.  Those terms are what grows
        #: with ``chunk_size``; with this on, what a training step keeps is the
        #: gathered rows and one pairwise array.  See ``chunk_sparse_delta``.
        self.recompute_pairwise = False

    # ── seams ─────────────────────────────────────────────────────────────

    def _address(
        self, scores: torch.Tensor, n_select: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Product keys: `(B, T, H·2√N)` scores → slots and weights, `(B, H, T, K)`.

        The `N` slot scores are the outer sum of two `√N` score vectors, and
        ``top_K(s¹ ⊕ s²) = top_K(top_K(s¹) ⊕ top_K(s²))``: a pair in the top `K`
        of the sums has each coordinate in the top `K` of its own half, or `K`
        better pairs would exist.  So the exact top `K` of `N` comes from `K²`
        candidates, and the `N` scores are never formed.

        The indices are distinct `(row, column)` pairs **by construction**, and
        :meth:`_scan` relies on that rather than re-checking it: the kernel's
        check is a host sync and a data-dependent branch, and this layer has to
        run under ``torch.func.vmap``.  **An override must keep write indices
        distinct within each position**; the kernels' tests hold what goes
        wrong otherwise.  Selection is not differentiable; the weights are, so
        the projections learn through the scores they select.
        """
        config = self.config
        batch, seq_len, _ = scores.shape
        root = config.sub_keys

        halves = scores.view(batch, seq_len, config.n_heads, 2, root).permute(
            0, 2, 1, 3, 4
        )
        per_half = min(n_select, root)
        top, index = halves.topk(per_half, dim=-1)
        candidates = top[..., 0, :, None] + top[..., 1, None, :]
        best, flat = candidates.flatten(-2).topk(n_select, dim=-1)
        row = index[..., 0, :].gather(-1, flat // per_half)
        column = index[..., 1, :].gather(-1, flat % per_half)

        if config.key_norm == "softmax":
            weight = torch.softmax(best, dim=-1)
        else:
            weight = F.normalize(best, dim=-1)
        return row * root + column, weight

    def _features(
        self, x: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """`(B, T, d)` → the kernel's inputs, in its argument order after `memory`.

        ``write_idx, write_val, log_decay, v, beta, read_idx, read_val``, each
        `(B, H, T, …)`.
        """
        config = self.config
        batch, seq_len, _ = x.shape

        write_idx, write_val = self._address(self.write_proj(x), config.n_writes)
        read_idx, read_val = self._address(self.read_proj(x), config.n_reads)

        # No activation on `v`: the paper's `v = W_v x`, where Gated DeltaNet
        # applies SiLU.  Inputs follow the paper wherever the paper specifies
        # the recurrence; the output side follows the house.
        v = self.v_proj(x).view(batch, seq_len, config.n_heads, config.d_v)
        v = v.transpose(1, 2)
        beta = (config.beta_max * torch.sigmoid(self.b_proj(x))).transpose(1, 2)

        # One decay per head, spread over the write set by `decay_weighting`.
        # The kernel takes a log-decay per write entry and cannot tell the two
        # arrangements apart, so this choice cannot fork it.
        log_alpha = -F.softplus(self.a_proj(x)).transpose(1, 2)
        if config.decay_weighting == "write_set":
            log_decay = log_alpha.unsqueeze(-1).expand_as(write_val)
        else:
            log_decay = log_alpha.unsqueeze(-1) * write_val

        return write_idx, write_val, log_decay, v, beta, read_idx, read_val

    def _scan(
        self, features: tuple[torch.Tensor, ...], memory: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The recurrence.  Override to swap in an accelerated kernel.

        The distinct-writes check is off: :meth:`_address` guarantees it by
        construction, and the check would cost a sync per call and break
        ``torch.func``.  How the table is held is the kernel's default: in
        place, or functionally under a ``torch.func`` transform.  Pass
        ``in_place=False`` from an override to differentiate twice.
        :attr:`recompute_pairwise` is honoured here, so an override should
        pass it on.
        """
        kernel = (
            triton_kernels.chunk_sparse_delta
            if self.config.backend == "triton"
            else chunk_sparse_delta
        )
        return kernel(
            memory,
            *features,
            chunk_size=self.config.chunk_size,
            check_writes=False,
            recompute_pairwise=self.recompute_pairwise and self.training,
        )

    def _out(self, o: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """`(B, H, T, d_v)` → per-head RMSNorm → SiLU gate → projection."""
        config = self.config
        batch, _, seq_len, _ = o.shape

        # Promote to *at least* fp32 rather than `.float()`, which would
        # silently demote an fp64 caller -- Gated DeltaNet's `_out` explains why
        # that matters to a whole-layer fp64 comparison.
        o = o.transpose(1, 2).to(torch.promote_types(o.dtype, torch.float32))
        o = rms_norm(o, self.head_norm, config.norm_eps)
        o = o.reshape(batch, seq_len, config.n_heads * config.d_v).to(x.dtype)
        return self.dropout(self.o_proj(o * F.silu(self.g_proj(x))))

    def residual_out_projections(self) -> tuple[nn.Module, ...]:
        """The projections whose output is added to a residual stream.

        See :meth:`lumen.gdn.GatedDeltaNet.residual_out_projections` for why
        this is asked rather than read off parameter names.  Override alongside
        :meth:`_out`.
        """
        return (self.o_proj,)

    def initial_memory_parameters(self) -> tuple[nn.Parameter, ...]:
        """The learned initial table, or nothing under ``"zero"``.

        A structural fact, exposed for the same reason
        :meth:`residual_out_projections` is: so a caller never has to match
        parameter names to find it.  The caller that exists is an optimiser
        that should not treat this table like a weight matrix -- ordinary
        weight decay shrinks every slot every step, read or not, so pretrained
        content nobody reads decays by optimiser step even while it is frozen
        in context.  What to do instead is the caller's decision, and an open
        one.
        """
        return () if self.initial_memory is None else (self.initial_memory,)

    # ── streaming ─────────────────────────────────────────────────────────

    def _initial_table(
        self, batch: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """`(B, H, N, d_v)`, broadcast from one `(H, N, d_v)` table."""
        config = self.config
        shape = (config.n_heads, config.n_slots, config.d_v)
        if self.initial_memory is not None:
            table = self.initial_memory.to(device=device, dtype=dtype)
        else:
            table = torch.zeros(shape, device=device, dtype=dtype)
        return table.expand(batch, *shape)

    def init_state(
        self,
        batch: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> SparseDeltaMemoryState:
        """The start of a stream: the initial table, empty or learned.

        Device and dtype follow the layer's own parameters unless overridden.
        Under ``"learned"`` the state is a view of the parameter, so a stream
        trained through its state carries a gradient back to the table.
        """
        reference = self.v_proj.weight
        device = reference.device if device is None else device
        dtype = reference.dtype if dtype is None else dtype
        return SparseDeltaMemoryState(memory=self._initial_table(batch, device, dtype))

    def step(
        self, x: torch.Tensor, state: SparseDeltaMemoryState, *, donate: bool = False
    ) -> tuple[torch.Tensor, SparseDeltaMemoryState]:
        """One position — `(B, 1, d_model)` → output and the successor state.

        Args:
            donate: the caller will not read ``state`` again, so the successor
                may be written into its table rather than a copy of it -- at a
                large batch, most of the step.  A permission, not a demand: a
                table that is not the stream's own to write is copied anyway,
                which makes the first donated step from :meth:`init_state` a
                copy and every one after it in place.
        """
        if x.shape[1] != 1:
            raise ValueError(
                f"step() consumes one position at a time, got {x.shape[1]}; "
                f"use forward(x, state=..., return_state=True) for a chunk"
            )
        features = tuple(t[:, :, 0] for t in self._features(x))
        o, memory = recurrent_sparse_delta(
            state.memory,
            *features,
            in_place=donate and _writable_in_place(state.memory),
        )
        return self._out(o.unsqueeze(2), x), SparseDeltaMemoryState(memory=memory)

    def read(self, x: torch.Tensor, state: SparseDeltaMemoryState) -> torch.Tensor:
        """One position, read-only — `(B, 1, d_model)` → output.  No successor.

        A step with the write switched off: the read address is formed exactly
        as :meth:`step` would form it -- through :meth:`_features`, so a
        subclass that changes addressing changes both -- and applied to
        ``state.memory`` as it stands.  Nothing is written and nothing decays.
        """
        if x.shape[1] != 1:
            raise ValueError(f"read() consumes one position at a time, got {x.shape[1]}")
        *_, read_idx, read_val = self._features(x)
        o = read_sparse_delta(state.memory, read_idx[:, :, 0], read_val[:, :, 0])
        return self._out(o.unsqueeze(2), x)

    # ── forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        *,
        state: SparseDeltaMemoryState | None = None,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, SparseDeltaMemoryState]:
        """`(B, T, d_model)` → `(B, T, d_model)`, optionally continuing a stream.

        Args:
            state: table to continue from.  ``None`` starts from the initial
                table -- empty or learned -- which is the beginning of a
                sequence.
            return_state: also return the state after consuming ``x``, so a
                prompt can be prefilled in one parallel pass and generation
                continued with :meth:`step`.
        """
        memory = (
            state.memory
            if state is not None
            else self._initial_table(x.shape[0], x.device, x.dtype)
        )
        o, memory = self._scan(self._features(x), memory)
        y = self._out(o, x)
        if return_state:
            # The reference kernel's table is a view of its working buffer,
            # which carries `B·H·C·W` scratch rows past the real ones -- several
            # tables' worth when `N` is small -- so a state kept from it would
            # keep the scratch too.  Copied here rather than in the kernel: a
            # training forward discards the table, and should not pay a second
            # one in peak memory for a state nobody asked for.
            return y, SparseDeltaMemoryState(memory=memory.clone())
        return y

    def extra_repr(self) -> str:
        config = self.config
        return (
            f"d_model={config.d_model}, n_heads={config.n_heads}, "
            f"n_slots={config.n_slots}, d_v={config.d_v}, "
            f"initial_memory={config.initial_memory}, "
            f"n_writes={config.n_writes}, n_reads={config.n_reads}, "
            f"key_norm={config.key_norm}, decay_weighting={config.decay_weighting}, "
            f"beta_max={config.beta_max}, backend={config.backend}"
        )
