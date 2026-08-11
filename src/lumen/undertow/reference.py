"""Undertow reference kernels — fp32, torch-only, correct before fast.

Two paths live here and they compute the same thing:

* the **windowed** path, ``O(T·W)``, which packs each query's `W` reachable
  keys into a `(B, H, T, W)` score tensor.  This is what the layer runs.
* the **dense** path, ``O(T²)``, which builds the full score matrix and adds a
  `(T, T)` log-space bias.  This is the *oracle* — it is the definition of the
  layer written in the most obvious possible way, and the windowed path is
  required to reproduce it to 1e-6 at every position (see
  ``notes/drafts/UNDERTOW.md`` §6).

Keeping the obvious implementation around, permanently, is the cheapest defence
against a clever one drifting.  It is not dead code; it is the specification.

Window indexing convention, used everywhere in this package
-----------------------------------------------------------
Slot ``w`` of query ``t`` holds key position ``t - (W-1) + w``.  So ``w = W-1``
is the query's own position and ``w = 0`` is the farthest reachable key::

    distance  δ(w) = (W-1) - w        δ(W-1) = 0 (self),  δ(0) = W-1 (edge)

A slot is *valid* only when the position it names is ``>= 0``.  Near the start
of a sequence some slots name positions that do not exist; they are masked to
``-inf`` rather than zero-padded, because a zeroed key still scores
``⟨q, 0⟩ = 0`` and ``exp(0) = 1`` — padding would draw real attention weight.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def decay_profile(
    window: int,
    plateau: int | None = None,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """The `(W,)` attenuation profile, indexed by window slot.

    Full strength out to ``plateau``, then a cosine ramp toward the window edge:

        p(δ) = 1                                  if δ < plateau
             = ½·(1 + cos(π·(δ-plateau)/(W-plateau)))   otherwise

    ``plateau=None`` gives a hard window — the profile is identically 1.

    The profile is **strictly positive**: the ramp would reach zero at δ = W,
    but the farthest slot sits at δ = W-1, so it never gets there.  That is
    load-bearing rather than incidental — ``log p`` has to be finite, and the
    placement-equivalence argument in the design record depends on it.

    Returns:
        `(window,)` tensor in (0, 1], where index ``w`` carries distance
        ``(window-1) - w``.
    """
    distance = torch.arange(window - 1, -1, -1, device=device, dtype=dtype)

    if plateau is None or plateau >= window:
        return torch.ones_like(distance)

    # clamp(min=0) makes the plateau region fall out of the same expression:
    # progress 0 -> cos(0) = 1 -> p = 1.  No branch needed, and no chance of a
    # negative progress quietly producing p < 1 inside the plateau.
    progress = (distance - plateau).clamp(min=0.0) / (window - plateau)
    return 0.5 * (1.0 + torch.cos(math.pi * progress))


def log_decay_profile(
    window: int,
    plateau: int | None = None,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """``log`` of :func:`decay_profile` — the form the layer actually adds.

    Finite everywhere, because the profile is strictly positive.
    """
    return decay_profile(window, plateau, device=device, dtype=dtype).log()


def window_validity(
    seq_len: int,
    window: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """`(T, W)` bool mask — does slot ``w`` of query ``t`` name a real position?

    False only in the first ``W-1`` rows, where part of the window lies before
    the start of the sequence.
    """
    t = torch.arange(seq_len, device=device)
    w = torch.arange(window, device=device)
    return (t[:, None] - (window - 1) + w[None, :]) >= 0


def windowed_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    window: int,
    scale: float,
) -> torch.Tensor:
    """Windowed QKᵀ — `(B, H, T, D)` × 2 → `(B, H, T, W)`.

    Computed one window offset at a time.  Each iteration touches only a
    `(B, H, T)` temporary and slices ``k`` as a view, so no `(B, H, T, W, D)`
    intermediate is ever materialised — that tensor is what makes the naive
    unfold formulation unusable at long sequence length.
    """
    seq_len = q.shape[2]
    k_padded = F.pad(k, (0, 0, window - 1, 0))
    return torch.stack(
        [(q * k_padded[:, :, w:w + seq_len, :]).sum(-1) for w in range(window)],
        dim=-1,
    ) / scale


def windowed_aggregate(
    weights: torch.Tensor,
    v: torch.Tensor,
    window: int,
) -> torch.Tensor:
    """Windowed weighted sum — `(B, H, T, W)` × `(B, H, T, D)` → `(B, H, T, D)`.

    The mirror of :func:`windowed_scores`, and it must use the same padding, or
    slot ``w`` means two different keys on the two sides of the softmax.
    """
    seq_len = v.shape[2]
    v_padded = F.pad(v, (0, 0, window - 1, 0))
    out = v.new_zeros(v.shape)
    for w in range(window):
        out = out + weights[..., w].unsqueeze(-1) * v_padded[:, :, w:w + seq_len, :]
    return out


def windowed_weights(
    q: torch.Tensor,
    k: torch.Tensor,
    window: int,
    plateau: int | None,
    *,
    log_profile: torch.Tensor | None = None,
) -> torch.Tensor:
    """`(B, H, T, W)` attention weights — the whole pre-aggregation path.

    ``scores + log p`` and the structural ``-inf`` are both pre-softmax additive
    terms, so the layer has exactly one masking mechanism.  That is the reason
    the log-space form was chosen over multiplying after the softmax; see the
    design record §3.1.

    Args:
        log_profile: optional precomputed `(W,)` log profile, so a caller that
            runs this repeatedly does not rebuild it.  Must be the profile for
            *this* ``window``/``plateau``.
    """
    seq_len = q.shape[2]
    scale = math.sqrt(q.shape[-1])

    scores = windowed_scores(q, k, window, scale)

    if log_profile is None:
        log_profile = log_decay_profile(
            window, plateau, device=q.device, dtype=scores.dtype
        )
    scores = scores + log_profile.to(scores.dtype)

    valid = window_validity(seq_len, window, device=q.device)
    scores = scores.masked_fill(~valid, float("-inf"))

    return torch.softmax(scores, dim=-1)


def windowed_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window: int,
    plateau: int | None = None,
    *,
    log_profile: torch.Tensor | None = None,
) -> torch.Tensor:
    """The `O(T·W)` operator — `(B, H, T, D)` in, `(B, H, T, D)` out."""
    weights = windowed_weights(q, k, window, plateau, log_profile=log_profile)
    return windowed_aggregate(weights, v, window)


def dense_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window: int,
    plateau: int | None = None,
) -> torch.Tensor:
    """The `O(T²)` oracle — same layer, written the obvious way.

    Builds the full `(T, T)` score matrix and adds a log-space bias that is
    ``log p(i-j)`` inside the window and ``-inf`` outside it.  Every row has at
    least the diagonal valid (δ = 0 is always in window), so no row is entirely
    ``-inf`` and the softmax is always well defined.

    Quadratic in sequence length and not meant for training.  It exists so the
    windowed path has something to be checked against.
    """
    seq_len = q.shape[2]
    scale = math.sqrt(q.shape[-1])

    scores = torch.matmul(q, k.transpose(-2, -1)) / scale

    # Reindex the profile by distance rather than by slot: flipping sends slot
    # w -> distance (W-1)-w, so element d of the result is p(d).
    log_p_by_distance = log_decay_profile(
        window, plateau, device=q.device, dtype=scores.dtype
    ).flip(0)

    positions = torch.arange(seq_len, device=q.device)
    distance = positions[:, None] - positions[None, :]
    in_window = (distance >= 0) & (distance < window)

    log_bias = torch.full(
        (seq_len, seq_len), float("-inf"), device=q.device, dtype=scores.dtype
    )
    log_bias[in_window] = log_p_by_distance[distance[in_window]]

    weights = torch.softmax(scores + log_bias, dim=-1)
    return torch.matmul(weights, v)
