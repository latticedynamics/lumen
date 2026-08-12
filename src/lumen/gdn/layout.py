"""Head layout — which key and which value each state of the memory uses.

The usual arrangement gives every head its own key *and* its own value: `H`
states out of `2H` projections.  One of Lumen's contributing lineages instead
defined `G_k` key groups and `G_v` value groups and built a state for **every
pair**, so `H = G_k · G_v` distinct memories come out of `G_k + G_v`
projections.

Those two are not nested, and the direction of the failure is the one that
matters: a Cartesian product equals its own diagonal only when both factors are
singletons, so a layer hard-wired to instantiate *all* pairs cannot express the
ordinary one-key-one-value arrangement at all.  The two families meet at
``H = 1`` and nowhere else.  A layer that cannot be configured into the standard
arrangement is a layer that can never be checked against one.

So this module parameterises the **assignment** rather than the grid.  A head is
a ``(key group, value group)`` pair and the layout is the list of pairs::

    HeadLayout.crossed(2, 4)      # all pairs               — 8 states
    HeadLayout.diagonal(8)        # the diagonal            — the ordinary layer
    HeadLayout.shared_key(8)      # one address, 8 payloads
    HeadLayout.shared_value(8)    # 8 addresses, one payload

This is pure logic — tuples and integers, no tensors and no device — so it
stays testable anywhere.  The tensor side lives in :mod:`lumen.gdn.reference`.

The rectangle rule
------------------
Sorted by key group, the heads must form a ``G_k × m`` rectangle: every key
group carries the same number of states.  All four constructors above satisfy
it, and it is what keeps the kernel batched — the UT/WY inverse is built **once
per key group** and reused by every state sharing that key, so the
``O(C²d_k + C³)`` term is paid `G_k` times rather than `H` times.

The rule also makes every reindexing a view for the layouts anyone runs: a
uniform assignment is a stride-0 broadcast over the key axis, and the diagonal
is a reshape.  Only a genuinely ragged assignment costs a copy, and this module
will tell you which one you have (:attr:`HeadLayout.reindex_is_free`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class HeadLayout:
    """An assignment of states to `(key group, value group)` pairs.

    Both tuples have length `H`, the number of states.  Head ``h`` is addressed
    by key group ``key_group[h]`` and written with value group
    ``value_group[h]``.

    Use the constructors — :meth:`crossed`, :meth:`diagonal`,
    :meth:`shared_key`, :meth:`shared_value` — unless you specifically want an
    assignment none of them expresses.

    Validated on construction:

    * both group labellings are dense from zero, so ``G_k`` and ``G_v`` are
      unambiguous and no projection is computed and then never used
    * ``key_group`` is non-decreasing, which fixes the head ordering as
      key-group-major and lets ``q`` be reshaped rather than gathered
    * every key group carries the same number of states — the rectangle rule
    * no two heads share *both* groups.  Such heads are the same memory
      computed twice: identical key, identical value, identical decay, so
      identical state.  Two readers of one memory is a coherent thing to want
      and is not this; it would be one state with two queries.
    """

    key_group: tuple[int, ...]
    value_group: tuple[int, ...]

    def __post_init__(self) -> None:
        # Coerce first: a caller passing lists should get a usable frozen
        # object, not an equality surprise three hours later.
        object.__setattr__(self, "key_group", tuple(int(g) for g in self.key_group))
        object.__setattr__(self, "value_group", tuple(int(g) for g in self.value_group))

        n_heads = len(self.key_group)
        if n_heads < 1:
            raise ValueError("a layout needs at least one head")
        if len(self.value_group) != n_heads:
            raise ValueError(
                f"key_group and value_group must be the same length, got "
                f"{n_heads} and {len(self.value_group)}"
            )

        for name, groups in (
            ("key_group", self.key_group),
            ("value_group", self.value_group),
        ):
            labels = sorted(set(groups))
            if labels != list(range(len(labels))):
                raise ValueError(
                    f"{name} must be labelled densely from 0, got {sorted(set(groups))}"
                )

        if list(self.key_group) != sorted(self.key_group):
            raise ValueError(
                f"key_group must be non-decreasing so heads are key-group-major, "
                f"got {self.key_group}"
            )

        counts = [self.key_group.count(i) for i in range(self.n_key_groups)]
        if len(set(counts)) != 1:
            raise ValueError(
                f"every key group must carry the same number of states (the "
                f"rectangle rule), got counts {counts}"
            )

        pairs = list(zip(self.key_group, self.value_group))
        if len(set(pairs)) != len(pairs):
            duplicate = next(p for p in pairs if pairs.count(p) > 1)
            raise ValueError(
                f"two heads share the pair {duplicate}; they would be the same "
                f"memory computed twice. For two readers of one memory, use one "
                f"state and two queries."
            )

    # ── constructors ──────────────────────────────────────────────────────

    @classmethod
    def crossed(cls, n_key_groups: int, n_value_groups: int) -> HeadLayout:
        """Every pair — ``G_k · G_v`` states from ``G_k + G_v`` projections.

        The generalisation one lineage was built around.  Retained because it
        is the correct generalisation; **not** the default, because the only
        controlled comparison of it went against it (see the design record).
        """
        if n_key_groups < 1 or n_value_groups < 1:
            raise ValueError("group counts must be >= 1")
        return cls(
            key_group=tuple(i for i in range(n_key_groups) for _ in range(n_value_groups)),
            value_group=tuple(j for _ in range(n_key_groups) for j in range(n_value_groups)),
        )

    @classmethod
    def diagonal(cls, n_heads: int) -> HeadLayout:
        """One key and one value per head — the ordinary arrangement.

        This is the row that makes the component checkable against published
        work.  Note that it is *not* ``crossed(H, H)``: that would be `H²`
        states.  It is the diagonal of that grid, which no ``(G_k, G_v)`` pair
        can name.
        """
        return cls(key_group=tuple(range(n_heads)), value_group=tuple(range(n_heads)))

    @classmethod
    def shared_key(cls, n_heads: int) -> HeadLayout:
        """One address space, ``H`` payloads — ``crossed(1, H)``.

        Every state answers to the same key and differs only in what it stores,
        so the layer has one forgetting timescale (`α` and `β` live on the key
        group).  That coupling is a choice rather than a necessity; see the
        design record on which of the two is forced.
        """
        return cls(key_group=(0,) * n_heads, value_group=tuple(range(n_heads)))

    @classmethod
    def shared_value(cls, n_heads: int) -> HeadLayout:
        """``H`` address spaces, one payload — ``crossed(H, 1)``.

        The grouped-query shape: many keys reading into a shared value stream.
        """
        return cls(key_group=tuple(range(n_heads)), value_group=(0,) * n_heads)

    # ── derived ───────────────────────────────────────────────────────────

    @property
    def n_heads(self) -> int:
        """Number of states.  The layout is the source of truth for this."""
        return len(self.key_group)

    @property
    def n_key_groups(self) -> int:
        return len(set(self.key_group))

    @property
    def n_value_groups(self) -> int:
        return len(set(self.value_group))

    @property
    def states_per_key_group(self) -> int:
        """``m`` — the rectangle's width, guaranteed equal across key groups."""
        return self.n_heads // self.n_key_groups

    @property
    def rows(self) -> tuple[tuple[int, ...], ...]:
        """`(G_k, m)` — the value groups each key group writes, in head order."""
        m = self.states_per_key_group
        return tuple(
            tuple(self.value_group[i * m:(i + 1) * m]) for i in range(self.n_key_groups)
        )

    @property
    def rows_are_uniform(self) -> bool:
        """Do all key groups write the same value groups, in the same order?

        True for every crossed layout and for both shared-* layouts, and it is
        the condition under which the value tensor is a stride-0 broadcast over
        the key axis rather than a gather.
        """
        rows = self.rows
        return all(row == rows[0] for row in rows)

    @property
    def reindex_is_free(self) -> bool:
        """Can the value tensor be assembled without copying?

        True for uniform layouts (broadcast) and for the diagonal (reshape).
        A ragged assignment is supported and copies; this is how a caller finds
        out which one it has built, rather than discovering it in a profile.
        """
        if self.rows_are_uniform:
            return True
        flat = tuple(g for row in self.rows for g in row)
        return flat == tuple(range(len(flat)))

    def describe(self) -> str:
        """A short name for the layout, for reprs and benchmark tables."""
        g_k, g_v, m = self.n_key_groups, self.n_value_groups, self.states_per_key_group
        if self == HeadLayout.diagonal(self.n_heads):
            return f"diagonal(H={self.n_heads})"
        if self == HeadLayout.shared_key(self.n_heads):
            return f"shared_key(H={self.n_heads})"
        if self == HeadLayout.shared_value(self.n_heads):
            return f"shared_value(H={self.n_heads})"
        if m == g_v and self == HeadLayout.crossed(g_k, g_v):
            return f"crossed(G_k={g_k}, G_v={g_v})"
        return f"custom(H={self.n_heads}, G_k={g_k}, G_v={g_v})"
