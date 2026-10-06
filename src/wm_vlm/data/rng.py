"""Tiny deterministic PRNG that is identical on every platform and Python version.

We avoid `random` and `numpy.random` for dataset construction because their
sampling helpers are not guaranteed to be stream-stable across releases.
"""

from __future__ import annotations

import hashlib
from collections.abc import MutableSequence, Sequence
from typing import TypeVar

T = TypeVar("T")
_MASK = (1 << 64) - 1


class DetRng:
    """SplitMix64 seeded from a SHA-256 digest of arbitrary labelled parts."""

    def __init__(self, *parts: object) -> None:
        digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
        self._state = int.from_bytes(digest[:8], "big")

    def next_u64(self) -> int:
        self._state = (self._state + 0x9E3779B97F4A7C15) & _MASK
        z = self._state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
        return z ^ (z >> 31)

    def randbelow(self, n: int) -> int:
        if type(n) is not int or not 0 < n <= 1 << 64:
            raise ValueError("n must be an integer in [1, 2**64]")
        limit = (1 << 64) - ((1 << 64) % n)
        while True:
            x = self.next_u64()
            if x < limit:
                return x % n

    def randint(self, lo: int, hi: int) -> int:
        """Inclusive on both ends."""
        return lo + self.randbelow(hi - lo + 1)

    def choice(self, seq: Sequence[T]) -> T:
        return seq[self.randbelow(len(seq))]

    def shuffle(self, items: MutableSequence[T]) -> None:
        for i in range(len(items) - 1, 0, -1):
            j = self.randbelow(i + 1)
            items[i], items[j] = items[j], items[i]
