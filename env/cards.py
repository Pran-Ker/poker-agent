"""
Card and Deck classes for poker game.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
import random


# ----------------------------
# Cards / Deck
# ----------------------------

RANKS = "23456789TJQKA"
SUITS = "SHDC"  # Spades, Hearts, Diamonds, Clubs


@dataclass(frozen=True)
class Card:
    rank: str  # '2'..'A'
    suit: str  # 'S','H','D','C'

    def __str__(self) -> str:
        return f"{self.rank}{self.suit}"

    @property
    def rank_value(self) -> int:
        return RANKS.index(self.rank) + 2  # 2..14


class Deck:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.cards = [Card(r, s) for r in RANKS for s in SUITS]
        self.rng.shuffle(self.cards)

    def draw(self) -> Card:
        if not self.cards:
            raise RuntimeError("Deck is empty")
        return self.cards.pop()
