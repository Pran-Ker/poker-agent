"""
Board class for managing community cards.
"""

from typing import List

from env.cards import Card


# ----------------------------
# Board
# ----------------------------

class Board:
    """
    Stores 5 community cards and which are "open" (visible).
    open_count is used instead of indexes for simplicity:
      0 => preflop (no community shown)
      3 => flop
      4 => turn
      5 => river
    """
    def __init__(self, cards: List[Card]):
        if len(cards) != 5:
            raise ValueError("Board must have exactly 5 cards")
        self.cards = cards
        self.open_count = 0

    def set_round(self, round_idx: int) -> None:
        # round 0: 0 open, round 1: 3 open, round 2: 4 open, round 3: 5 open
        if round_idx == 0:
            self.open_count = 0
        elif round_idx == 1:
            self.open_count = 3
        elif round_idx == 2:
            self.open_count = 4
        elif round_idx == 3:
            self.open_count = 5
        else:
            raise ValueError("round_idx must be 0..3")

    def get_board(self) -> List[str]:
        visible = []
        for i, c in enumerate(self.cards):
            visible.append(str(c) if i < self.open_count else "X")
        return visible
