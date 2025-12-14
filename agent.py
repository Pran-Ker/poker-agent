"""
Player agent for poker game.
Override the get_action method to implement custom strategies.
"""

from __future__ import annotations
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from poker import Card, Board

Action = Tuple[str, int]  # ("call"/"raise"/"fold", amount)


class Player:
    def __init__(self, name: str, starting_cash: int):
        self.name = name
        self.stack = starting_cash
        self.hole: List[Card] = []
        self.folded = False
        self.all_in = False

    def init_hand(self, c1: Card, c2: Card) -> None:
        self.hole = [c1, c2]
        self.folded = False
        self.all_in = False

    def get_action(
        self,
        *,
        to_call: int,
        min_raise: int,
        pot: int,
        board: Board,
        round_idx: int,
        history: List[Tuple[str, str, int]],
    ) -> Action:
        """
        Default policy: always call (or check if to_call == 0).
        Override this in a subclass to implement strategies.
        """
        if to_call > 0:
            return ("call", 0)
        return ("call", 0)  # check

    def __str__(self) -> str:
        return f"{self.name}(stack={self.stack}, hole={[str(c) for c in self.hole]})"
