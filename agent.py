"""
Player agent for poker game.
Override the get_action method to implement custom strategies.
"""

from __future__ import annotations
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from env.cards import Card
    from env.board import Board

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
        hand_history: List[List[Tuple[str, str, int]]],
    ) -> Action:
        """
        Human input-based action selection.

        Args:
            history: Betting history for current round [(player, action, amount), ...]
            hand_history: History of previous hands, each containing its betting history
                         [[(player, action, amount), ...], ...]
        """
        print(f"\n{self.name}'s turn:")
        print(f"Hand: {[str(c) for c in self.hole]}")
        print(f"Board: {board.get_board()}")
        print(f"Stack: {self.stack} | Pot: {pot}")
        print(f"To call: {to_call} | Min raise: {min_raise}")

        while True:
            action = input("Choose action (f=fold, c=call, r=raise): ").lower().strip()

            if action == 'f':
                return ("fold", 0)
            elif action == 'c':
                return ("call", 0)
            elif action == 'r':
                while True:
                    try:
                        raise_amount = int(input(f"Raise amount (min {min_raise}): "))
                        if raise_amount >= min_raise:
                            return ("raise", raise_amount)
                        print(f"Raise must be at least {min_raise}")
                    except ValueError:
                        print("Please enter a valid number")
            else:
                print("Invalid action. Use f, c, or r")
                

    def __str__(self) -> str:
        return f"{self.name}(stack={self.stack}, hole={[str(c) for c in self.hole]})"

