"""
Minimal heads-up Texas Hold'em simulator (2 players) based on your pseudocode.

Key simplifications (easy to extend later):
- No blinds/antes (Player1 is forced to open with a raise of 1 each betting round if possible)
- Default player policy: always "call" (i.e., check if nothing to call)
- Betting is a simple heads-up loop: raise -> response -> end (supports folds and more raises if you later add them)
- Showdown evaluates best 5-card hand out of 7 (2 hole + 5 board)
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import List, Tuple, Optional, Dict
import random

from agent import Player, Action


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


# ----------------------------
# Hand evaluation (showdown)
# ----------------------------

def _is_straight(values_desc: List[int]) -> Optional[int]:
    """
    values_desc: sorted unique ranks, descending (e.g. [14,13,12,11,10,...])
    Returns the high card of the straight if present, else None.
    Handles wheel A-2-3-4-5 as high=5.
    """
    vals = sorted(set(values_desc), reverse=True)
    if 14 in vals:  # Ace can be low
        vals.append(1)

    run = 1
    for i in range(len(vals) - 1):
        if vals[i] - 1 == vals[i + 1]:
            run += 1
            if run >= 5:
                # high card is the start of this 5-run
                high = vals[i - (run - 2)]
                return 5 if high == 1 else high
        else:
            run = 1
    return None


def rank_5card_hand(cards5: List[Card]) -> Tuple[int, List[int]]:
    """
    Returns a comparable rank:
      (category, kickers...)
    category: 8 straight flush, 7 quads, 6 full house, 5 flush, 4 straight,
              3 trips, 2 two pair, 1 pair, 0 high card
    Kickers list breaks ties.
    """
    vals = sorted([c.rank_value for c in cards5], reverse=True)
    suits = [c.suit for c in cards5]

    # counts
    counts: Dict[int, int] = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    # sort by (count desc, value desc)
    groups = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

    is_flush = len(set(suits)) == 1
    straight_high = _is_straight(vals)

    if is_flush and straight_high is not None:
        return (8, [straight_high])

    if groups[0][1] == 4:
        quad = groups[0][0]
        kicker = max(v for v in vals if v != quad)
        return (7, [quad, kicker])

    if groups[0][1] == 3 and groups[1][1] == 2:
        trips = groups[0][0]
        pair = groups[1][0]
        return (6, [trips, pair])

    if is_flush:
        return (5, vals)

    if straight_high is not None:
        return (4, [straight_high])

    if groups[0][1] == 3:
        trips = groups[0][0]
        kickers = [v for v in vals if v != trips]
        return (3, [trips] + kickers)

    if groups[0][1] == 2 and groups[1][1] == 2:
        high_pair = max(groups[0][0], groups[1][0])
        low_pair = min(groups[0][0], groups[1][0])
        kicker = max(v for v in vals if v != high_pair and v != low_pair)
        return (2, [high_pair, low_pair, kicker])

    if groups[0][1] == 2:
        pair = groups[0][0]
        kickers = [v for v in vals if v != pair]
        return (1, [pair] + kickers)

    return (0, vals)


def best_hand_rank(seven_cards: List[Card]) -> Tuple[int, List[int]]:
    best = (-1, [])
    for combo in combinations(seven_cards, 5):
        r = rank_5card_hand(list(combo))
        if r > best:
            best = r
    return best


# ----------------------------
# Game + betting + play loop
# ----------------------------

class Game:
    def __init__(self, starting_cash: int = 10, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.player1 = Player("Player1", starting_cash)
        self.player2 = Player("Player2", starting_cash)
        self.board: Optional[Board] = None

    def next_hand(self) -> bool:
        """
        Deal hole cards + 5 board cards (no repeats by construction).
        Returns False if a player is broke and game can't continue.
        """
        if self.player1.stack <= 0 or self.player2.stack <= 0:
            return False

        deck = Deck(self.rng)

        p1c1, p1c2 = deck.draw(), deck.draw()
        p2c1, p2c2 = deck.draw(), deck.draw()
        board_cards = [deck.draw() for _ in range(5)]

        self.player1.init_hand(p1c1, p1c2)
        self.player2.init_hand(p2c1, p2c2)
        self.board = Board(board_cards)
        return True

    def _pay_to_pot(self, player: Player, amount: int) -> int:
        """
        Deduct up to 'amount' from player's stack into pot.
        Returns actual paid (can be less if all-in).
        """
        if amount <= 0:
            return 0
        paid = min(player.stack, amount)
        player.stack -= paid
        if player.stack == 0:
            player.all_in = True
        return paid

    def bet_round(self, round_idx: int, pot: int) -> Tuple[List[Tuple[str, str, int]], int, Optional[Player]]:
        """
        Runs one betting round. Returns:
          - betstate list: [(player_name, action, amount_put_in_now), ...]
          - new pot total
          - winner if someone folded else None
        """
        assert self.board is not None

        betstate: List[Tuple[str, str, int]] = []
        contrib = {self.player1.name: 0, self.player2.name: 0}  # this round only
        current_to_call = 0
        min_raise = 1

        p1, p2 = self.player1, self.player2

        # Force first action: Player1 must "raise(1)" if possible, else just "call/check".
        if not p1.folded and not p1.all_in and p1.stack > 0:
            raise_amt = min_raise
            paid = self._pay_to_pot(p1, raise_amt)
            contrib[p1.name] += paid
            pot += paid
            current_to_call = contrib[p1.name] - contrib[p2.name]
            betstate.append((p1.name, "raise", paid))
        else:
            betstate.append((p1.name, "call", 0))

        # Now Player2 responds, then (optionally) P1 responds to further raises, etc.
        actor, other = p2, p1
        pending_response = True  # there is something to respond to if to_call > 0

        while True:
            if actor.folded or actor.all_in:
                # if actor can't act, treat as call/check when possible
                pass

            to_call = max(0, contrib[other.name] - contrib[actor.name])

            if actor.folded:
                return betstate, pot, other

            if actor.all_in:
                # can't add more; if still behind, it's effectively an all-in call of zero additional
                betstate.append((actor.name, "call", 0))
                if pending_response and to_call == 0:
                    pending_response = False
                if not pending_response:
                    return betstate, pot, None
                actor, other = other, actor
                continue

            # choose action
            action, amt = actor.get_action(
                to_call=to_call,
                min_raise=min_raise,
                pot=pot,
                board=self.board,
                round_idx=round_idx,
                history=betstate,
            )

            if action == "fold":
                actor.folded = True
                betstate.append((actor.name, "fold", 0))
                return betstate, pot, other

            if action == "call":
                paid = self._pay_to_pot(actor, to_call)
                contrib[actor.name] += paid
                pot += paid
                betstate.append((actor.name, "call", paid))

                # If they fully matched, response is complete
                if contrib[actor.name] >= contrib[other.name]:
                    pending_response = False

                if not pending_response:
                    return betstate, pot, None

            elif action == "raise":
                # amt is the "raise size" beyond calling, but we clamp to at least min_raise
                raise_size = max(min_raise, amt)
                total_put_in = to_call + raise_size
                paid = self._pay_to_pot(actor, total_put_in)
                contrib[actor.name] += paid
                pot += paid
                betstate.append((actor.name, "raise", paid))
                pending_response = True  # other must respond

            else:
                raise ValueError(f"Unknown action: {action}")

            actor, other = other, actor  # switch turns

    def showdown(self, pot: int) -> Player:
        """
        Determine winner at showdown (both not folded).
        """
        assert self.board is not None
        p1, p2 = self.player1, self.player2

        seven1 = p1.hole + self.board.cards
        seven2 = p2.hole + self.board.cards

        r1 = best_hand_rank(seven1)
        r2 = best_hand_rank(seven2)

        if r1 > r2:
            return p1
        if r2 > r1:
            return p2

        # Tie-break: split pot. For simplicity return Player1 but split outside.
        return p1

    def play(self, max_hands: int = 1000, verbose: bool = False) -> None:
        hand_num = 0
        while hand_num < max_hands and self.next_hand():
            hand_num += 1
            assert self.board is not None

            pot = 0
            hand_winner: Optional[Player] = None
            tie = False

            if verbose:
                print(f"\n=== Hand {hand_num} ===")
                print(f"P1 hole: {[str(c) for c in self.player1.hole]} | stack={self.player1.stack}")
                print(f"P2 hole: {[str(c) for c in self.player2.hole]} | stack={self.player2.stack}")

            for round_idx in range(4):
                self.board.set_round(round_idx)
                if verbose:
                    round_names = ["Preflop", "Flop", "Turn", "River"]
                    print(f"\n--- {round_names[round_idx]} ---")
                    print(f"Board: {' '.join(self.board.get_board())}")
                    print(f"Pot: {pot}")

                betstate, pot, fold_winner = self.bet_round(round_idx, pot)

                if verbose:
                    print("Actions:")
                    for player_name, action, amount in betstate:
                        if action == "fold":
                            print(f"  {player_name}: {action}")
                        elif amount > 0:
                            print(f"  {player_name}: {action} {amount}")
                        else:
                            print(f"  {player_name}: {action}")
                    print(f"Pot after betting: {pot}")

                if fold_winner is not None:
                    hand_winner = fold_winner
                    break

                # if someone is all-in, you could skip remaining betting rounds
                if self.player1.all_in or self.player2.all_in:
                    # run out board to river (already dealt; just reveal)
                    self.board.set_round(3)
                    break

            if hand_winner is None:
                # showdown (or tie)
                winner = self.showdown(pot)
                # check if true tie
                r1 = best_hand_rank(self.player1.hole + self.board.cards)
                r2 = best_hand_rank(self.player2.hole + self.board.cards)

                if verbose:
                    hand_names = ["High Card", "Pair", "Two Pair", "Three of a Kind",
                                  "Straight", "Flush", "Full House", "Four of a Kind", "Straight Flush"]
                    print(f"\n--- Showdown ---")
                    print(f"Board: {' '.join([str(c) for c in self.board.cards])}")
                    print(f"Player1: {hand_names[r1[0]]}")
                    print(f"Player2: {hand_names[r2[0]]}")

                if r1 == r2:
                    tie = True
                hand_winner = winner

            # award pot
            if tie:
                half = pot // 2
                self.player1.stack += half
                self.player2.stack += pot - half
                if verbose:
                    print(f"TIE -> split pot {pot}: P1+={half}, P2+={pot-half}")
            else:
                hand_winner.stack += pot
                if verbose:
                    print(f"Winner: {hand_winner.name} wins pot={pot}")

            if verbose:
                print(f"Stacks: P1={self.player1.stack} P2={self.player2.stack}")

        if verbose:
            print("\n=== Game Over ===")
        print(f"Final stacks after {hand_num} hands: P1={self.player1.stack}, P2={self.player2.stack}")


# ----------------------------
# Example usage
# ----------------------------

if __name__ == "__main__":
    game = Game(starting_cash=10)
    game.play(max_hands=3, verbose=True)  # set verbose=True to see each hand/round
    