"""
Minimal heads-up Texas Hold'em simulator (2 players) - Entry point.

Key simplifications (easy to extend later):
- No blinds/antes (both players start with equal stacks)
- Both players use LLM to make decisions (Player1 acts first each round)
- Betting is a simple heads-up loop with full raise/call/fold options
- Showdown evaluates best 5-card hand out of 7 (2 hole + 5 board)
"""

from typing import List, Tuple, Optional, Dict, Any
import random

from env.cards import Card, Deck
from env.board import Board
from env.hand_evaluator import best_hand_rank
from agent import Player


# ----------------------------
# Game + betting + play loop
# ----------------------------

class Game:
    def __init__(self, starting_cash: int = 10, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.player1 = Player("Player1", starting_cash)
        self.player2 = Player("Player2", starting_cash)
        self.board: Optional[Board] = None
        self.hand_history: List[Dict[str, Any]] = []
        self.hand_counter = 0

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

    def bet_round(self, round_idx: int, pot: int) -> Tuple[List[Dict[str, Any]], int, Optional[Player]]:
        """
        Runs one betting round. Returns:
          - betstate list: [{"player": name, "action": action, "amount": chips_put_in, "street": street}, ...]
          - new pot total
          - winner if someone folded else None
        """
        assert self.board is not None

        street_names = ["preflop", "flop", "turn", "river"]
        street = street_names[round_idx] if round_idx < len(street_names) else f"round_{round_idx}"

        betstate: List[Dict[str, Any]] = []
        contrib = {self.player1.name: 0, self.player2.name: 0}  # this round only
        current_to_call = 0
        min_raise = 100

        p1, p2 = self.player1, self.player2

        # Player1 acts first (LLM decision, not forced)
        actor, other = p1, p2
        pending_response = False  # No action to respond to yet
        actions_count = 0  # Track number of actions to ensure both players act

        while True:
            actions_count += 1
            if actor.folded or actor.all_in:
                # if actor can't act, treat as call/check when possible
                pass

            to_call = max(0, contrib[other.name] - contrib[actor.name])

            if actor.folded:
                return betstate, pot, other

            if actor.all_in:
                # can't add more; if still behind, it's effectively an all-in call of zero additional
                betstate.append({"player": actor.name, "action": "call", "amount": 0, "street": street})
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
                hand_history=self.hand_history,
            )

            if action == "fold":
                actor.folded = True
                betstate.append({"player": actor.name, "action": "fold", "amount": 0, "street": street})
                return betstate, pot, other

            if action == "call":
                paid = self._pay_to_pot(actor, to_call)
                contrib[actor.name] += paid
                pot += paid
                betstate.append({"player": actor.name, "action": "call", "amount": paid, "street": street})

                # If they fully matched, response is complete
                if contrib[actor.name] >= contrib[other.name]:
                    pending_response = False

                # End round only if no pending response AND both players have acted (at least 2 actions)
                if not pending_response and actions_count >= 2:
                    return betstate, pot, None

            elif action == "raise":
                # amt is the "raise size" beyond calling, but we clamp to at least min_raise
                raise_size = max(min_raise, amt)
                total_put_in = to_call + raise_size
                paid = self._pay_to_pot(actor, total_put_in)
                contrib[actor.name] += paid
                pot += paid
                betstate.append({"player": actor.name, "action": "raise", "amount": paid, "street": street})
                pending_response = True  # other must respond

            else:
                raise ValueError(f"Unknown action: {action}")

            actor, other = other, actor  # switch turns

    def showdown(self, pot: int) -> Player:
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
        return p1

    def play(self, max_hands: int = 1000, verbose: bool = False) -> None:
        hand_num = 0
        while hand_num < max_hands and self.next_hand():
            hand_num += 1
            self.hand_counter += 1
            assert self.board is not None

            # Record starting state for hand metadata
            starting_stacks = {
                self.player1.name: self.player1.stack,
                self.player2.name: self.player2.stack
            }
            button = self.player2.name if hand_num % 2 == 0 else self.player1.name

            pot = 0
            hand_winner: Optional[Player] = None
            tie = False
            all_rounds_history: List[Dict[str, Any]] = []
            went_to_showdown = False

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
                all_rounds_history.extend(betstate)

                if verbose:
                    print("Actions:")
                    for act in betstate:
                        player_name = act["player"]
                        action = act["action"]
                        amount = act["amount"]
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
                went_to_showdown = True
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

            # Build board runout
            board_dict: Dict[str, List[str]] = {}
            if len(self.board.cards) >= 3:
                board_dict["flop"] = [str(c) for c in self.board.cards[:3]]
            if len(self.board.cards) >= 4:
                board_dict["turn"] = [str(self.board.cards[3])]
            if len(self.board.cards) >= 5:
                board_dict["river"] = [str(self.board.cards[4])]

            # Build showdown info (only if hand went to showdown)
            showdown_dict: Optional[Dict[str, Optional[List[str]]]] = None
            if went_to_showdown:
                showdown_dict = {
                    self.player1.name: [str(c) for c in self.player1.hole],
                    self.player2.name: [str(c) for c in self.player2.hole]
                }

            # Store structured hand record
            hand_record = {
                "hand_id": self.hand_counter,
                "button": button,
                "starting_stacks": starting_stacks,
                "actions": all_rounds_history,
                "board": board_dict,
                "showdown": showdown_dict,
                "result": {
                    "winner": hand_winner.name if hand_winner and not tie else "tie",
                    "pot": pot
                }
            }
            self.hand_history.append(hand_record)

        if verbose:
            print("\n=== Game Over ===")
        print(f"Final stacks after {hand_num} hands: P1={self.player1.stack}, P2={self.player2.stack}")


    