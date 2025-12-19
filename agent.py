"""
Player agent for poker game.
Override the get_action method to implement custom strategies.
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Any, TYPE_CHECKING, Optional
import os
import json
from collections import defaultdict
from datetime import datetime
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from env.normalization import (
    normalize,
    denormalize,
    normalize_action_history,
    normalize_hand_history,
)

console = Console()

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
        history: List[Dict[str, Any]],
        hand_history: List[Dict[str, Any]],
        starting_stack: int,
    ) -> Action:
        """
        Model-based action selection with structured history and opponent stats.

        Args:
            history: Betting history for current round [{"player": ..., "action": ..., "amount": ..., "street": ...}, ...]
            hand_history: Structured history of previous hands with metadata
        """
        # Log what the agent sees before making decision
        self._log_game_state(to_call, min_raise, pot, board, round_idx, history)

        # Extract visible board cards (no X placeholders)
        visible_cards = [str(c) for i, c in enumerate(board.cards) if i < board.open_count]

        result = call_model(
            player_name=self.name,
            hole_cards=[str(c) for c in self.hole],
            stack=self.stack,
            to_call=to_call,
            min_raise=min_raise,
            pot=pot,
            board_cards=visible_cards,
            round_idx=round_idx,
            history=history,
            hand_history=hand_history,
            starting_stack=starting_stack,
        )

        action = result.get("action")
        # Denormalize: model returns fraction, convert back to chips
        normalized_amount = float(result.get("amount", 0) or 0)
        amount = denormalize(normalized_amount, starting_stack)

        # Safety / validity clamps
        if action == "fold":
            self._log_action("fold", 0)
            return ("fold", 0)

        if action == "call":
            self._log_action("call", to_call)
            return ("call", to_call)

        if action == "raise":
            # enforce min_raise on raise size
            if amount < min_raise:
                amount = min_raise
            self._log_action("raise", amount)
            return ("raise", amount)

        # Fallback if model returns something unexpected
        self._log_action("call (fallback)", to_call)
        return ("call", to_call)

    def _log_game_state(
        self,
        to_call: int,
        min_raise: int,
        pot: int,
        board: Board,
        round_idx: int,
        history: List[Dict[str, Any]],
    ) -> None:
        """Log the game state that the agent sees."""
        # Create a table for game state
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(style="cyan")
        table.add_column(style="white")

        # Round info
        rounds = ["Pre-Flop", "Flop", "Turn", "River"]
        round_name = rounds[round_idx] if round_idx < len(rounds) else f"Round {round_idx}"
        table.add_row("Round:", f"[bold]{round_name}[/bold]")

        # Player info
        table.add_row("Player:", f"[bold yellow]{self.name}[/bold yellow]")
        table.add_row("Hole Cards:", f"[bold green]{' '.join(str(c) for c in self.hole)}[/bold green]")
        table.add_row("Stack:", f"${self.stack}")

        # Board info
        board_cards = board.get_board()
        if board_cards:
            table.add_row("Board:", f"[bold magenta]{' '.join(board_cards)}[/bold magenta]")
        else:
            table.add_row("Board:", "[dim]No cards yet[/dim]")

        # Pot and betting info
        table.add_row("Pot:", f"[bold]${pot}[/bold]")
        table.add_row("To Call:", f"[bold red]${to_call}[/bold red]")
        table.add_row("Min Raise:", f"${min_raise}")

        # Current round history
        if history:
            history_str = ", ".join([
                f"{act['player']}: {act['action']}" + (f" ${act['amount']}" if act['amount'] > 0 else "")
                for act in history
            ])
            table.add_row("History:", history_str)

        console.print(Panel(table, title=f"[bold]{self.name}'s Turn[/bold]", border_style="blue"))

    def _log_action(self, action: str, amount: int) -> None:
        """Log the action chosen by the agent."""
        if action == "fold":
            action_text = f"[bold red]FOLD[/bold red]"
        elif action == "call":
            action_text = f"[bold yellow]CALL ${amount}[/bold yellow]"
        elif action.startswith("raise"):
            action_text = f"[bold green]RAISE to ${amount}[/bold green]"
        else:
            action_text = f"[bold]{action.upper()}[/bold]"

        console.print(f"  → {self.name} chose: {action_text}\n")

    def __str__(self) -> str:
        return f"{self.name}(stack={self.stack}, hole={[str(c) for c in self.hole]})"


def log_model_trace(input_data: Dict[str, Any], output_data: str, log_file: str = "model_traces.jsonl") -> None:
    """
    Log model input/output to JSONL for finetuning.

    Format: {"input": {...}, "output": {...}}
    One line per model call.
    """
    trace = {
        "input": input_data,
        "output": output_data
    }

    with open(log_file, "a") as f:
        f.write(json.dumps(trace, ensure_ascii=False) + "\n")


def summarize_opponent(hand_history: List[Dict[str, Any]], hero_name: str) -> Dict[str, Any]:
    """
    Compute opponent statistics from hand history.

    Args:
        hand_history: List of structured hand records
        hero_name: Name of the hero player (to filter out their actions)

    Returns:
        Dictionary of opponent statistics including:
        - hands_seen: total hands played
        - action_freq_by_street: frequency of fold/call/raise per street
        - avg_raise_amount: average raise size
        - aggression_factor: raises / calls ratio
        - fold_after_raise_rate: how often opponent raises then folds later
    """
    if not hand_history:
        return {
            "hands_seen": 0,
            "action_freq_by_street": {},
            "avg_raise_amount": 0.0,
            "aggression_factor": 0.0,
            "fold_after_raise_rate": 0.0,
            "note": "No history available"
        }

    hands_seen = len(hand_history)
    action_counts = defaultdict(lambda: defaultdict(int))  # street -> action -> count
    raise_amounts = []
    hands_where_opp_raised = 0
    hands_where_opp_raised_then_folded = 0
    total_raises = 0
    total_calls = 0

    for hand in hand_history:
        actions = hand.get("actions", [])
        opp_raised_this_hand = False
        opp_folded_this_hand = False

        for action in actions:
            player = action.get("player")
            if player == hero_name:
                continue  # Skip hero actions

            act_type = action.get("action")
            amount = action.get("amount", 0)
            street = action.get("street", "unknown")

            # Count action by street
            action_counts[street][act_type] += 1

            # Track raises
            if act_type == "raise":
                total_raises += 1
                opp_raised_this_hand = True
                if amount > 0:
                    raise_amounts.append(amount)
            elif act_type == "call":
                total_calls += 1
            elif act_type == "fold":
                opp_folded_this_hand = True

        # Track raise-then-fold pattern
        if opp_raised_this_hand:
            hands_where_opp_raised += 1
            if opp_folded_this_hand:
                hands_where_opp_raised_then_folded += 1

    # Compute aggregate stats
    avg_raise = sum(raise_amounts) / len(raise_amounts) if raise_amounts else 0.0
    aggression = total_raises / max(1, total_calls)
    fold_after_raise_rate = (hands_where_opp_raised_then_folded / max(1, hands_where_opp_raised)
                             if hands_where_opp_raised > 0 else 0.0)

    # Compute action frequencies by street
    action_freq_by_street = {}
    for street, counts in action_counts.items():
        total_actions = sum(counts.values())
        action_freq_by_street[street] = {
            action: count / total_actions
            for action, count in counts.items()
        }

    return {
        "hands_seen": hands_seen,
        "action_freq_by_street": action_freq_by_street,
        "avg_raise_amount": round(avg_raise, 2),
        "aggression_factor": round(aggression, 2),
        "fold_after_raise_rate": round(fold_after_raise_rate, 2),
        "total_opponent_raises": total_raises,
        "total_opponent_calls": total_calls
    }


def call_model(
    *,
    player_name: str,
    hole_cards: List[str],
    stack: int,
    to_call: int,
    min_raise: int,
    pot: int,
    board_cards: List[str],
    round_idx: int,
    history: List[Dict[str, Any]],
    hand_history: List[Dict[str, Any]],
    starting_stack: int,
    model: str = "gpt-4o",
) -> Dict[str, Any]:

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # In a real system you'd raise; here we fail safe.
        return {"action": "call", "amount": 0}

    client = OpenAI(api_key=api_key)

    system_prompt = """You are a poker agent playing heads-up Texas Hold'em.
Your goal is to maximize your stack through exploitative play.

Your task: output the next action as strict JSON with keys:
- "action": one of ["fold","call","raise"]
- "amount": raise size if action=="raise", else 0

IMPORTANT - Normalized Values:
All monetary values are normalized as fractions of starting stack (0.0 to 1.0+).
Examples: stack=0.9 means 90% of starting stack, pot=0.5 means pot is half starting stack.
When raising, output the fraction you want to raise (e.g., 0.2 = 20% of starting stack).

Rules:
- If you choose "raise", your amount MUST be >= min_raise.
- If you choose "call", that means match the required to_call (or check if to_call==0).
- Output JSON ONLY. No prose. No markdown.

Game State:
- street: Current betting round ("preflop", "flop", "turn", "river")
- board_cards: Visible community cards (empty for preflop, 3 for flop, 4 for turn, 5 for river)
- All amounts in history are normalized fractions of starting stack

History:
- previous_hands_recent: Recent hands with full context (actions, board, showdown, results)
- opponent_summary: Aggregate statistics about opponent tendencies
- Look for exploitable patterns: frequent bluffs, tight play, positional tendencies
"""

    # Compute opponent summary statistics
    opponent_summary = summarize_opponent(hand_history, player_name)

    # Send recent hands (last 20) + opponent summary instead of all history
    recent_hands = hand_history[-20:] if len(hand_history) > 20 else hand_history

    # Derive street name from round_idx
    street_names = ["preflop", "flop", "turn", "river"]
    street = street_names[round_idx] if round_idx < len(street_names) else f"round_{round_idx}"

    # Normalize all monetary values by starting stack
    payload = {
        "player_name": player_name,
        "hole_cards": hole_cards,
        "stack": normalize(stack, starting_stack),
        "street": street,
        "board_cards": board_cards,
        "pot": normalize(pot, starting_stack),
        "to_call": normalize(to_call, starting_stack),
        "min_raise": normalize(min_raise, starting_stack),
        "current_round_history": normalize_action_history(history, starting_stack),
        "previous_hands_recent": normalize_hand_history(recent_hands, starting_stack),
        "opponent_summary": opponent_summary,
    }

    user_prompt = (
        "Choose the best next action.\n"
        "State is provided as JSON below:\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )


    # console.print(f"Calling model {model} with payload: {payload}")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    content = resp.choices[0].message.content or "{}"

    # Log input/output for finetuning
    log_model_trace(
        input_data=messages,
        output_data=content
    )

    data = json.loads(content)

    # Minimal normalization
    action = str(data.get("action", "call")).lower().strip()
    amount = int(data.get("amount", 0) or 0)

    if action not in ("fold", "call", "raise"):
        return {"action": "call", "amount": 0}

    if action != "raise":
        amount = 0
    return {"action": action, "amount": amount}