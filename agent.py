"""
Player agent for poker game.
Override the get_action method to implement custom strategies.
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Any, TYPE_CHECKING
import os
import json
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

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
        history: List[Tuple[str, str, int]],
        hand_history: List[List[Tuple[str, str, int]]],
    ) -> Action:
        """
        Model-based action selection (no truncation of history).

        Args:
            history: Betting history for current round [(player, action, amount), ...]
            hand_history: History of previous hands, each containing its betting history
                         [[(player, action, amount), ...], ...]
        """
        # Log what the agent sees before making decision
        self._log_game_state(to_call, min_raise, pot, board, round_idx, history)

        result = call_model(
            player_name=self.name,
            hole_cards=[str(c) for c in self.hole],
            stack=self.stack,
            to_call=to_call,
            min_raise=min_raise,
            pot=pot,
            board_view=board.get_board(),
            round_idx=round_idx,
            history=history,               # keep full
            hand_history=hand_history,     # keep full
        )

        action = result.get("action")
        amount = int(result.get("amount", 0) or 0)

        # Safety / validity clamps
        if action == "fold":
            self._log_action("fold", 0)
            return ("fold", 0)

        if action == "call":
            self._log_action("call", to_call)
            return ("call", 0)

        if action == "raise":
            # enforce min_raise on raise size
            if amount < min_raise:
                amount = min_raise
            self._log_action("raise", amount)
            return ("raise", amount)

        # Fallback if model returns something unexpected
        self._log_action("call (fallback)", to_call)
        return ("call", 0)

    def _log_game_state(
        self,
        to_call: int,
        min_raise: int,
        pot: int,
        board: Board,
        round_idx: int,
        history: List[Tuple[str, str, int]],
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
            history_str = ", ".join([f"{player}: {action}" + (f" ${amt}" if amt > 0 else "")
                                     for player, action, amt in history])
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



def call_model(
    *,
    player_name: str,
    hole_cards: List[str],
    stack: int,
    to_call: int,
    min_raise: int,
    pot: int,
    board_view: List[str],
    round_idx: int,
    history: List[Tuple[str, str, int]],
    hand_history: List[List[Tuple[str, str, int]]],
    model: str = "gpt-4o",
) -> Dict[str, Any]:
    """
    Calls an LLM using the standard OpenAI Chat Completions interface.

    Returns a dict like:
      {"action": "call", "amount": 0}
      {"action": "raise", "amount": 10}
      {"action": "fold", "amount": 0}

    Notes:
    - Does NOT truncate history or hand_history.
    - Uses JSON-only output to keep parsing robust.
    - You can swap model name via the 'model' parameter.
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # In a real system you'd raise; here we fail safe.
        return {"action": "call", "amount": 0}

    client = OpenAI(api_key=api_key)

    system_prompt = """You are a poker agent playing heads-up Texas Hold'em.
Your task: output the next action as strict JSON with keys:
- "action": one of ["fold","call","raise"]
- "amount": integer raise size if action=="raise", else 0

Rules:
- If you choose "raise", your amount MUST be >= min_raise.
- If you choose "call", that means match the required to_call (or check if to_call==0).
- Output JSON ONLY. No prose. No markdown.
"""

    # Keep full histories: no truncation.
    payload = {
        "player_name": player_name,
        "hole_cards": hole_cards,
        "stack": stack,
        "round_idx": round_idx,
        "board_view": board_view,
        "pot": pot,
        "to_call": to_call,
        "min_raise": min_raise,
        "current_round_history": history,
        "previous_hands_history": hand_history,
    }

    user_prompt = (
        "Choose the best next action.\n"
        "State is provided as JSON below:\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )


    console.print(f"Calling model {model} with payload: {payload}")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    content = resp.choices[0].message.content or "{}"
    data = json.loads(content)

    # Minimal normalization
    action = str(data.get("action", "call")).lower().strip()
    amount = int(data.get("amount", 0) or 0)

    if action not in ("fold", "call", "raise"):
        return {"action": "call", "amount": 0}

    if action != "raise":
        amount = 0
    return {"action": action, "amount": amount}