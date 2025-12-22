#!/usr/bin/env python3
"""
Generate poker trajectories for distillation training.

This script runs multiple poker games and logs all LLM interactions
to model_traces.jsonl for use in distillation training.

Usage:
    python3 generate_trajectories.py --num-games 1000 --hands-per-game 10
"""

import argparse
import os
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from game import Game

console = Console()


def count_lines(filepath: str) -> int:
    """Count lines in a file efficiently."""
    if not os.path.exists(filepath):
        return 0
    with open(filepath, 'rb') as f:
        return sum(1 for _ in f)


def generate_trajectories(
    num_games: int,
    hands_per_game: int,
    starting_stack: int,
    output_file: str,
    seed_start: int = 0,
) -> None:
    """
    Generate poker trajectories by simulating games.

    Args:
        num_games: Number of games to simulate
        hands_per_game: Maximum hands per game (game may end earlier if player goes broke)
        starting_stack: Starting chip stack for each player
        output_file: Output file path for traces (will append)
        seed_start: Starting seed for reproducibility
    """

    # Count existing trajectories
    initial_count = count_lines(output_file)
    console.print(f"\n[cyan]Starting trajectory generation...[/cyan]")
    console.print(f"[dim]Existing trajectories: {initial_count}[/dim]")
    console.print(f"[dim]Target games: {num_games}[/dim]")
    console.print(f"[dim]Max hands per game: {hands_per_game}[/dim]")
    console.print(f"[dim]Output: {output_file}[/dim]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:

        task = progress.add_task(
            "[cyan]Generating games...",
            total=num_games
        )

        for game_idx in range(num_games):
            # Create game with unique seed for reproducibility
            seed = seed_start + game_idx
            game = Game(starting_cash=starting_stack, seed=seed)

            # Play game (verbose=False to avoid cluttering output)
            game.play(max_hands=hands_per_game, verbose=False)

            # Update progress
            progress.update(task, advance=1)

            # Log progress every 100 games
            if (game_idx + 1) % 100 == 0:
                current_count = count_lines(output_file)
                new_traces = current_count - initial_count
                progress.console.print(
                    f"[dim]Game {game_idx + 1}/{num_games}: "
                    f"Generated {new_traces} new trajectories[/dim]"
                )

    # Final count
    final_count = count_lines(output_file)
    new_trajectories = final_count - initial_count

    console.print(f"\n[green]✓ Generation complete![/green]")
    console.print(f"[cyan]New trajectories: {new_trajectories}[/cyan]")
    console.print(f"[cyan]Total trajectories: {final_count}[/cyan]")


def main():
    parser = argparse.ArgumentParser(
        description="Generate poker trajectories for distillation training"
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=1000,
        help="Number of games to simulate (default: 1000)"
    )
    parser.add_argument(
        "--hands-per-game",
        type=int,
        default=10,
        help="Maximum hands per game (default: 10)"
    )
    parser.add_argument(
        "--starting-stack",
        type=int,
        default=5000,
        help="Starting chip stack for each player (default: 5000)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="model_traces.jsonl",
        help="Output file for trajectories (default: model_traces.jsonl)"
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="Starting seed for reproducibility (default: 0)"
    )
    parser.add_argument(
        "--target-trajectories",
        type=int,
        help="Target number of total trajectories (estimates num_games needed)"
    )

    args = parser.parse_args()

    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[red]Error: OPENAI_API_KEY environment variable not set[/red]")
        console.print("[yellow]Set it with: export OPENAI_API_KEY='your-api-key'[/yellow]")
        return 1

    # If target_trajectories is specified, estimate num_games needed
    if args.target_trajectories:
        # Estimate: ~20-30 actions per game on average (depends on hands_per_game)
        # For conservative estimate, use 20 actions per game
        estimated_actions_per_game = args.hands_per_game * 2
        current_count = count_lines(args.output)
        remaining = args.target_trajectories - current_count

        if remaining <= 0:
            console.print(f"[green]Target already reached! Current: {current_count}[/green]")
            return 0

        estimated_games = max(1, remaining // estimated_actions_per_game)
        console.print(f"[yellow]Target: {args.target_trajectories} trajectories[/yellow]")
        console.print(f"[yellow]Current: {current_count} trajectories[/yellow]")
        console.print(f"[yellow]Need ~{remaining} more trajectories[/yellow]")
        console.print(f"[yellow]Estimated games needed: ~{estimated_games}[/yellow]\n")

        # Override num_games
        args.num_games = estimated_games

    generate_trajectories(
        num_games=args.num_games,
        hands_per_game=args.hands_per_game,
        starting_stack=args.starting_stack,
        output_file=args.output,
        seed_start=args.seed_start,
    )

    return 0


if __name__ == "__main__":
    exit(main())
