#!/usr/bin/env python3
"""
Train Qwen 8B to play poker via distillation from GPT-4o.

This script implements on-policy distillation where:
1. The student model (Qwen 8B) generates poker actions
2. The teacher model (GPT-4o) provides training signal via KL penalty
3. No ground-truth rewards are used - learning is purely from teacher

Usage:
    python3 distillation/train.py \
        --traces-file model_traces.jsonl \
        --log-path ./logs/poker_distillation \
        --num-batches 100
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add tinker-cookbook to path if needed
sys.path.insert(0, str(Path.home() / "Developer" / "tinker-cookbook"))

from tinker_cookbook.distillation.datasets import (
    CompositeDataset,
    DistillationDatasetConfig,
    TeacherConfig,
)
from tinker_cookbook.distillation.train_on_policy import Config, main

from distillation.poker_dataset import PokerDatasetBuilder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_poker_distillation_config(
    traces_file: str,
    log_path: str,
    student_model: str = "Qwen/Qwen2.5-8B-Instruct",
    teacher_model: str = "gpt-4o",
    groups_per_batch: int = 4,
    group_size: int = 4,
    learning_rate: float = 1e-5,
    kl_penalty_coef: float = 1.0,
    lora_rank: int = 32,
    max_tokens: int = 100,
    temperature: float = 1.0,
    num_batches: int | None = None,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
) -> Config:
    """
    Create configuration for poker distillation training.

    Args:
        traces_file: Path to model_traces.jsonl with poker trajectories
        log_path: Directory for logs and checkpoints
        student_model: Student model to train (default: Qwen2.5-8B-Instruct)
        teacher_model: Teacher model for KL penalty (default: gpt-4o)
        groups_per_batch: Number of poker states per batch
        group_size: Number of rollouts per state (for variance reduction)
        learning_rate: Learning rate for optimizer
        kl_penalty_coef: Coefficient for KL penalty (higher = stay closer to teacher)
        lora_rank: LoRA rank for efficient fine-tuning
        max_tokens: Maximum tokens to generate per action
        temperature: Sampling temperature
        num_batches: Number of training batches (None = full dataset)
        wandb_project: Weights & Biases project name (optional)
        wandb_name: Weights & Biases run name (optional)

    Returns:
        Config object for distillation training
    """

    # Create poker dataset builder
    poker_dataset_builder = PokerDatasetBuilder(
        traces_file=traces_file,
        groups_per_batch=groups_per_batch,
        group_size=group_size,
        model_name_for_tokenizer=student_model,
        renderer_name="json",  # Poker uses JSON output format
        train_fraction=0.95,
    )

    # Configure teacher model (GPT-4o from OpenAI)
    teacher_config = TeacherConfig(
        base_model=teacher_model,
        load_checkpoint_path=None,  # Using API model directly
    )

    # Create dataset config
    dataset_config = DistillationDatasetConfig(
        dataset_builder=poker_dataset_builder,
        teacher_config=teacher_config,
        groups_per_batch=groups_per_batch,
    )

    # Create training config
    config = Config(
        # Model configuration
        model_name=student_model,
        lora_rank=lora_rank,

        # Dataset configuration
        dataset_configs=[dataset_config],

        # Training hyperparameters
        learning_rate=learning_rate,
        kl_penalty_coef=kl_penalty_coef,
        kl_discount_factor=0.0,  # No discounting for poker (short episodes)

        # Sampling parameters
        max_tokens=max_tokens,
        temperature=temperature,

        # Loss function
        loss_fn="importance_sampling",
        num_substeps=1,

        # Logging and checkpointing
        log_path=log_path,
        eval_every=10,
        save_every=10,
        compute_post_kl=True,

        # Weights & Biases
        wandb_project=wandb_project,
        wandb_name=wandb_name,

        # Infrastructure
        base_url=None,  # Use default tinker service
        enable_trace=False,
    )

    return config


def validate_setup(traces_file: str, student_model: str):
    """Validate that required files and models exist."""
    # Check traces file
    if not Path(traces_file).exists():
        logger.error(f"Traces file not found: {traces_file}")
        logger.error("Run generate_trajectories.py first to create training data")
        sys.exit(1)

    # Count traces
    with open(traces_file) as f:
        num_traces = sum(1 for _ in f)

    logger.info(f"Found {num_traces} poker trajectories in {traces_file}")

    if num_traces < 100:
        logger.warning(
            f"Only {num_traces} trajectories found. "
            "Consider generating more with generate_trajectories.py"
        )

    return num_traces


def main_cli():
    parser = argparse.ArgumentParser(
        description="Train Qwen 8B to play poker via distillation from GPT-4o"
    )

    # Required arguments
    parser.add_argument(
        "--traces-file",
        type=str,
        default="model_traces.jsonl",
        help="Path to poker trajectories (default: model_traces.jsonl)"
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default="./logs/poker_distillation",
        help="Directory for logs and checkpoints (default: ./logs/poker_distillation)"
    )

    # Model configuration
    parser.add_argument(
        "--student-model",
        type=str,
        default="Qwen/Qwen2.5-8B-Instruct",
        help="Student model to train (default: Qwen/Qwen2.5-8B-Instruct)"
    )
    parser.add_argument(
        "--teacher-model",
        type=str,
        default="gpt-4o",
        help="Teacher model for KL penalty (default: gpt-4o)"
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=32,
        help="LoRA rank for efficient fine-tuning (default: 32)"
    )

    # Training hyperparameters
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate (default: 1e-5)"
    )
    parser.add_argument(
        "--kl-penalty",
        type=float,
        default=1.0,
        help="KL penalty coefficient (default: 1.0)"
    )
    parser.add_argument(
        "--groups-per-batch",
        type=int,
        default=4,
        help="Number of poker states per batch (default: 4)"
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=4,
        help="Number of rollouts per state (default: 4)"
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        help="Number of training batches (default: full dataset)"
    )

    # Sampling parameters
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens per action (default: 100)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)"
    )

    # Logging
    parser.add_argument(
        "--wandb-project",
        type=str,
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        help="Weights & Biases run name"
    )

    args = parser.parse_args()

    # Validate setup
    num_traces = validate_setup(args.traces_file, args.student_model)

    # Create log directory
    Path(args.log_path).mkdir(parents=True, exist_ok=True)

    # Create config
    logger.info("Creating distillation configuration...")
    config = create_poker_distillation_config(
        traces_file=args.traces_file,
        log_path=args.log_path,
        student_model=args.student_model,
        teacher_model=args.teacher_model,
        groups_per_batch=args.groups_per_batch,
        group_size=args.group_size,
        learning_rate=args.learning_rate,
        kl_penalty_coef=args.kl_penalty,
        lora_rank=args.lora_rank,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        num_batches=args.num_batches,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
    )

    # Log configuration
    logger.info("=" * 80)
    logger.info("Poker Distillation Training Configuration")
    logger.info("=" * 80)
    logger.info(f"Student Model:      {args.student_model}")
    logger.info(f"Teacher Model:      {args.teacher_model}")
    logger.info(f"Training Data:      {num_traces} trajectories from {args.traces_file}")
    logger.info(f"Batch Size:         {args.groups_per_batch} states × {args.group_size} rollouts")
    logger.info(f"Learning Rate:      {args.learning_rate}")
    logger.info(f"KL Penalty:         {args.kl_penalty}")
    logger.info(f"LoRA Rank:          {args.lora_rank}")
    logger.info(f"Log Path:           {args.log_path}")
    logger.info("=" * 80)

    # Run training
    logger.info("Starting distillation training...")
    asyncio.run(main(config))


if __name__ == "__main__":
    main_cli()
