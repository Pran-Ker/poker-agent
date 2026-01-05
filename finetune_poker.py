#!/usr/bin/env python3
"""
Supervised fine-tuning for poker agent using Tinker API.

This script trains a model directly on poker examples using cross-entropy loss,
as opposed to distillation which uses KL penalty from a teacher model.

Usage:
    python3 finetune_poker.py \
        --data-file model_traces.jsonl \
        --log-path ./logs/poker_sft \
        --model-name Qwen/Qwen2.5-8B-Instruct \
        --num-epochs 3 \
        --learning-rate 1e-4
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.supervised.common import datum_from_model_input_weights
from tinker_cookbook.supervised.train import Config, main
from tinker_cookbook.supervised.types import (
    ChatDatasetBuilderCommonConfig,
    SupervisedDataset,
    SupervisedDatasetBuilder,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PokerSupervisedDataset(SupervisedDataset):
    """Dataset for supervised fine-tuning on poker data."""

    def __init__(
        self,
        examples: list[dict],
        batch_size: int,
        renderer: renderers.Renderer,
        max_length: int | None = None,
        dataset_name: str = "poker",
    ):
        """
        Args:
            examples: List of examples with 'input' (messages) and 'output' (action)
            batch_size: Number of examples per batch
            renderer: Renderer for formatting prompts/responses
            max_length: Maximum sequence length
            dataset_name: Name for logging
        """
        self.examples = examples
        self.batch_size = batch_size
        self.renderer = renderer
        self.max_length = max_length
        self.dataset_name = dataset_name
        self.shuffled_examples = examples.copy()

    def get_batch(self, index: int) -> list[tinker.Datum]:
        """Get a batch of training examples."""
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.shuffled_examples))

        batch_examples = self.shuffled_examples[start_idx:end_idx]
        datums = []

        for example in batch_examples:
            # Convert to conversation format expected by renderer
            messages = example["input"]
            output = example["output"]

            # Add the assistant response to the messages
            conversation = messages + [{"role": "assistant", "content": output}]

            # Build supervised example (train only on assistant response)
            model_input, weights = self.renderer.build_supervised_example(
                conversation,
                train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES
            )

            # Convert to Datum
            datum = datum_from_model_input_weights(model_input, weights, self.max_length)
            datums.append(datum)

        return datums

    def set_epoch(self, seed: int = 0):
        """Shuffle data for a new epoch."""
        import random
        random.seed(seed)
        self.shuffled_examples = self.examples.copy()
        random.shuffle(self.shuffled_examples)

    def __len__(self) -> int:
        """Number of batches in the dataset."""
        return (len(self.examples) + self.batch_size - 1) // self.batch_size


class PokerSupervisedDatasetBuilder(SupervisedDatasetBuilder):
    """Builder for poker supervised fine-tuning dataset."""

    def __init__(
        self,
        data_file: str,
        model_name: str,
        renderer_name: str = "tool_use",
        batch_size: int = 16,
        max_length: int | None = 8192,
        train_fraction: float = 0.95,
    ):
        """
        Args:
            data_file: Path to model_traces.jsonl
            model_name: Model name for tokenizer
            renderer_name: Renderer to use (tool_use for JSON outputs)
            batch_size: Batch size for training
            max_length: Maximum sequence length
            train_fraction: Fraction of data for training (rest for test)
        """
        self.data_file = data_file
        self.model_name = model_name
        self.renderer_name = renderer_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.train_fraction = train_fraction

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        """Build train and optional test datasets."""
        # Load tokenizer and renderer
        tokenizer = get_tokenizer(self.model_name)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        # Load poker data
        logger.info(f"Loading poker data from {self.data_file}")
        examples = []

        with open(self.data_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    example = json.loads(line)
                    # Validate format
                    if "input" not in example or "output" not in example:
                        logger.warning(f"Line {line_num}: Missing 'input' or 'output' field")
                        continue
                    examples.append(example)
                except json.JSONDecodeError as e:
                    logger.warning(f"Line {line_num}: Failed to parse JSON: {e}")
                    continue

        logger.info(f"Loaded {len(examples)} poker examples")

        if len(examples) == 0:
            raise ValueError(f"No valid examples found in {self.data_file}")

        # Split into train/test
        split_idx = int(len(examples) * self.train_fraction)
        train_examples = examples[:split_idx]
        test_examples = examples[split_idx:]

        logger.info(f"Train: {len(train_examples)} examples, Test: {len(test_examples)} examples")

        # Create datasets
        train_dataset = PokerSupervisedDataset(
            examples=train_examples,
            batch_size=self.batch_size,
            renderer=renderer,
            max_length=self.max_length,
            dataset_name="poker_train",
        )

        test_dataset = None
        if len(test_examples) > 0:
            test_dataset = PokerSupervisedDataset(
                examples=test_examples,
                batch_size=self.batch_size,
                renderer=renderer,
                max_length=self.max_length,
                dataset_name="poker_test",
            )

        return train_dataset, test_dataset


def validate_setup(data_file: str) -> int:
    """Validate that required files exist and return number of examples."""
    if not Path(data_file).exists():
        logger.error(f"Data file not found: {data_file}")
        logger.error("Generate training data first")
        sys.exit(1)

    # Count examples
    with open(data_file) as f:
        num_examples = sum(1 for line in f if line.strip())

    logger.info(f"Found {num_examples} poker examples in {data_file}")

    if num_examples < 100:
        logger.warning(
            f"Only {num_examples} examples found. "
            "Consider generating more training data."
        )

    return num_examples


def main_cli():
    parser = argparse.ArgumentParser(
        description="Supervised fine-tuning for poker agent using Tinker API"
    )

    # Required arguments
    parser.add_argument(
        "--data-file",
        type=str,
        default="model_traces.jsonl",
        help="Path to poker training data (default: model_traces.jsonl)"
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default="./logs/poker_sft",
        help="Directory for logs and checkpoints (default: ./logs/poker_sft)"
    )

    # Model configuration
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-8B-Instruct",
        help="Model to fine-tune (default: Qwen/Qwen2.5-8B-Instruct)"
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default=None,
        help="Load weights from checkpoint path (optional)"
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
        default=1e-4,
        help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (default: 16)"
    )

    # Dataset parameters
    parser.add_argument(
        "--max-length",
        type=int,
        default=8192,
        help="Maximum sequence length (default: 8192)"
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.95,
        help="Fraction of data for training (default: 0.95)"
    )
    parser.add_argument(
        "--renderer",
        type=str,
        default="tool_use",
        help="Renderer for formatting (default: tool_use)"
    )

    # Checkpointing and evaluation
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="Save checkpoint every N steps (default: 50)"
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=20,
        help="Evaluate every N steps (default: 20)"
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

    # Infrastructure
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Tinker service base URL (default: None, uses default)"
    )

    args = parser.parse_args()

    # Validate setup
    num_examples = validate_setup(args.data_file)

    # Create log directory
    Path(args.log_path).mkdir(parents=True, exist_ok=True)

    # Create dataset builder
    logger.info("Creating poker dataset builder...")
    dataset_builder = PokerSupervisedDatasetBuilder(
        data_file=args.data_file,
        model_name=args.model_name,
        renderer_name=args.renderer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        train_fraction=args.train_fraction,
    )

    # Create training config
    logger.info("Creating training configuration...")
    config = Config(
        # Paths and model
        log_path=args.log_path,
        model_name=args.model_name,
        load_checkpoint_path=args.load_checkpoint,

        # Dataset
        dataset_builder=dataset_builder,

        # Training hyperparameters
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        lora_rank=args.lora_rank,

        # Checkpointing and evaluation
        save_every=args.save_every,
        eval_every=args.eval_every,

        # Logging
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,

        # Infrastructure
        base_url=args.base_url,
    )

    # Log configuration
    logger.info("=" * 80)
    logger.info("Poker Supervised Fine-Tuning Configuration")
    logger.info("=" * 80)
    logger.info(f"Model:              {args.model_name}")
    logger.info(f"Training Data:      {num_examples} examples from {args.data_file}")
    logger.info(f"Batch Size:         {args.batch_size}")
    logger.info(f"Num Epochs:         {args.num_epochs}")
    logger.info(f"Learning Rate:      {args.learning_rate}")
    logger.info(f"LoRA Rank:          {args.lora_rank}")
    logger.info(f"Max Length:         {args.max_length}")
    logger.info(f"Renderer:           {args.renderer}")
    logger.info(f"Log Path:           {args.log_path}")
    logger.info("=" * 80)

    # Run training
    logger.info("Starting supervised fine-tuning...")
    asyncio.run(main(config))

    logger.info("=" * 80)
    logger.info("Training complete!")
    logger.info(f"Model checkpoints saved to: {args.log_path}/checkpoints/")
    logger.info("=" * 80)


if __name__ == "__main__":
    main_cli()
