"""
Poker dataset adapter for tinker distillation.

This module provides a dataset that loads poker game states from model_traces.jsonl
and creates prompts for distillation training.
"""

import json
import logging
import math
from functools import partial
from typing import List, Literal, Sequence

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from tinker_cookbook.rl.types import (
    Action,
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)


class PokerEnv(ProblemEnv):
    """
    Environment for poker action selection during distillation.

    Unlike typical RL environments, this provides zero reward and only
    relies on KL penalty from teacher model for training signal.
    """

    def __init__(
        self,
        system_prompt: str,
        user_prompt: str,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
    ):
        # Set format_coef to 0 since we rely on KL penalty only
        super().__init__(renderer, convo_prefix, format_coef=0.0)
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

    def get_question(self) -> str:
        """Return the poker game state as the question."""
        return self.user_prompt

    def check_format(self, sample_str: str) -> bool:
        """
        Always return True - we don't enforce format during distillation.
        The teacher model provides the training signal.
        """
        return True

    def check_answer(self, sample_str: str) -> bool:
        """
        Always return False - no ground truth answer in poker.
        Training signal comes from teacher model KL penalty.
        """
        return False

    def get_reference_answer(self) -> str:
        """No reference answer for poker."""
        return ""

    async def step(self, action: Action) -> StepResult:
        """Return zero reward - training signal is KL penalty only."""
        message, parse_success = self.renderer.parse_response(action)
        return StepResult(
            reward=0.0,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={},
        )


class PokerDataset(RLDataset):
    """
    Dataset of poker game states for distillation.

    Loads prompts from model_traces.jsonl and provides them as
    training examples for distillation.
    """

    def __init__(
        self,
        traces: List[dict],
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        dataset_name: str = "poker",
    ):
        """
        Args:
            traces: List of trace dictionaries with 'input' and 'output' keys
            batch_size: Number of prompts per batch
            group_size: Number of rollouts per prompt (for variance reduction)
            renderer: Renderer for formatting prompts/responses
            dataset_name: Name for logging purposes
        """
        self.traces = traces
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.dataset_name = dataset_name

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        """Get a batch of environment builders for training."""
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.traces))
        assert batch_start < batch_end, "Incorrect batch size"

        builders = []
        for trace in self.traces[batch_start:batch_end]:
            # Extract messages from trace
            messages = trace["input"]
            system_prompt = ""
            user_prompt = ""

            for msg in messages:
                if msg["role"] == "system":
                    system_prompt = msg["content"]
                elif msg["role"] == "user":
                    user_prompt = msg["content"]

            # Create environment builder
            builders.append(
                ProblemGroupBuilder(
                    env_thunk=partial(
                        PokerEnv,
                        system_prompt,
                        user_prompt,
                        self.renderer,
                    ),
                    num_envs=self.group_size,
                    dataset_name=self.dataset_name,
                )
            )

        return builders

    def __len__(self) -> int:
        """Number of batches in the dataset."""
        return math.ceil(len(self.traces) / self.batch_size)


def load_poker_traces(
    filepath: str,
    split: Literal["train", "test"] = "train",
    train_fraction: float = 0.95,
) -> List[dict]:
    """
    Load poker traces from JSONL file.

    Args:
        filepath: Path to model_traces.jsonl
        split: Which split to load ('train' or 'test')
        train_fraction: Fraction of data to use for training

    Returns:
        List of trace dictionaries
    """
    traces = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                trace = json.loads(line)
                traces.append(trace)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line: {e}")
                continue

    # Split into train/test
    split_idx = int(len(traces) * train_fraction)

    if split == "train":
        return traces[:split_idx]
    else:
        return traces[split_idx:]


class PokerDatasetBuilder(RLDatasetBuilder):
    """
    Builder for poker distillation dataset.

    This loads poker game states from model_traces.jsonl and creates
    a dataset for distillation training.
    """

    def __init__(
        self,
        traces_file: str,
        groups_per_batch: int,
        group_size: int,
        model_name_for_tokenizer: str,
        renderer_name: str = "json",
        train_fraction: float = 0.95,
    ):
        """
        Args:
            traces_file: Path to model_traces.jsonl
            groups_per_batch: Number of prompts per training batch
            group_size: Number of rollouts per prompt
            model_name_for_tokenizer: Model name for tokenizer (e.g., "Qwen/Qwen2.5-8B-Instruct")
            renderer_name: Renderer to use (default: "json")
            train_fraction: Fraction of data for training (rest for validation)
        """
        self.traces_file = traces_file
        self.groups_per_batch = groups_per_batch
        self.group_size = group_size
        self.model_name_for_tokenizer = model_name_for_tokenizer
        self.renderer_name = renderer_name
        self.train_fraction = train_fraction

    async def __call__(self) -> tuple[PokerDataset, PokerDataset | None]:
        """Build train and optional test datasets."""
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        # Load traces
        logger.info(f"Loading poker traces from {self.traces_file}")
        train_traces = load_poker_traces(
            self.traces_file, split="train", train_fraction=self.train_fraction
        )
        test_traces = load_poker_traces(
            self.traces_file, split="test", train_fraction=self.train_fraction
        )

        logger.info(f"Loaded {len(train_traces)} train traces, {len(test_traces)} test traces")

        # Create datasets
        train_dataset = PokerDataset(
            traces=train_traces,
            batch_size=self.groups_per_batch,
            group_size=self.group_size,
            renderer=renderer,
            dataset_name="poker_train",
        )

        test_dataset = (
            PokerDataset(
                traces=test_traces,
                batch_size=self.groups_per_batch,
                group_size=1,  # Use group_size=1 for test
                renderer=renderer,
                dataset_name="poker_test",
            )
            if len(test_traces) > 0
            else None
        )

        return train_dataset, test_dataset
