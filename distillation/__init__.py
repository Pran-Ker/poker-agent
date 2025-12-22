"""
Poker distillation module.

This module provides tools for distilling a teacher model (GPT-4o) into
a smaller student model (Qwen 8B) for poker playing.
"""

from .poker_dataset import PokerDataset, PokerDatasetBuilder, PokerEnv

__all__ = ["PokerDataset", "PokerDatasetBuilder", "PokerEnv"]
