2. 
	3. Opponent summary: add uncertainty / sample size per stat

Your opponent_summary is fine in spirit, but make it confidence-aware:
	•	For each stat, store (value, n) or n_obs_*
Example:
	•	fold_to_flop_cbet: { "p": 0.62, "n": 34 }
	•	aggression_factor: { "v": 2.1, "n": 120 }
This prevents the model from overreacting to tiny samples.

1) history_spec (omit)

This is repeated natural-language schema description. It’s constant across all examples, so it’s pure token waste and a memorization magnet.
	•	Keep it in your system prompt or documentation, not per-example.

2) player_name (omit)

It’s irrelevant and can create spurious correlations if names differ.

3) round_idx (usually omit)

Unless your environment uses it for something meaningful (e.g., stack depth changes predictably by round), it’s just an index that can leak ordering patterns.


5) opponent_summary.note (omit)

Free-form text encourages the model to key off wording. Keep summaries numeric/structured.