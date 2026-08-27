import importlib.util
from pathlib import Path

from direction_learning.dataset import DirectionExample
from direction_learning.tokenization import reverse_complement


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "direction_learning" / "carbon-model" / "run.py"

spec = importlib.util.spec_from_file_location("carbon_run", MODULE_PATH)
carbon_run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(carbon_run)


def test_early_stopping_updates_patience_and_trigger():
	best_metric, patience_counter, should_stop = carbon_run._update_early_stopping_state(
		current_metric=0.62,
		best_metric=0.60,
		patience_counter=0,
		patience=2,
		min_delta=0.0,
	)
	assert best_metric == 0.62
	assert patience_counter == 0
	assert should_stop is False

	best_metric, patience_counter, should_stop = carbon_run._update_early_stopping_state(
		current_metric=0.62,
		best_metric=0.62,
		patience_counter=1,
		patience=2,
		min_delta=0.0,
	)
	assert best_metric == 0.62
	assert patience_counter == 2
	assert should_stop is True


def test_split_size_aware_flag_changes_objective():
	per_example_subarrays = [{"shared"} for _ in range(100)]
	candidate = {
		"train": list(range(95)),
		"val": [],
		"test": list(range(95, 100)),
	}
	default_objective = carbon_run._split_candidate_objective(
		candidate,
		n_examples=100,
		target_test_fraction=0.10,
		optimize_target="test",
		per_example_subarrays=per_example_subarrays,
		size_aware=False,
	)
	size_aware_objective = carbon_run._split_candidate_objective(
		candidate,
		n_examples=100,
		target_test_fraction=0.10,
		optimize_target="test",
		per_example_subarrays=per_example_subarrays,
		size_aware=True,
	)

	assert default_objective["objective"] != size_aware_objective["objective"]
	assert size_aware_objective["objective"] > default_objective["objective"]


def test_split_group_keeps_reverse_complements_together():
	spacers = ["ACGT", "TTAA", "GGCC"]
	repeats = ["GTTT", "GTTT", "GTTT", "GTTT"]
	rc_spacers = [reverse_complement(sequence) for sequence in reversed(spacers)]
	rc_repeats = [reverse_complement(sequence) for sequence in reversed(repeats)]

	examples = [
		DirectionExample(
			array_name="forward",
			group_name="",
			agreement="",
			evor_direction="Forward",
			label=1,
			orientation_variant="native",
			source_variant="native",
			spacers=spacers,
			repeats=repeats,
			cas_subtype="I-B",
		),
		DirectionExample(
			array_name="reverse_complement",
			group_name="",
			agreement="",
			evor_direction="Reverse",
			label=0,
			orientation_variant="reverse_complement",
			source_variant="native",
			spacers=rc_spacers,
			repeats=rc_repeats,
			cas_subtype="I-B",
		),
	]

	components = carbon_run._build_group_components(examples)

	assert components == [[0, 1]]


def test_standard_split_keeps_reverse_complements_together_without_split_group():
	spacers = ["ACGT", "TTAA", "GGCC"]
	repeats = ["GTTT", "GTTT", "GTTT", "GTTT"]
	rc_spacers = [reverse_complement(sequence) for sequence in reversed(spacers)]
	rc_repeats = [reverse_complement(sequence) for sequence in reversed(repeats)]

	examples = [
		DirectionExample(
			array_name="forward",
			group_name="",
			agreement="",
			evor_direction="Forward",
			label=1,
			orientation_variant="native",
			source_variant="native",
			spacers=spacers,
			repeats=repeats,
			cas_subtype="I-B",
		),
		DirectionExample(
			array_name="reverse_complement",
			group_name="",
			agreement="",
			evor_direction="Reverse",
			label=0,
			orientation_variant="reverse_complement",
			source_variant="native",
			spacers=rc_spacers,
			repeats=rc_repeats,
			cas_subtype="I-B",
		),
		DirectionExample(
			array_name="other_a",
			group_name="",
			agreement="",
			evor_direction="Forward",
			label=1,
			orientation_variant="native",
			source_variant="native",
			spacers=["AAAA", "CCCC", "GGGG"],
			repeats=["TTTT", "TTTT", "TTTT", "TTTT"],
			cas_subtype="I-B",
		),
		DirectionExample(
			array_name="other_b",
			group_name="",
			agreement="",
			evor_direction="Reverse",
			label=0,
			orientation_variant="native",
			source_variant="native",
			spacers=["ATAT", "CGCG", "TATA"],
			repeats=["GCGC", "GCGC", "GCGC", "GCGC"],
			cas_subtype="I-B",
		),
	]

	for seed in range(25):
		splits = carbon_run._build_splits_once(
			examples,
			seed=seed,
			test_fraction=0.25,
			stratify_mode="cas_subtype_and_label",
			split_group=False,
		)
		membership: dict[int, str] = {}
		for split_name, indices in splits.items():
			for index in indices:
				membership[index] = split_name

		assert membership[0] == membership[1]