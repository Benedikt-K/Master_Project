from direction_learning.dataset import DirectionExample
from direction_learning.train import make_subarray_augment_fn


def run_test():
    example = DirectionExample(
        array_name="arr1",
        group_name="g1",
        agreement="yes",
        evor_direction="Forward",
        label=1,
        orientation_variant="native",
        source_variant="native",
        spacers=["s1", "s2", "s3", "s4", "s5"],
        repeats=["r1", "r2", "r3", "r4", "r5"],
        cas_subtype="I-F",
        left_flank="",
        right_flank="",
        source_json="{}",
    )

    fn = make_subarray_augment_fn(prob=1.0, seed=123)

    # run multiple times to observe different subsets
    seen_sizes = set()
    seen_orders_ok = True
    for _ in range(20):
        aug = fn(example)
        # must be a DirectionExample
        assert isinstance(aug, DirectionExample)
        # size must be between 1 and 4 (original 5, proper subset)
        n = len(aug.spacers)
        assert 1 <= n <= 4
        seen_sizes.add(n)
        # order must be preserved relative to original
        orig_positions = [example.spacers.index(s) for s in aug.spacers]
        if orig_positions != sorted(orig_positions):
            seen_orders_ok = False

    assert seen_orders_ok, "Order of spacers not preserved"
    assert len(seen_sizes) >= 2, "Augmentation did not produce variable sizes"


if __name__ == '__main__':
    run_test()
    print('test_augment_subarrays: OK')
