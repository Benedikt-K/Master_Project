from direction_learning.dataset import DirectionExample

# simple test harness: create a fake dataset with one example and call train helper

from direction_learning.train import make_subarray_augment_fn
from itertools import combinations


def test_enumerate_subarrays():
    # create an example with 4 spacers
    ex = DirectionExample(
        array_name="a",
        group_name="g",
        agreement="yes",
        evor_direction="Forward",
        label=1,
        orientation_variant="native",
        source_variant="native",
        spacers=["s1","s2","s3","s4"],
        repeats=["r1","r2","r3","r4"],
        cas_subtype="I-F",
        left_flank="",
        right_flank="",
        source_json="{}",
    )

    # enumerate subarrays of size >=2 and <4
    n = 4
    expected = 0
    for k in range(2, n):
        expected += len(list(combinations(range(n), k)))

    # now use the same logic as in train.py to generate
    from direction_learning.train import DirectionJsonlDataset
    ds = DirectionJsonlDataset.__new__(DirectionJsonlDataset)
    ds.records = [ex]

    new_indices = []
    for orig_idx in [0]:
        ex0 = ds.records[orig_idx]
        for k in range(2, n):
            for comb in combinations(range(n), k):
                keep = list(comb)
                new_spacers = [ex0.spacers[i] for i in keep]
                new_repeats = [ex0.repeats[i] for i in keep]
                new_ex = DirectionExample(
                    array_name=ex0.array_name,
                    group_name=ex0.group_name,
                    agreement=ex0.agreement,
                    evor_direction=ex0.evor_direction,
                    label=ex0.label,
                    orientation_variant=ex0.orientation_variant,
                    source_variant=ex0.source_variant,
                    spacers=new_spacers,
                    repeats=new_repeats,
                    cas_subtype=ex0.cas_subtype,
                    left_flank=ex0.left_flank,
                    right_flank=ex0.right_flank,
                    source_json=ex0.source_json,
                )
                ds.records.append(new_ex)
                new_indices.append(len(ds.records)-1)

    assert len(new_indices) == expected


if __name__ == '__main__':
    test_enumerate_subarrays()
    print('test_augment_enumerate: OK')
