import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "Standalone" / "predict_direction.py"

spec = importlib.util.spec_from_file_location("predict_direction", MODULE_PATH)
predict_direction = importlib.util.module_from_spec(spec)
spec.loader.exec_module(predict_direction)


def test_load_input_objects_supports_fasta(tmp_path):
    fasta_path = tmp_path / "example.fna"
    fasta_path.write_text(
        ">array_1\n"
        "ACGTACGTACGTACGTACGTACGT\n"
        "CCCCGGGGTTTTAAAA\n"
        ">array_2\n"
        "TTTTAAAAGGGGCCCCNNNN\n"
        "ACGTACGTACGTACGT\n"
        "GCGCGCGCATATATAT\n"
    )

    objects = predict_direction.load_input_objects(fasta_path)

    assert len(objects) == 2
    assert objects[0]["repeats"] == []
    assert objects[0]["spacers"] == [
        "ACGTACGTACGTACGTACGTACGT",
        "CCCCGGGGTTTTAAAA",
    ]
    assert objects[1]["repeats"] == []
    assert objects[1]["spacers"] == [
        "TTTTAAAAGGGGCCCCNNNN",
        "ACGTACGTACGTACGT",
        "GCGCGCGCATATATAT",
    ]
