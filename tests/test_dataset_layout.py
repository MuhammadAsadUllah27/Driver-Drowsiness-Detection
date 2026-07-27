import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train import ensure_dataset_layout


def test_ensure_dataset_layout_creates_train_and_val_folders(tmp_path):
    raw_dir = tmp_path / "_auto_captures"
    raw_dir.mkdir()
    (raw_dir / "awake_001.jpg").write_bytes(b"img")
    (raw_dir / "awake_002.jpg").write_bytes(b"img")
    (raw_dir / "drowsy_001.jpg").write_bytes(b"img")
    (raw_dir / "drowsy_002.jpg").write_bytes(b"img")

    ensure_dataset_layout(tmp_path)

    assert (tmp_path / "train" / "awake").exists()
    assert (tmp_path / "train" / "drowsy").exists()
    assert (tmp_path / "val" / "awake").exists()
    assert (tmp_path / "val" / "drowsy").exists()
