"""
Unit and integration tests for dataset downloading, validation, and offline reading.
"""

import os
import shutil
import tempfile
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
import pytest

from download_datasets import (
    DATASET_CONFIGS,
    DATASETS_ROOT,
    PROJECT_ROOT,
    download_and_verify_all,
    format_size,
    is_file_valid_parquet,
)


def test_format_size():
    assert format_size(500) == "500 B"
    assert "KB" in format_size(2048)
    assert "MB" in format_size(1024 * 1024 * 2)


def test_is_file_valid_parquet_edge_cases():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # 1. Non-existent file
        assert not is_file_valid_parquet(tmp_path / "nonexistent.parquet")

        # 2. 0-byte file
        empty_file = tmp_path / "empty.parquet"
        empty_file.touch()
        assert not is_file_valid_parquet(empty_file)

        # 3. Garbage text file
        garbage_file = tmp_path / "garbage.parquet"
        garbage_file.write_text("not a parquet file")
        assert not is_file_valid_parquet(garbage_file)

        # 4. Parquet file with corrupted internal data page
        sample_pq = (
            DATASETS_ROOT
            / "arc_challenge"
            / "ARC-Challenge"
            / "validation-00000-of-00001.parquet"
        )
        assert sample_pq.exists(), "Sample parquet file should exist"

        corrupted_pq = tmp_path / "corrupted.parquet"
        with open(sample_pq, "rb") as f:
            data = bytearray(f.read())
        # Corrupt bytes at offset 1000 inside compressed data block
        data[1000:1100] = b"\xff" * 100
        with open(corrupted_pq, "wb") as f:
            f.write(data)

        # pq.ParquetFile might succeed on metadata, but is_file_valid_parquet must fail!
        assert not is_file_valid_parquet(corrupted_pq)


def test_downloaded_files_exist_and_valid():
    """Verify all 6 splits across both datasets exist and have expected content."""
    expected_splits = {
        "CommonsenseQA": {
            "dir": DATASETS_ROOT / "commonsense_qa",
            "files": {
                "train": ("data/train-00000-of-00001.parquet", 9741),
                "validation": ("data/validation-00000-of-00001.parquet", 1221),
                "test": ("data/test-00000-of-00001.parquet", 1140),
            },
        },
        "ARC-Challenge": {
            "dir": DATASETS_ROOT / "arc_challenge",
            "files": {
                "train": ("ARC-Challenge/train-00000-of-00001.parquet", 1119),
                "validation": ("ARC-Challenge/validation-00000-of-00001.parquet", 299),
                "test": ("ARC-Challenge/test-00000-of-00001.parquet", 1172),
            },
        },
    }

    for ds_name, info in expected_splits.items():
        base_dir = info["dir"]
        for split, (rel_path, expected_rows) in info["files"].items():
            full_path = base_dir / rel_path
            assert full_path.is_file(), f"File {full_path} must exist"
            assert full_path.stat().st_size > 0, f"File {full_path} must not be empty"

            # Check with is_file_valid_parquet
            assert is_file_valid_parquet(full_path), f"File {full_path} must be valid Parquet"

            # Check PyArrow reading
            tbl = pq.read_table(str(full_path))
            assert tbl.num_rows == expected_rows, f"{split} row count mismatch"
            assert tbl.num_columns > 0

            # Check Pandas reading (offline capability)
            df = pd.read_parquet(full_path)
            assert len(df) == expected_rows
            assert "question" in df.columns
            assert "choices" in df.columns


def test_idempotent_execution():
    """Ensure running download_and_verify_all skips already downloaded valid files."""
    report = download_and_verify_all()
    assert report["CommonsenseQA"]["success"] is True
    assert report["ARC-Challenge"]["success"] is True

    for ds_name in ["CommonsenseQA", "ARC-Challenge"]:
        for file_info in report[ds_name]["files"]:
            assert file_info["status"] == "Skipped (Already exists and valid)"
            assert file_info["readable_offline"] is True
            assert file_info["size_bytes"] > 0
            assert file_info["num_rows"] > 0


def test_corrupt_file_auto_recovery():
    """Verify that if a local file is corrupted, re-running recovers it cleanly."""
    target_file = (
        DATASETS_ROOT
        / "arc_challenge"
        / "ARC-Challenge"
        / "validation-00000-of-00001.parquet"
    )
    backup_file = target_file.with_suffix(".parquet.testbak")
    shutil.copy2(target_file, backup_file)

    try:
        # Corrupt data page inside target file
        with open(target_file, "r+b") as f:
            f.seek(1000)
            f.write(b"\xff" * 100)

        assert not is_file_valid_parquet(target_file)

        # Run download_and_verify_all to trigger auto-recovery
        report = download_and_verify_all()
        assert report["ARC-Challenge"]["success"] is True

        # Check that the target file was repaired
        val_info = next(
            f for f in report["ARC-Challenge"]["files"] if f["split"] == "validation"
        )
        assert val_info["status"] == "Downloaded successfully"
        assert val_info["readable_offline"] is True
        assert val_info["num_rows"] == 299

        # Confirm PyArrow and Pandas can read the repaired file
        tbl = pq.read_table(str(target_file))
        assert tbl.num_rows == 299
    finally:
        if backup_file.exists():
            backup_file.unlink()
