"""
Script to download CommonsenseQA and ARC-Challenge datasets from Hugging Face Hub
and save them locally in the datasets/ directory.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

# Define project base directory and datasets folder
PROJECT_ROOT = Path(__file__).resolve().parent
DATASETS_ROOT = PROJECT_ROOT / "datasets"

# Dataset definitions
DATASET_CONFIGS = {
    "CommonsenseQA": {
        "repo_id": "tau/commonsense_qa",
        "target_dir": DATASETS_ROOT / "commonsense_qa",
        "splits": {
            "train": "data/train-00000-of-00001.parquet",
            "validation": "data/validation-00000-of-00001.parquet",
            "test": "data/test-00000-of-00001.parquet",
        },
    },
    "ARC-Challenge": {
        "repo_id": "allenai/ai2_arc",
        "target_dir": DATASETS_ROOT / "arc_challenge",
        "splits": {
            "train": "ARC-Challenge/train-00000-of-00001.parquet",
            "validation": "ARC-Challenge/validation-00000-of-00001.parquet",
            "test": "ARC-Challenge/test-00000-of-00001.parquet",
        },
    },
}


def is_file_valid_parquet(file_path: Path) -> bool:
    """Check if file exists, is non-empty, and can be fully read/decompressed as valid Parquet."""
    if not file_path.is_file() or file_path.stat().st_size == 0:
        return False
    try:
        # read_table fully reads and decompresses row groups, detecting data corruption
        table = pq.read_table(str(file_path))
        return table.num_rows > 0
    except Exception:
        return False


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB ({size_bytes:,} bytes)"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB ({size_bytes:,} bytes)"


def download_and_verify_all():
    DATASETS_ROOT.mkdir(parents=True, exist_ok=True)
    summary_report = {}

    for ds_name, config in DATASET_CONFIGS.items():
        repo_id = config["repo_id"]
        target_dir = config["target_dir"]
        target_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n==========================================")
        print(f"Dataset: {ds_name} (Repository: {repo_id})")
        print(f"Target Directory: {target_dir}")
        print(f"==========================================")

        ds_files = []
        total_ds_size = 0
        all_splits_ok = True

        for split, rel_filename in config["splits"].items():
            expected_local_path = target_dir / rel_filename
            already_downloaded = is_file_valid_parquet(expected_local_path)

            if already_downloaded:
                print(f"[{split.upper()}] Already fully downloaded: {expected_local_path}")
                download_status = "Skipped (Already exists and valid)"
                local_path = expected_local_path
            else:
                print(f"[{split.upper()}] Downloading {rel_filename} from {repo_id}...")
                try:
                    # Remove any corrupted or partial local file before re-downloading
                    if expected_local_path.exists():
                        try:
                            expected_local_path.unlink()
                        except Exception:
                            pass

                    downloaded_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=rel_filename,
                        repo_type="dataset",
                        local_dir=str(target_dir),
                        force_download=True,
                    )
                    local_path = Path(downloaded_file)
                    download_status = "Downloaded successfully"
                    print(f"[{split.upper()}] Downloaded to: {local_path}")
                except Exception as e:
                    print(f"[{split.upper()}] ERROR downloading: {e}")
                    download_status = f"Error: {e}"
                    all_splits_ok = False
                    rel_to_proj = (
                        str(expected_local_path.relative_to(PROJECT_ROOT).as_posix())
                        if expected_local_path.is_relative_to(PROJECT_ROOT)
                        else str(expected_local_path)
                    )
                    ds_files.append({
                        "split": split,
                        "rel_path": rel_to_proj,
                        "abs_path": str(expected_local_path),
                        "status": download_status,
                        "size_bytes": 0,
                        "readable_offline": False,
                        "num_rows": 0,
                        "num_cols": 0,
                    })
                    continue

            # Verify offline readability and extract file size
            try:
                table = pq.read_table(str(local_path))
                num_rows = table.num_rows
                num_cols = table.num_columns
                size_bytes = local_path.stat().st_size
                total_ds_size += size_bytes
                readable_offline = True
                print(f"[{split.upper()}] Verification: OK ({num_rows} rows, {num_cols} columns, {format_size(size_bytes)})")
            except Exception as e:
                readable_offline = False
                all_splits_ok = False
                size_bytes = local_path.stat().st_size if local_path.exists() else 0
                num_rows = 0
                num_cols = 0
                print(f"[{split.upper()}] Verification FAILED: {e}")

            rel_to_proj = (
                str(local_path.relative_to(PROJECT_ROOT).as_posix())
                if local_path.is_relative_to(PROJECT_ROOT)
                else str(local_path)
            )
            ds_files.append({
                "split": split,
                "rel_path": rel_to_proj,
                "abs_path": str(local_path),
                "status": download_status,
                "size_bytes": size_bytes,
                "readable_offline": readable_offline,
                "num_rows": num_rows,
                "num_cols": num_cols,
            })

        summary_report[ds_name] = {
            "success": all_splits_ok,
            "total_size_bytes": total_ds_size,
            "total_size_formatted": format_size(total_ds_size),
            "files": ds_files,
        }

    # Final summary printout
    print("\n" + "=" * 60)
    print("FINAL SUMMARY REPORT")
    print("=" * 60)
    for ds_name, data in summary_report.items():
        status_str = "SUCCESS" if data["success"] else "FAILED"
        print(f"\nDataset: {ds_name} -> Status: {status_str}")
        print(f"Total Dataset Size: {data['total_size_formatted']}")
        print("Files:")
        for f in data["files"]:
            print(f"  - Split: {f['split']}")
            print(f"    Path: {f['rel_path']}")
            print(f"    Absolute: {f['abs_path']}")
            print(f"    Size: {format_size(f['size_bytes'])}")
            print(f"    Offline Readable: {f['readable_offline']}")
            print(f"    Status: {f['status']}")

    return summary_report


if __name__ == "__main__":
    report = download_and_verify_all()
    all_ok = all(ds_info["success"] for ds_info in report.values())
    if not all_ok:
        sys.exit(1)
