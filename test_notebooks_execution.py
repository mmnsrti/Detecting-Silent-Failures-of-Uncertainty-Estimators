"""
Tests to verify that the notebooks and outputs are completely and correctly generated.
"""

from pathlib import Path
import json
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHARTS_DIR = OUTPUTS_DIR / "charts"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


def test_numeric_features():
    path = OUTPUTS_DIR / "numeric_features.csv"
    assert path.is_file()
    df = pd.read_csv(path)
    assert len(df) == 14692
    assert set(["dataset", "split", "question_words", "question_chars", "avg_option_words", "option_count"]).issubset(df.columns)
    assert (df["dataset"] == "CommonsenseQA").sum() == 12102
    assert (df["dataset"] == "ARC-Challenge").sum() == 2590


def test_descriptive_statistics():
    stats_path = OUTPUTS_DIR / "descriptive_statistics.csv"
    assert stats_path.is_file()
    df = pd.read_csv(stats_path)
    assert len(df) == 8  # 2 datasets * 4 features
    assert "Range" in df.columns
    assert "Mean" in df.columns
    assert "Median" in df.columns
    assert "Q1" in df.columns
    assert "Q3" in df.columns
    assert "Q4_upper" in df.columns

    quartiles_path = OUTPUTS_DIR / "quartile_intervals.csv"
    assert quartiles_path.is_file()
    q_df = pd.read_csv(quartiles_path)
    assert len(q_df) == 8


def test_empirical_rule():
    path = OUTPUTS_DIR / "empirical_rule_vs_normal.csv"
    assert path.is_file()
    df = pd.read_csv(path)
    assert len(df) == 24  # 2 datasets * 4 features * 3 sigmas


def test_charts_generated():
    assert CHARTS_DIR.is_dir()
    png_files = list(CHARTS_DIR.glob("*.png"))
    assert len(png_files) == 16
    for f in png_files:
        assert f.stat().st_size > 5000  # valid image files


def test_normalized_features():
    norm_path = OUTPUTS_DIR / "normalized_features.csv"
    assert norm_path.is_file()
    df = pd.read_csv(norm_path)
    assert len(df) == 14692
    assert "question_words__minmax" in df.columns
    assert "question_words__zscore" in df.columns
    assert "question_words__decimal" in df.columns

    params_path = OUTPUTS_DIR / "normalization_parameters.csv"
    assert params_path.is_file()
    p_df = pd.read_csv(params_path)
    assert len(p_df) == 8


def test_classification_outputs():
    cls_dir = OUTPUTS_DIR / "classification"
    assert cls_dir.is_dir()
    assert (cls_dir / "classification_report_bbn.csv").is_file()
    assert (cls_dir / "confusion_matrix_bbn.csv").is_file()
    assert (cls_dir / "confusion_matrix_bbn.png").is_file()
    assert (cls_dir / "run_summary.json").is_file()
    assert (cls_dir / "test_predictions_bbn.csv").is_file()
    assert (cls_dir / "pearson_all.png").is_file()

    summary_file = cls_dir / "run_summary.json"
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    assert "accuracy" in summary
    assert "confusion_matrix" in summary
    assert summary["accuracy"] > 0.6


def test_notebooks_contain_outputs():
    candidate_nb4 = [
        "04_pearson_bbn_classification_corrected.ipynb",
        "04_pearson_bbn_classification_rewritten.ipynb",
        "04_pearson_bbn_classification.ipynb",
    ]
    found_nb4 = None
    for cand in candidate_nb4:
        if (NOTEBOOKS_DIR / cand).is_file():
            found_nb4 = cand
            break

    target_notebooks = [
        "01_prepare_numeric_features.ipynb",
        "02_statistics_and_distributions.ipynb",
        "03_three_normalizations.ipynb",
    ]
    if found_nb4:
        target_notebooks.append(found_nb4)

    for name in target_notebooks:
        nb_file = NOTEBOOKS_DIR / name
        assert nb_file.is_file()
        with open(nb_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        code_cells = [c for c in data["cells"] if c["cell_type"] == "code"]
        cells_with_outputs = [c for c in code_cells if len(c.get("outputs", [])) > 0]
        # Check that executed cells have outputs
        assert len(cells_with_outputs) >= len(code_cells) * 0.6
        assert len(cells_with_outputs) >= 3
