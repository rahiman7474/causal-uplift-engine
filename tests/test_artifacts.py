import os
import json
from pathlib import Path
import pytest
import joblib


def test_artifacts_existence_and_size():
    artifact_dir = Path("artifacts")
    if not artifact_dir.exists():
        from src.train import run_pipeline
        run_pipeline(output_dir="artifacts", n_samples=5000)

    expected_files = [
        "models.joblib",
        "metrics.json",
        "balance_report.json",
        "curve_data.json"
    ]
    total_bytes = 0
    for filename in expected_files:
        fpath = artifact_dir / filename
        assert fpath.exists(), f"Missing required artifact: {filename}"
        total_bytes += fpath.stat().st_size

    total_mb = total_bytes / (1024 * 1024)
    # Verification requirement: artifacts must be under 10 MB
    assert total_mb < 10.0, f"Artifacts exceeded 10 MB limit: {total_mb:.2f} MB"


def test_artifacts_integrity():
    artifact_dir = Path("artifacts")
    if not (artifact_dir / "metrics.json").exists():
        from src.train import run_pipeline
        run_pipeline(output_dir="artifacts", n_samples=5000)

    # Test JSON files parse correctly
    with open(artifact_dir / "metrics.json") as f:
        metrics = json.load(f)
        assert "model_metrics" in metrics
        assert "roi_summary" in metrics

    with open(artifact_dir / "balance_report.json") as f:
        balance = json.load(f)
        assert "max_smd" in balance
        assert "covariates_smd" in balance

    with open(artifact_dir / "curve_data.json") as f:
        curves = json.load(f)
        assert "qini_curves" in curves
        assert "roi_curve" in curves
