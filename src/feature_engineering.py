"""
feature_engineering.py
CT Guanabara Futevolei — Performance Analysis Project
------------------------------------------------------
Feature construction for the predictive model.
"""

import pandas as pd
import numpy as np
from data_pipeline import build_full_timeline, load_raw


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add previous-week IPG as a lag feature per athlete."""
    df = df.sort_values(["athlete_id", "week"]).copy()
    df["ipg_prev_week"] = df.groupby("athlete_id")["ipg_computed"].shift(1)
    df["ipg_rolling_mean_3w"] = (
        df.groupby("athlete_id")["ipg_computed"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )
    return df


def add_load_features(df: pd.DataFrame, log_df: pd.DataFrame) -> pd.DataFrame:
    """Merge training load features from the weekly log."""
    log_sel = log_df[["athlete_id", "week", "sessions_per_week",
                       "rest_days", "avg_session_duration_min", "focus_area"]].copy()
    df = df.merge(log_sel, on=["athlete_id", "week"], how="left",
                  suffixes=("", "_log"))
    return df


def add_bmi(df: pd.DataFrame, profile_df: pd.DataFrame) -> pd.DataFrame:
    """Merge BMI from athlete profile."""
    df = df.merge(profile_df[["athlete_id", "bmi", "experience_years"]],
                  on="athlete_id", how="left", suffixes=("", "_profile"))
    return df


def encode_focus(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode focus_area column."""
    if "focus_area" in df.columns:
        dummies = pd.get_dummies(df["focus_area"], prefix="focus", drop_first=True)
        df = pd.concat([df, dummies], axis=1)
    return df


def compute_delta_ipg(df: pd.DataFrame) -> pd.DataFrame:
    """Compute week-over-week IPG change as model target."""
    df = df.sort_values(["athlete_id", "week"]).copy()
    df["delta_ipg"] = df.groupby("athlete_id")["ipg_computed"].diff()
    df["delta_ipg_pct"] = (
        df["delta_ipg"] / df.groupby("athlete_id")["ipg_computed"].shift(1) * 100
    ).round(3)
    return df


FEATURE_COLS = [
    "sessions_per_week",
    "rest_days",
    "bmi",
    "ipg_prev_week",
    "experience_years",
]

TARGET_COL = "delta_ipg_pct"


def build_model_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """
    Build the final feature matrix (X) and target vector (y)
    ready for model training.

    Returns
    -------
    X : pd.DataFrame  — feature matrix
    y : pd.Series     — target (week-over-week IPG change %)
    """
    raw      = load_raw()
    timeline = build_full_timeline()
    timeline = add_lag_features(timeline)
    timeline = add_load_features(timeline, raw["log"])
    timeline = add_bmi(timeline, raw["profile"])
    timeline = encode_focus(timeline)
    timeline = compute_delta_ipg(timeline)

    # Drop week 1 (no lag available) and any rows with NaN in key cols
    model_df = timeline[timeline["week"] > 1].copy()
    model_df = model_df.dropna(subset=FEATURE_COLS + [TARGET_COL])

    X = model_df[FEATURE_COLS]
    y = model_df[TARGET_COL]

    print(f"Model dataset: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"Target range: [{y.min():.3f}, {y.max():.3f}]")
    return X, y, model_df


if __name__ == "__main__":
    X, y, df = build_model_dataset()
    print("\nFeature matrix preview:")
    print(X.head(10).to_string())
    print(f"\nCorrelation with target ({TARGET_COL}):")
    print(X.corrwith(y).round(3).sort_values(ascending=False))
