"""
data_pipeline.py
CT Guanabara Futevolei — Performance Analysis Project
------------------------------------------------------
Handles data loading, merging, cleaning, and IPG calculation.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── PATHS ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# ── IPG WEIGHTS ───────────────────────────────────────────────────────────────
IPG_WEIGHTS = {
    "jump_height_norm": 0.25,
    "ball_control_norm": 0.20,
    "serve_accuracy_norm": 0.20,
    "reception_efficiency_norm": 0.20,
    "attack_rate_norm": 0.15,
}
MAX_JUMP = 80    # reference max (cm)
MAX_BC   = 80    # reference max (touches)


def load_raw() -> dict[str, pd.DataFrame]:
    """Load all raw CSV files and return as a dict of DataFrames."""
    files = {
        "profile": "athletes_profile.csv",
        "pre":     "pre_intervention.csv",
        "post":    "post_intervention.csv",
        "log":     "weekly_training_log.csv",
    }
    return {key: pd.read_csv(DATA_DIR / fname) for key, fname in files.items()}


def compute_ipg(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the General Performance Index (IPG) for each row."""
    df = df.copy()
    df["jump_height_norm"]         = df["jump_height_cm"]          / MAX_JUMP
    df["ball_control_norm"]        = df["ball_control_touches"]     / MAX_BC
    df["serve_accuracy_norm"]      = df["serve_accuracy_pct"]       / 100
    df["reception_efficiency_norm"]= df["reception_efficiency_pct"] / 100
    df["attack_rate_norm"]         = df["attack_rate_pct"]          / 100

    df["ipg_computed"] = (
        IPG_WEIGHTS["jump_height_norm"]          * df["jump_height_norm"]          +
        IPG_WEIGHTS["ball_control_norm"]         * df["ball_control_norm"]         +
        IPG_WEIGHTS["serve_accuracy_norm"]       * df["serve_accuracy_norm"]       +
        IPG_WEIGHTS["reception_efficiency_norm"] * df["reception_efficiency_norm"] +
        IPG_WEIGHTS["attack_rate_norm"]          * df["attack_rate_norm"]
    )
    return df


def detect_outliers_iqr(df: pd.DataFrame, col: str) -> pd.Series:
    """Return boolean mask of outliers using IQR method."""
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    return (df[col] < lower) | (df[col] > upper)


def clean(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """
    Clean a DataFrame:
    - Fill missing values with per-athlete median
    - Flag and cap outliers via IQR
    - Enforce correct dtypes
    """
    df = df.copy()

    # Missing value imputation — per-athlete median
    for col in numeric_cols:
        df[col] = df.groupby("athlete_id")[col].transform(
            lambda x: x.fillna(x.median())
        )

    # Outlier detection (flag only — no drop; coach reviews)
    for col in numeric_cols:
        mask = detect_outliers_iqr(df, col)
        if mask.any():
            print(f"[IQR] {mask.sum()} outlier(s) flagged in '{col}'")

    # Enforce types
    int_cols   = ["jump_height_cm", "ball_control_touches", "week",
                  "sessions_per_week", "rest_days"]
    float_cols = ["serve_accuracy_pct", "reception_efficiency_pct",
                  "attack_rate_pct", "ipg"]
    for c in int_cols:
        if c in df.columns:
            df[c] = df[c].astype(int)
    for c in float_cols:
        if c in df.columns:
            df[c] = df[c].astype(float)

    return df


def build_comparison() -> pd.DataFrame:
    """
    Build a wide-format DataFrame comparing pre (week 1) vs post (week 8)
    metrics for each athlete, including delta % calculations.
    """
    raw   = load_raw()
    pre   = clean(raw["pre"],  ["jump_height_cm","ball_control_touches",
                                 "serve_accuracy_pct","reception_efficiency_pct",
                                 "attack_rate_pct"])
    post  = clean(raw["post"], ["jump_height_cm","ball_control_touches",
                                 "serve_accuracy_pct","reception_efficiency_pct",
                                 "attack_rate_pct"])

    pre_w1  = pre[pre["week"]  == 1].set_index("athlete_id")
    post_w8 = post[post["week"] == 8].set_index("athlete_id")

    metrics = ["jump_height_cm", "ball_control_touches",
               "serve_accuracy_pct", "reception_efficiency_pct",
               "attack_rate_pct", "ipg"]

    comp = pd.DataFrame(index=pre_w1.index)
    comp["name"] = pre_w1["name"]

    for m in metrics:
        comp[f"{m}_pre"]  = pre_w1[m].values
        comp[f"{m}_post"] = post_w8[m].values
        comp[f"{m}_delta_pct"] = (
            (comp[f"{m}_post"] - comp[f"{m}_pre"]) / comp[f"{m}_pre"] * 100
        ).round(1)

    return comp.reset_index()


def build_full_timeline() -> pd.DataFrame:
    """Merge pre + post into a single timeline DataFrame with IPG computed."""
    raw  = load_raw()
    pre  = compute_ipg(raw["pre"])
    post = compute_ipg(raw["post"])
    full = pd.concat([pre, post], ignore_index=True)
    full = full.sort_values(["athlete_id", "week"]).reset_index(drop=True)
    return full


if __name__ == "__main__":
    comp = build_comparison()
    print("\n=== PRE vs POST COMPARISON ===")
    print(comp[["name", "ipg_pre", "ipg_post", "ipg_delta_pct"]].to_string(index=False))
    timeline = build_full_timeline()
    print(f"\nFull timeline: {len(timeline)} records across {timeline['athlete_id'].nunique()} athletes")
