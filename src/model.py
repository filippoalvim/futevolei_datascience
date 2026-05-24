"""
model.py
CT Guanabara Futevolei — Performance Analysis Project
------------------------------------------------------
Random Forest Regressor — training, cross-validation, and evaluation.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

import sys
sys.path.insert(0, str(Path(__file__).parent))
from feature_engineering import build_model_dataset, FEATURE_COLS, TARGET_COL

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUTPUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── MODEL CONFIG ─────────────────────────────────────────────────────────────
RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": 42,
}

COLORS = {
    "dark_green":  "#1B4332",
    "mid_green":   "#2D6A4F",
    "light_green": "#52B788",
    "pale_green":  "#D8F3DC",
    "orange":      "#E76F51",
    "gray":        "#6C757D",
}


def train_and_evaluate() -> dict:
    """
    Train Random Forest with k-fold cross-validation.
    Returns a summary dict of metrics.
    """
    X, y, _ = build_model_dataset()

    model = RandomForestRegressor(**RF_PARAMS)
    kf    = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_results = cross_validate(
        model, X, y, cv=kf,
        scoring=["neg_root_mean_squared_error", "r2"],
        return_train_score=True,
    )

    rmse_train = -cv_results["train_neg_root_mean_squared_error"]
    rmse_val   = -cv_results["test_neg_root_mean_squared_error"]
    r2_train   = cv_results["train_r2"]
    r2_val     = cv_results["test_r2"]

    summary = {
        "cv_folds": 5,
        "rmse_train_mean":  round(float(rmse_train.mean()), 4),
        "rmse_train_std":   round(float(rmse_train.std()),  4),
        "rmse_val_mean":    round(float(rmse_val.mean()),   4),
        "rmse_val_std":     round(float(rmse_val.std()),    4),
        "r2_train_mean":    round(float(r2_train.mean()),   4),
        "r2_train_std":     round(float(r2_train.std()),    4),
        "r2_val_mean":      round(float(r2_val.mean()),     4),
        "r2_val_std":       round(float(r2_val.std()),      4),
        "per_fold": {
            f"fold_{i+1}": {
                "rmse_train": round(float(rmse_train[i]),4),
                "rmse_val":   round(float(rmse_val[i]),  4),
                "r2_train":   round(float(r2_train[i]),  4),
                "r2_val":     round(float(r2_val[i]),    4),
            }
            for i in range(5)
        }
    }

    print("\n=== CROSS-VALIDATION RESULTS ===")
    print(f"RMSE  train : {summary['rmse_train_mean']:.4f} ± {summary['rmse_train_std']:.4f}")
    print(f"RMSE  val   : {summary['rmse_val_mean']:.4f}   ± {summary['rmse_val_std']:.4f}")
    print(f"R²    train : {summary['r2_train_mean']:.4f}   ± {summary['r2_train_std']:.4f}")
    print(f"R²    val   : {summary['r2_val_mean']:.4f}     ± {summary['r2_val_std']:.4f}")

    # Fit on full data for feature importance
    model.fit(X, y)
    fi = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    summary["feature_importance"] = fi.round(4).to_dict()

    # Save metrics
    with open(OUTPUT_DIR / "cv_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── PLOTS ─────────────────────────────────────────────────────────────────
    _plot_cv_results(rmse_train, rmse_val, r2_train, r2_val)
    _plot_feature_importance(fi)
    _plot_predictions(model, X, y)

    return summary, model, X, y


def _plot_cv_results(rmse_tr, rmse_v, r2_tr, r2_v):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    folds = [f"Fold {i+1}" for i in range(5)]

    # RMSE
    x = np.arange(5); w = 0.35
    axes[0].bar(x-w/2, rmse_tr, w, label="Train", color=COLORS["mid_green"])
    axes[0].bar(x+w/2, rmse_v,  w, label="Validation", color=COLORS["orange"])
    axes[0].set_xticks(x); axes[0].set_xticklabels(folds)
    axes[0].set_title("RMSE per Fold", fontweight="bold")
    axes[0].set_ylabel("RMSE"); axes[0].legend()
    axes[0].axhline(rmse_v.mean(), color=COLORS["orange"], linestyle="--", alpha=0.7,
                    label=f"Val mean: {rmse_v.mean():.3f}")

    # R²
    axes[1].bar(x-w/2, r2_tr, w, label="Train", color=COLORS["mid_green"])
    axes[1].bar(x+w/2, r2_v,  w, label="Validation", color=COLORS["orange"])
    axes[1].set_xticks(x); axes[1].set_xticklabels(folds)
    axes[1].set_title("R² per Fold", fontweight="bold")
    axes[1].set_ylabel("R²"); axes[1].legend()
    axes[1].set_ylim(0, 1.05)

    plt.suptitle("Random Forest — 5-Fold Cross-Validation", fontsize=14, fontweight="bold",
                 color=COLORS["dark_green"])
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cv_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: cv_results.png")


def _plot_feature_importance(fi: pd.Series):
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [COLORS["dark_green"] if i == 0 else COLORS["mid_green"] if i == 1
              else COLORS["light_green"] for i in range(len(fi))]
    bars = ax.barh(fi.index[::-1], fi.values[::-1], color=colors[::-1], edgecolor="white")
    for bar, val in zip(bars, fi.values[::-1]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val*100:.1f}%", va="center", fontsize=10, color=COLORS["dark_green"])
    ax.set_xlabel("Relative Importance", fontsize=11)
    ax.set_title("Feature Importance — Random Forest", fontsize=13,
                 fontweight="bold", color=COLORS["dark_green"])
    ax.set_xlim(0, fi.max() * 1.22)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: feature_importance.png")


def _plot_predictions(model, X, y):
    y_pred = model.predict(X)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y, y_pred, color=COLORS["mid_green"], alpha=0.75, edgecolors="white", s=80)
    lim = [min(y.min(), y_pred.min()) - 0.5, max(y.max(), y_pred.max()) + 0.5]
    ax.plot(lim, lim, "--", color=COLORS["orange"], lw=1.5, label="Perfect prediction")
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2   = r2_score(y, y_pred)
    ax.set_xlabel("Actual Δ IPG (%)", fontsize=11)
    ax.set_ylabel("Predicted Δ IPG (%)", fontsize=11)
    ax.set_title(f"Actual vs Predicted — RMSE: {rmse:.3f} | R²: {r2:.3f}",
                 fontsize=12, fontweight="bold", color=COLORS["dark_green"])
    ax.legend(); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "actual_vs_predicted.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: actual_vs_predicted.png")


if __name__ == "__main__":
    summary, model, X, y = train_and_evaluate()
    print("\n=== FEATURE IMPORTANCE ===")
    for feat, imp in summary["feature_importance"].items():
        print(f"  {feat:<30} {imp*100:.1f}%")
