"""
Radiomic signature classifier for prostate cancer risk stratification.

Reads the extracted feature CSV (output of extract_t2_radiomics.py), derives a
binary risk label from first-order / shape features (high volume + high entropy
= high-risk), then trains and cross-validates three classifiers:
  - Logistic Regression (L2)
  - Random Forest
  - XGBoost (or GradientBoosting fallback)

Outputs
-------
results/classification_report.csv   Per-model CV metrics (AUC, F1, accuracy)
results/figures/roc_curves.png      Overlaid ROC curves for all models
results/figures/confusion_matrix_<model>.png
"""
from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    auc,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier  # type: ignore

    _XGB = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )
except ImportError:
    _XGB = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)


MODELS: dict[str, object] = {
    "LogisticRegression": Pipeline(
        [("scaler", StandardScaler()), ("clf", LogisticRegression(C=0.1, max_iter=1000, random_state=42))]
    ),
    "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=2, random_state=42),
    "XGBoost": _XGB,
}


# ---------------------------------------------------------------------------
# Label derivation
# ---------------------------------------------------------------------------

def derive_labels(df: pd.DataFrame) -> pd.Series:
    """
    Binary risk label derived from radiomic features (no external annotations
    required).  High-risk (1) = above-median tumour volume AND above-median
    entropy — a clinically motivated proxy for aggressive morphology.
    """
    vol_col = next(
        (c for c in df.columns if "volume_mm3" in c or "volume_voxels" in c),
        None,
    )
    ent_col = next(
        (c for c in df.columns if "entropy" in c and "glcm" not in c),
        None,
    )

    if vol_col and ent_col:
        high_vol = df[vol_col] >= df[vol_col].median()
        high_ent = df[ent_col] >= df[ent_col].median()
        labels = (high_vol & high_ent).astype(int)
    elif vol_col:
        labels = (df[vol_col] >= df[vol_col].median()).astype(int)
    else:
        # Last resort: median split on first numeric column
        numeric = df.select_dtypes(include="number").columns
        labels = (df[numeric[0]] >= df[numeric[0]].median()).astype(int)

    logging.info(
        "Label distribution  →  high-risk: %d  |  low-risk: %d",
        labels.sum(),
        (labels == 0).sum(),
    )
    return labels


# ---------------------------------------------------------------------------
# Feature matrix
# ---------------------------------------------------------------------------

def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only numeric original_ radiomic columns; drop near-zero-variance."""
    cols = [c for c in df.columns if c.startswith("original_") and df[c].dtype != object]
    X = df[cols].copy()
    # Drop columns with >30 % NaN
    X = X.dropna(axis=1, thresh=int(0.7 * len(X)))
    X = X.fillna(X.median())
    # Drop near-constant columns (std < 1e-6)
    X = X.loc[:, X.std() > 1e-6]
    logging.info("Feature matrix: %d cases × %d features", *X.shape)
    return X


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def evaluate_model(name: str, model, X: np.ndarray, y: np.ndarray, cv: StratifiedKFold, figures_dir: Path) -> dict:
    proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)

    fpr, tpr, _ = roc_curve(y, proba)
    roc_auc = auc(fpr, tpr)
    accuracy = float((pred == y).mean())
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    logging.info(
        "%s  →  AUC=%.3f  Acc=%.3f  F1=%.3f  Prec=%.3f  Rec=%.3f",
        name, roc_auc, accuracy, f1, precision, recall,
    )

    # Confusion matrix figure
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(y, pred, ax=ax, colorbar=False,
                                            display_labels=["Low-risk", "High-risk"])
    ax.set_title(f"{name} — CV confusion matrix")
    fig.tight_layout()
    fig.savefig(figures_dir / f"confusion_matrix_{name.lower().replace(' ', '_')}.png", dpi=150)
    plt.close(fig)

    return {"model": name, "auc": roc_auc, "accuracy": accuracy, "f1": f1,
            "precision": precision, "recall": recall, "fpr": fpr, "tpr": tpr}


# ---------------------------------------------------------------------------
# ROC overlay figure
# ---------------------------------------------------------------------------

def plot_roc_overlay(results: list[dict], figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["#2dd4bf", "#f472b6", "#fb923c"]
    for res, color in zip(results, colors):
        ax.plot(res["fpr"], res["tpr"], color=color, lw=2,
                label=f"{res['model']}  (AUC = {res['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Receiver Operating Characteristic — 5-fold CV", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_facecolor("#0b1014")
    fig.patch.set_facecolor("#0b1014")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.legend(facecolor="#1a2530", labelcolor="white", fontsize=10)
    fig.tight_layout()
    fig.savefig(figures_dir / "roc_curves.png", dpi=150)
    plt.close(fig)
    logging.info("Saved ROC overlay → %s", figures_dir / "roc_curves.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train radiomic classifiers for prostate cancer risk stratification.")
    parser.add_argument("--features", default="results/t2_radiomics_features.csv",
                        help="Path to extracted radiomics CSV.")
    parser.add_argument("--figures-dir", default="results/figures")
    parser.add_argument("--output", default="results/classification_report.csv")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    features_path = Path(args.features)
    if not features_path.exists():
        logging.error("Feature CSV not found: %s\nRun extract_t2_radiomics.py first.", features_path)
        raise SystemExit(1)

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(features_path)
    logging.info("Loaded %d cases from %s", len(df), features_path)

    X_df = build_feature_matrix(df)
    y = derive_labels(X_df).values
    X = X_df.values

    if len(np.unique(y)) < 2:
        logging.error("Only one class present — cannot train a classifier.")
        raise SystemExit(1)

    cv = StratifiedKFold(n_splits=min(args.folds, int(np.bincount(y).min())), shuffle=True, random_state=42)

    results = []
    for name, model in MODELS.items():
        res = evaluate_model(name, model, X, y, cv, figures_dir)
        results.append(res)

    plot_roc_overlay(results, figures_dir)

    report_rows = [{k: v for k, v in r.items() if k not in ("fpr", "tpr")} for r in results]
    report = pd.DataFrame(report_rows).sort_values("auc", ascending=False)
    report.to_csv(args.output, index=False)
    logging.info("Saved classification report → %s", args.output)
    print("\n" + report.to_string(index=False))


if __name__ == "__main__":
    main()
