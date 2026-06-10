"""
Radiomic feature selection via LASSO (L1 logistic regression) and mutual
information ranking.

Reads ADC radiomic features + ADC inter-reader variability CSV, labels cases
by Dice inter-reader agreement (Dice < 0.608 → ambiguous), then:

  1. Runs LASSO with strict C values (logspace -3 to -0.5) so it genuinely
     zeros out uninformative features on this 67-case dataset.
  2. Reports the top surviving features (non-zero coefficients).
  3. Ranks ALL features by mutual information with the label.
  4. Saves two figures:
       results/figures/lasso_feature_importance.png
       results/figures/mutual_information_top20.png
  5. Saves results/selected_features.csv with LASSO coefficients + MI scores.
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
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DICE_THRESHOLD = 0.608

def load_merged(features_path: Path, variability_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Merge ADC features with ADC inter-reader Dice label."""
    feats = pd.read_csv(features_path)
    feats["case_id"] = feats["case_id"].astype(str).str.zfill(3)
    var = pd.read_csv(variability_path)
    var["case_id"] = var["case_id"].astype(str).str.zfill(3)
    merged = feats.merge(var[["case_id", "dice"]], on="case_id", how="inner")
    y = (merged["dice"] < DICE_THRESHOLD).astype(int).values
    logging.info("Merged: %d cases  ambiguous=%d  clear=%d", len(merged), y.sum(), (y == 0).sum())
    return merged, y


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c.startswith("original_") and df[c].dtype != object]
    X = df[cols].copy()
    X = X.dropna(axis=1, thresh=int(0.7 * len(X)))
    X = X.fillna(X.median())
    X = X.loc[:, X.std() > 1e-6]
    return X


def short_name(col: str) -> str:
    """Shorten original_firstorder_entropy → fo_entropy, etc."""
    col = col.replace("original_", "")
    replacements = {
        "firstorder_": "fo_",
        "shape_": "sh_",
        "glcm_": "glcm_",
        "glrlm_": "glrlm_",
        "glszm_": "glszm_",
        "ngtdm_": "ngtdm_",
        "gldm_": "gldm_",
    }
    for long, short in replacements.items():
        col = col.replace(long, short)
    # Truncate direction indices for GLCM to keep labels readable
    if col.startswith("glcm_direction_") and len(col) > 30:
        col = col[:30] + "…"
    return col


# ---------------------------------------------------------------------------
# LASSO selection
# ---------------------------------------------------------------------------

def run_lasso(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Force strict regularization: C in [0.001, 0.316] — much smaller than
    # sklearn's default range. With n=67 and p=168, weak C selects everything.
    strict_Cs = np.logspace(-3, -0.5, 20)
    lasso = LogisticRegressionCV(
        Cs=strict_Cs,
        cv=5,
        penalty="l1",
        solver="liblinear",
        max_iter=5000,
        random_state=42,
        scoring="roc_auc",
    )
    lasso.fit(X_scaled, y)
    coefs = lasso.coef_.ravel()

    df_lasso = pd.DataFrame({"feature": feature_names, "lasso_coef": coefs})
    df_lasso = df_lasso[df_lasso["lasso_coef"].abs() > 1e-6].copy()
    df_lasso = df_lasso.reindex(df_lasso["lasso_coef"].abs().sort_values(ascending=False).index)

    best_C = float(lasso.C_[0])
    logging.info("LASSO best C=%.4f  |  %d / %d features selected", best_C, len(df_lasso), len(feature_names))
    return df_lasso


def plot_lasso(df_lasso: pd.DataFrame, figures_dir: Path, top_n: int = 20) -> None:
    top = df_lasso.head(top_n).copy()
    top["short"] = top["feature"].apply(short_name)
    colors = ["#2dd4bf" if c > 0 else "#f472b6" for c in top["lasso_coef"]]

    fig, ax = plt.subplots(figsize=(8, max(4, len(top) * 0.35)))
    bars = ax.barh(top["short"][::-1], top["lasso_coef"][::-1], color=colors[::-1], height=0.65)
    ax.axvline(0, color="#555", lw=1)
    ax.set_xlabel("LASSO coefficient (standardised)", fontsize=11)
    ax.set_title(f"Top {len(top)} LASSO-selected radiomic features", fontsize=12)
    ax.set_facecolor("#0b1014")
    fig.patch.set_facecolor("#0b1014")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.tick_params(colors="white", labelsize=8)
    ax.xaxis.label.set_color("white")
    ax.title.set_color("white")
    # Colour legend
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#2dd4bf", label="Ambiguous ↑"),
                        Patch(color="#f472b6", label="Clear ↑")],
              facecolor="#1a2530", labelcolor="white", fontsize=9)
    fig.tight_layout()
    out = figures_dir / "lasso_feature_importance.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logging.info("Saved LASSO figure → %s", out)


# ---------------------------------------------------------------------------
# Mutual information ranking
# ---------------------------------------------------------------------------

def run_mutual_info(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    mi = mutual_info_classif(X, y, random_state=42)
    df_mi = pd.DataFrame({"feature": feature_names, "mutual_info": mi})
    df_mi = df_mi.sort_values("mutual_info", ascending=False).reset_index(drop=True)
    return df_mi


def plot_mutual_info(df_mi: pd.DataFrame, figures_dir: Path, top_n: int = 20) -> None:
    top = df_mi.head(top_n).copy()
    top["short"] = top["feature"].apply(short_name)

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    ax.barh(top["short"][::-1], top["mutual_info"][::-1], color="#a78bfa", height=0.65)
    ax.set_xlabel("Mutual information with inter-reader ambiguity label", fontsize=11)
    ax.set_title(f"Top {top_n} features by mutual information", fontsize=12)
    ax.set_facecolor("#0b1014")
    fig.patch.set_facecolor("#0b1014")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.tick_params(colors="white", labelsize=8)
    ax.xaxis.label.set_color("white")
    ax.title.set_color("white")
    fig.tight_layout()
    out = figures_dir / "mutual_information_top20.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logging.info("Saved MI figure → %s", out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LASSO + mutual information feature selection for radiomic signatures.")
    parser.add_argument("--features", default="results/adc_radiomics_features.csv")
    parser.add_argument("--variability", default="results/adc_inter_reader_variability.csv")
    parser.add_argument("--figures-dir", default="results/figures")
    parser.add_argument("--output", default="results/selected_features.csv")
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    features_path = Path(args.features)
    variability_path = Path(args.variability)
    if not features_path.exists():
        logging.error("Feature CSV not found: %s\nRun adc_pipeline.py first.", features_path)
        raise SystemExit(1)
    if not variability_path.exists():
        logging.error("Variability CSV not found: %s", variability_path)
        raise SystemExit(1)

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df, y = load_merged(features_path, variability_path)

    X_df = build_feature_matrix(df)
    # align y to X_df index (merge may have reordered rows)
    y = y[X_df.index] if len(y) == len(df) else y
    X = X_df.values
    feature_names = list(X_df.columns)

    if len(np.unique(y)) < 2:
        logging.error("Only one class present — cannot run feature selection.")
        raise SystemExit(1)

    # LASSO
    df_lasso = run_lasso(X, y, feature_names)
    plot_lasso(df_lasso, figures_dir, top_n=args.top_n)

    # Mutual information
    df_mi = run_mutual_info(X, y, feature_names)
    plot_mutual_info(df_mi, figures_dir, top_n=args.top_n)

    # Merge and save
    merged = df_mi.merge(df_lasso[["feature", "lasso_coef"]], on="feature", how="left")
    merged["lasso_selected"] = merged["lasso_coef"].notna()
    merged.to_csv(args.output, index=False)
    logging.info("Saved selected features → %s", args.output)

    print("\nTop 10 by mutual information:")
    print(merged.head(10)[["feature", "mutual_info", "lasso_coef", "lasso_selected"]].to_string(index=False))
    print(f"\nLASSO selected {df_lasso.shape[0]} features total:")
    if not df_lasso.empty:
        print(df_lasso.head(10)[["feature", "lasso_coef"]].to_string(index=False))


if __name__ == "__main__":
    main()
