"""
Radiomic predictor of inter-reader segmentation ambiguity on prostate ADC MRI.

Label: Dice(reader1, reader2) < threshold → ambiguous tumour (1) else clear (0).
This is a clinically meaningful, non-circular label — the Dice score is an
independent ground truth derived from expert annotations, while features come
from ADC texture inside the tumour mask.

Hypothesis (relevant to autophagy-deficient cancer detection):
    Tumours with unique microenvironmental phenotypes (e.g. high necrosis,
    absent hypoxia) appear heterogeneous on ADC maps, causing reader disagreement.
    A radiomic signature of ambiguity may serve as a non-invasive proxy for
    such phenotypes — directly relevant to the MAASTRO ADC-detection goal.

Outputs
-------
results/ambiguity_classification_report.csv
results/figures/ambiguity_roc_curves.png
results/figures/ambiguity_lasso_features.png
results/figures/ambiguity_dice_distribution.png
results/figures/ambiguity_confusion_matrix_<model>.png
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
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import ConfusionMatrixDisplay, auc, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    _XGB = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                         subsample=0.8, use_label_encoder=False,
                         eval_metric="logloss", random_state=42)
except ImportError:
    _XGB = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42)

MODELS = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=0.1, max_iter=2000, random_state=42)),
    ]),
    "RandomForest": RandomForestClassifier(
        n_estimators=300, max_depth=5, min_samples_leaf=3, random_state=42),
    "XGBoost": _XGB,
}

DARK_BG = "#0b1014"
COLORS = ["#2dd4bf", "#f472b6", "#fb923c"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_and_merge(features_csv: Path, variability_csv: Path, dice_threshold: float) -> pd.DataFrame:
    feats = pd.read_csv(features_csv)
    var = pd.read_csv(variability_csv)

    # Normalise case_id: zero-pad to 3 digits
    feats["case_id"] = feats["case_id"].astype(str).str.zfill(3)
    var["case_id"] = var["case_id"].astype(str).str.zfill(3)

    merged = feats.merge(var[["case_id", "dice"]], on="case_id", how="inner")
    merged["ambiguous"] = (merged["dice"] < dice_threshold).astype(int)

    logging.info(
        "Merged dataset: %d cases  |  ambiguous (Dice < %.2f): %d  |  clear: %d",
        len(merged), dice_threshold,
        merged["ambiguous"].sum(), (merged["ambiguous"] == 0).sum(),
    )
    return merged


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c.startswith("original_") and df[c].dtype != object]
    X = df[cols].dropna(axis=1, thresh=int(0.7 * len(df)))
    X = X.fillna(X.median())
    X = X.loc[:, X.std() > 1e-6]
    logging.info("Feature matrix: %d cases x %d features", *X.shape)
    return X


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_dice_distribution(df: pd.DataFrame, threshold: float, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(df["dice"], bins=20, color="#2dd4bf", edgecolor="#0b1014", alpha=0.85)
    ax.axvline(threshold, color="#f472b6", lw=2, linestyle="--",
               label=f"Ambiguity threshold (Dice={threshold})")
    ax.set_xlabel("Dice score (reader1 vs reader2)", fontsize=11)
    ax.set_ylabel("Number of cases", fontsize=11)
    ax.set_title("Inter-reader Dice distribution — ADC tumour masks", fontsize=12)
    ax.legend(facecolor="#1a2530", labelcolor="white")
    ax.set_facecolor(DARK_BG); fig.patch.set_facecolor(DARK_BG)
    for spine in ax.spines.values(): spine.set_edgecolor("#333")
    ax.tick_params(colors="white"); ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white"); ax.title.set_color("white")
    fig.tight_layout()
    out = figures_dir / "ambiguity_dice_distribution.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    logging.info("Saved dice distribution -> %s", out)


def plot_roc_overlay(results: list[dict], figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for res, color in zip(results, COLORS):
        ax.plot(res["fpr"], res["tpr"], color=color, lw=2,
                label=f"{res['model']}  (AUC={res['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC — Predicting inter-reader ambiguity from ADC radiomics", fontsize=11)
    ax.legend(loc="lower right", facecolor="#1a2530", labelcolor="white", fontsize=10)
    ax.set_facecolor(DARK_BG); fig.patch.set_facecolor(DARK_BG)
    for spine in ax.spines.values(): spine.set_edgecolor("#333")
    ax.tick_params(colors="white"); ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white"); ax.title.set_color("white")
    fig.tight_layout()
    out = figures_dir / "ambiguity_roc_curves.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    logging.info("Saved ROC overlay -> %s", out)


def plot_lasso_features(feature_names: list[str], coefs: np.ndarray, figures_dir: Path, top_n: int = 15) -> None:
    df = pd.DataFrame({"feature": feature_names, "coef": coefs})
    df = df[df["coef"].abs() > 1e-6].sort_values("coef", key=abs, ascending=False).head(top_n)
    if df.empty:
        logging.warning("No LASSO features selected — skipping feature plot.")
        return

    short = df["feature"].str.replace("original_", "").str.replace("firstorder_", "fo_") \
                         .str.replace("glcm_", "glcm_").str.replace("shape_", "sh_")
    colors = ["#2dd4bf" if c > 0 else "#f472b6" for c in df["coef"]]

    fig, ax = plt.subplots(figsize=(8, max(3.5, len(df) * 0.38)))
    ax.barh(short.values[::-1], df["coef"].values[::-1], color=colors[::-1], height=0.65)
    ax.axvline(0, color="#555", lw=1)
    ax.set_xlabel("LASSO coefficient (standardised)", fontsize=11)
    ax.set_title("LASSO-selected ADC radiomic features for ambiguity prediction", fontsize=11)
    ax.set_facecolor(DARK_BG); fig.patch.set_facecolor(DARK_BG)
    for spine in ax.spines.values(): spine.set_edgecolor("#333")
    ax.tick_params(colors="white", labelsize=8); ax.xaxis.label.set_color("white")
    ax.title.set_color("white")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#2dd4bf", label="-> Ambiguous"),
                        Patch(color="#f472b6", label="-> Clear")],
              facecolor="#1a2530", labelcolor="white", fontsize=9)
    fig.tight_layout()
    out = figures_dir / "ambiguity_lasso_features.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    logging.info("Saved LASSO feature plot -> %s", out)


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------

def evaluate(name: str, model, X: np.ndarray, y: np.ndarray,
             cv: StratifiedKFold, figures_dir: Path) -> dict:
    proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y, proba)
    roc_auc = auc(fpr, tpr)
    acc = float((pred == y).mean())
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)

    logging.info("%s -> AUC=%.3f  Acc=%.3f  F1=%.3f  Prec=%.3f  Rec=%.3f",
                 name, roc_auc, acc, f1, prec, rec)

    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(
        y, pred, ax=ax, colorbar=False,
        display_labels=["Clear", "Ambiguous"])
    ax.set_title(f"{name} — confusion matrix")
    fig.tight_layout()
    fig.savefig(figures_dir / f"ambiguity_confusion_{name.lower().replace(' ','_')}.png", dpi=150)
    plt.close(fig)

    return {"model": name, "auc": roc_auc, "accuracy": acc, "f1": f1,
            "precision": prec, "recall": rec, "fpr": fpr, "tpr": tpr}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="results/t2_radiomics_features.csv",
                        help="Extracted radiomics CSV (T2 or ADC).")
    parser.add_argument("--variability", default="results/adc_inter_reader_variability.csv")
    parser.add_argument("--dice-threshold", type=float, default=0.5,
                        help="Dice < threshold -> ambiguous (label=1).")
    parser.add_argument("--figures-dir", default="results/figures")
    parser.add_argument("--output", default="results/ambiguity_classification_report.csv")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_merge(Path(args.features), Path(args.variability), args.dice_threshold)

    plot_dice_distribution(df, args.dice_threshold, figures_dir)

    X_df = build_feature_matrix(df)
    y = df.loc[X_df.index, "ambiguous"].values
    X = X_df.values
    feature_names = list(X_df.columns)

    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    if n_pos < 3 or n_neg < 3:
        logging.error(
            "Too few samples in one class (pos=%d, neg=%d). "
            "Try adjusting --dice-threshold.", n_pos, n_neg)
        raise SystemExit(1)

    n_folds = min(args.folds, n_pos)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # LASSO feature selection
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lasso = LogisticRegressionCV(Cs=20, cv=n_folds, penalty="l1",
                                  solver="liblinear", max_iter=2000,
                                  random_state=42, scoring="roc_auc")
    lasso.fit(X_scaled, y)
    plot_lasso_features(feature_names, lasso.coef_.ravel(), figures_dir)
    n_selected = int((lasso.coef_.ravel() != 0).sum())
    logging.info("LASSO selected %d / %d features  (C=%.4f)",
                 n_selected, len(feature_names), float(lasso.C_[0]))

    # Mutual information top features
    mi = mutual_info_classif(X, y, random_state=42)
    mi_df = pd.DataFrame({"feature": feature_names, "mutual_info": mi}) \
              .sort_values("mutual_info", ascending=False)
    logging.info("Top 5 features by MI:\n%s",
                 mi_df.head(5)[["feature", "mutual_info"]].to_string(index=False))

    # Train & evaluate classifiers
    results = []
    for name, model in MODELS.items():
        res = evaluate(name, model, X, y, cv, figures_dir)
        results.append(res)

    plot_roc_overlay(results, figures_dir)

    report = pd.DataFrame(
        [{k: v for k, v in r.items() if k not in ("fpr", "tpr")} for r in results]
    ).sort_values("auc", ascending=False)
    report["dice_threshold"] = args.dice_threshold
    report["n_ambiguous"] = n_pos
    report["n_clear"] = n_neg
    report["lasso_features_selected"] = n_selected
    report.to_csv(args.output, index=False)
    logging.info("Saved report -> %s", args.output)

    print("\n" + "=" * 65)
    print("  Ambiguity prediction from ADC radiomics — cross-validated")
    print("=" * 65)
    print(report[["model", "auc", "accuracy", "f1", "precision", "recall"]].to_string(index=False))
    print(f"\n  Dice threshold : {args.dice_threshold}")
    print(f"  Ambiguous cases: {n_pos}  |  Clear cases: {n_neg}")
    print(f"  LASSO features : {n_selected} / {len(feature_names)}")
    print("=" * 65)


if __name__ == "__main__":
    main()
