from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def numeric_feature_columns(features: pd.DataFrame) -> list[str]:
    excluded = {"case_id", "ID"}
    return [
        column
        for column in features.select_dtypes("number").columns
        if column not in excluded and features[column].notna().sum() >= 2 and features[column].nunique(dropna=True) > 1
    ]


def shorten_feature_name(name: str) -> str:
    return (
        name.replace("original_", "")
        .replace("firstorder_", "fo_")
        .replace("shape_", "shape_")
        .replace("glcm_direction_", "glcm_d")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create summary plots for extracted radiomics features.")
    parser.add_argument("--features", default="results/t2_radiomics_features.csv", help="Radiomics CSV path.")
    parser.add_argument("--figures-dir", default="results/figures", help="Directory for generated figures.")
    parser.add_argument("--top-n", type=int, default=12, help="Number of high-variance features to plot.")
    args = parser.parse_args()

    features = pd.read_csv(args.features)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    numeric_columns = numeric_feature_columns(features)
    if not numeric_columns:
        raise ValueError("Need at least two cases with varying numeric radiomics features to make plots.")

    if len(features) < 5:
        print(f"Warning: only {len(features)} cases found. Plots will be more useful with at least 10 cases.")

    variances = features[numeric_columns].var(numeric_only=True).sort_values(ascending=False)
    selected = variances.head(args.top_n).index.tolist()

    selected_data = features[selected].apply(pd.to_numeric, errors="coerce")
    z_scores = (selected_data - selected_data.mean()) / selected_data.std(ddof=0).replace(0, np.nan)
    z_scores = z_scores.rename(columns=shorten_feature_name)
    z_scores.insert(0, "case_id", features["case_id"].astype(str).str.zfill(3))

    melted = z_scores.melt(id_vars="case_id", var_name="feature", value_name="z_score").dropna()
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=melted, x="z_score", y="feature", color="#82b6d9")
    sns.stripplot(data=melted, x="z_score", y="feature", color="#263238", size=3, alpha=0.55)
    plt.axvline(0, color="#444444", linewidth=1, linestyle="--")
    plt.xlabel("standardized value across cases")
    plt.ylabel("feature")
    plt.title("T2 radiomics feature distributions")
    plt.tight_layout()
    plt.savefig(figures_dir / "t2_feature_distributions.png", dpi=200)
    plt.close()

    heatmap_data = z_scores.set_index("case_id")
    plt.figure(figsize=(12, max(5, len(features) * 0.35)))
    sns.heatmap(heatmap_data, cmap="vlag", center=0, robust=True)
    plt.xlabel("feature")
    plt.ylabel("case")
    plt.title("T2 radiomics feature z-score heatmap")
    plt.tight_layout()
    plt.savefig(figures_dir / "t2_feature_heatmap.png", dpi=200)
    plt.close()

    if len(features) >= 5:
        corr = selected_data.corr()
        corr.index = [shorten_feature_name(name) for name in corr.index]
        corr.columns = [shorten_feature_name(name) for name in corr.columns]
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, cmap="vlag", center=0, vmin=-1, vmax=1)
        plt.title("Radiomics feature correlation")
        plt.tight_layout()
        plt.savefig(figures_dir / "t2_feature_correlation.png", dpi=200)
        plt.close()
    else:
        print("Skipping correlation heatmap: need at least 5 cases for a meaningful correlation plot.")


if __name__ == "__main__":
    main()
