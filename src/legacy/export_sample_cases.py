"""
Export 3 sample inference cases for the portfolio page.

Selects:
  - Case A: clearly ambiguous (lowest Dice)
  - Case B: borderline (Dice near median)
  - Case C: clearly clear (highest Dice)

Per case outputs:
  prostate-case-<id>-adc-slice.png   middle axial slice of ADC volume
  prostate-case-<id>-inference.json  prediction + top features + metadata

Upload both files to Supabase under the same bucket as spine-demo.
"""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
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

DATASET_ROOT = Path("E:/prostate158_train/prostate158_train")
DICE_THRESHOLD = 0.608
OUTPUT_DIR = Path("results/sample_cases")
FIGURES_DIR = Path("results/figures")


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c.startswith("original_") and df[c].dtype != object]
    X = df[cols].dropna(axis=1, thresh=int(0.7 * len(df)))
    X = X.fillna(X.median())
    return X.loc[:, X.std() > 1e-6]


def train_model(X: np.ndarray, y: np.ndarray):
    model = Pipeline([("scaler", StandardScaler()), ("clf", _XGB)])
    model.fit(X, y)
    return model


def pick_cases(var_df: pd.DataFrame) -> dict[str, str]:
    """Pick case IDs for low/mid/high Dice."""
    var_df = var_df.sort_values("dice").reset_index(drop=True)
    low = var_df.iloc[0]["case_id"]
    mid_idx = len(var_df) // 2
    mid = var_df.iloc[mid_idx]["case_id"]
    high = var_df.iloc[-1]["case_id"]
    return {"A": str(low).zfill(3), "B": str(mid).zfill(3), "C": str(high).zfill(3)}


def render_adc_slice(case_id: str, output_path: Path) -> None:
    """Save middle axial slice of ADC volume as PNG."""
    adc_path = DATASET_ROOT / "train" / case_id / "adc.nii.gz"
    if not adc_path.exists():
        raise FileNotFoundError(f"ADC not found: {adc_path}")

    img = nib.load(str(adc_path))
    data = img.get_fdata()
    mid_slice = data[:, :, data.shape[2] // 2]

    # Normalize and flip for display
    p1, p99 = np.percentile(mid_slice[mid_slice > 0], [1, 99]) if (mid_slice > 0).any() else (0, 1)
    mid_slice = np.clip(mid_slice, p1, p99)
    mid_slice = (mid_slice - p1) / max(p99 - p1, 1e-6)
    mid_slice = np.rot90(mid_slice)

    # Overlay tumor mask if available
    mask_path = DATASET_ROOT / "train" / case_id / "adc_tumor_reader1.nii.gz"
    mask_overlay = None
    if mask_path.exists():
        mask_data = nib.load(str(mask_path)).get_fdata()
        mask_slice = mask_data[:, :, mask_data.shape[2] // 2]
        mask_overlay = np.rot90(mask_slice > 0)

    fig, ax = plt.subplots(figsize=(4, 4), facecolor="black")
    ax.imshow(mid_slice, cmap="gray", interpolation="bilinear")
    if mask_overlay is not None and mask_overlay.any():
        overlay_rgba = np.zeros((*mask_overlay.shape, 4))
        overlay_rgba[mask_overlay, 0] = 0.18   # R
        overlay_rgba[mask_overlay, 1] = 0.83   # G (teal)
        overlay_rgba[mask_overlay, 2] = 0.75   # B
        overlay_rgba[mask_overlay, 3] = 0.45   # alpha
        ax.imshow(overlay_rgba, interpolation="bilinear")
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor="black", pad_inches=0)
    plt.close(fig)
    logging.info("Saved slice -> %s", output_path)


def feature_contributions(model: Pipeline, x_row: np.ndarray,
                           feature_names: list[str], top_n: int = 5) -> list[dict]:
    """Approximate per-feature contributions using coefficient * scaled value."""
    scaler = model.named_steps["scaler"]
    clf = model.named_steps["clf"]
    x_scaled = scaler.transform(x_row.reshape(1, -1))[0]

    # XGBoost / GBT: use feature importances weighted by scaled value
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        scores = importances * np.abs(x_scaled)
    else:
        coef = clf.coef_[0] if hasattr(clf, "coef_") else np.ones(len(feature_names))
        scores = coef * x_scaled

    top_idx = np.argsort(np.abs(scores))[::-1][:top_n]
    result = []
    for i in top_idx:
        name = feature_names[i]
        # Shorten name for display
        short = name.replace("original_", "").replace("firstorder_", "fo_") \
                    .replace("glcm_direction_", "glcm_d").replace("shape_", "sh_")
        result.append({
            "name": short,
            "raw_value": round(float(x_row[i]), 4),
            "contribution": round(float(scores[i]), 4),
        })
    return result


def build_inference_json(case_id: str, label: str, score: float,
                          dice: float, features: list[dict],
                          var_row: pd.Series) -> dict:
    return {
        "case_id": case_id,
        "predicted_label": "Ambiguous" if score >= 0.5 else "Clear",
        "ambiguity_score": round(score, 3),
        "label_groundtruth": label,
        "dice_reader1_reader2": round(dice, 3),
        "dice_threshold": DICE_THRESHOLD,
        "top_features": features,
        "reader1_volume_voxels": int(var_row.get("reader1_volume_voxels", 0)),
        "reader2_volume_voxels": int(var_row.get("reader2_volume_voxels", 0)),
        "hausdorff_distance_voxels": round(float(var_row.get("hausdorff_distance_voxels", 0)), 2),
        "modality": "ADC",
        "dataset": "Prostate158",
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    features_df = pd.read_csv("results/t2_radiomics_features.csv")
    var_df = pd.read_csv("results/adc_inter_reader_variability.csv")

    features_df["case_id"] = features_df["case_id"].astype(str).str.zfill(3)
    var_df["case_id"] = var_df["case_id"].astype(str).str.zfill(3)

    merged = features_df.merge(var_df[["case_id", "dice", "reader1_volume_voxels",
                                        "reader2_volume_voxels", "hausdorff_distance_voxels"]],
                                on="case_id", how="inner")
    merged["ambiguous"] = (merged["dice"] < DICE_THRESHOLD).astype(int)

    X_df = build_feature_matrix(merged)
    y = merged.loc[X_df.index, "ambiguous"].values
    X = X_df.values
    feature_names = list(X_df.columns)

    model = train_model(X, y)
    proba = model.predict_proba(X)[:, 1]
    merged_aligned = merged.loc[X_df.index].copy()
    merged_aligned["score"] = proba

    # Pick 3 representative cases
    cases_map = pick_cases(var_df[var_df["case_id"].isin(merged_aligned["case_id"])])
    case_labels = {"A": "Clearly Ambiguous", "B": "Borderline", "C": "Clearly Clear"}

    export_list = []
    for letter, case_id in cases_map.items():
        row = merged_aligned[merged_aligned["case_id"] == case_id]
        if row.empty:
            logging.warning("Case %s not in merged set, skipping.", case_id)
            continue

        row = row.iloc[0]
        var_row = var_df[var_df["case_id"] == case_id].iloc[0]
        x_row = X_df.loc[row.name].values
        score = float(row["score"])
        dice = float(row["dice"])

        # Slice PNG
        slice_path = OUTPUT_DIR / f"prostate-case-{letter.lower()}-adc-slice.png"
        try:
            render_adc_slice(case_id, slice_path)
        except FileNotFoundError as e:
            logging.warning("Skipping slice for %s: %s", case_id, e)

        # Feature contributions
        contribs = feature_contributions(model, x_row, feature_names)

        # Inference JSON
        inference = build_inference_json(
            case_id=case_id,
            label=case_labels[letter],
            score=score,
            dice=dice,
            features=contribs,
            var_row=var_row,
        )
        json_path = OUTPUT_DIR / f"prostate-case-{letter.lower()}-inference.json"
        with open(json_path, "w") as f:
            json.dump(inference, f, indent=2)
        logging.info("Saved inference -> %s", json_path)

        export_list.append({
            "letter": letter,
            "case_id": case_id,
            "dice": round(dice, 3),
            "score": round(score, 3),
            "predicted": "Ambiguous" if score >= 0.5 else "Clear",
            "slice_file": slice_path.name,
            "json_file": json_path.name,
        })

    print("\nExport summary:")
    print(pd.DataFrame(export_list).to_string(index=False))
    print(f"\nFiles saved to: {OUTPUT_DIR.resolve()}")
    print("Upload both .png and .json files to your Supabase bucket.")


if __name__ == "__main__":
    main()
