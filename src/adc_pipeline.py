"""
ADC feature extraction and initial ambiguity model training.

Steps:
  1. Extract radiomics features (shape, first-order, GLCM) from ADC images
     using ADC reader-1 tumour masks. Features are cached to CSV on first run.
  2. Merge with ADC inter-reader Dice variability CSV.
  3. Train an XGBoost ambiguity classifier (Dice < threshold → ambiguous).
  4. Export 3 representative sample cases (A / B / C).

Note: For calibrated probability outputs suitable for portfolio visualisation,
use adc_pipeline_calibrated.py instead.

Run from project root:
    python src/adc_pipeline.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from radiomics_project.fallback_features import extract_fallback_features
from radiomics_project.dataset import case_dirs

try:
    from xgboost import XGBClassifier
    def make_clf():
        return XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, eval_metric="logloss", random_state=42,
        )
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    def make_clf():
        return GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── paths ────────────────────────────────────────────────────────────────────
DATASET_ROOT   = Path("E:/prostate158_train/prostate158_train")
RESULTS_DIR    = Path("results")
OUTPUT_DIR     = RESULTS_DIR / "sample_cases"
ADC_FEATS_CSV  = RESULTS_DIR / "adc_radiomics_features.csv"
VARIABILITY_CSV = RESULTS_DIR / "adc_inter_reader_variability.csv"
DICE_THRESHOLD = 0.608

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── STEP 1: extract ADC radiomics features ───────────────────────────────────
def extract_adc_features() -> pd.DataFrame:
    if ADC_FEATS_CSV.exists():
        log.info("ADC features CSV already exists — loading from %s", ADC_FEATS_CSV)
        return pd.read_csv(ADC_FEATS_CSV)

    log.info("Extracting ADC radiomics features (this takes ~5-15 min)…")
    cases = case_dirs(DATASET_ROOT, "train")
    rows = []
    for case_dir in tqdm(cases, desc="ADC radiomics"):
        adc_img  = case_dir / "adc.nii.gz"
        adc_mask = case_dir / "adc_tumor_reader1.nii.gz"
        if not adc_img.exists() or not adc_mask.exists():
            log.warning("Skipping %s — missing ADC image or mask", case_dir.name)
            continue
        try:
            feats = extract_fallback_features(adc_img, adc_mask)
            feats["case_id"] = case_dir.name
            rows.append(feats)
        except ValueError as e:
            log.warning("Skipping %s — %s", case_dir.name, e)
        except Exception as e:
            log.warning("Error on %s — %s", case_dir.name, e)

    df = pd.DataFrame(rows)
    df.to_csv(ADC_FEATS_CSV, index=False)
    log.info("Saved %d cases to %s", len(df), ADC_FEATS_CSV)
    return df


# ── STEP 2: build feature matrix ─────────────────────────────────────────────
def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c.startswith("original_") and df[c].dtype != object]
    X = df[cols].dropna(axis=1, thresh=int(0.7 * len(df)))
    X = X.fillna(X.median())
    return X.loc[:, X.std() > 1e-6]


# ── STEP 3: train classifier ──────────────────────────────────────────────────
def train(X: np.ndarray, y: np.ndarray) -> Pipeline:
    model = Pipeline([("scaler", StandardScaler()), ("clf", make_clf())])
    model.fit(X, y)
    return model


# ── STEP 4: render ADC slice PNG ──────────────────────────────────────────────
def render_slice(case_id: str, label: str) -> Path:
    case_dir  = DATASET_ROOT / "train" / case_id
    adc_path  = case_dir / "adc.nii.gz"
    mask_path = case_dir / "adc_tumor_reader1.nii.gz"

    img  = nib.load(str(adc_path)).get_fdata()
    mid  = img.shape[2] // 2
    sl   = img[:, :, mid]
    p1, p99 = np.percentile(sl[sl > 0], [1, 99]) if (sl > 0).any() else (0, 1)
    sl   = np.clip(sl, p1, p99)
    sl   = (sl - p1) / max(p99 - p1, 1e-6)
    sl   = np.rot90(sl)

    mask = nib.load(str(mask_path)).get_fdata()[:, :, mid]
    mask = np.rot90(mask > 0)

    fig, ax = plt.subplots(figsize=(4, 4), facecolor="black")
    ax.imshow(sl, cmap="gray", interpolation="bilinear")
    if mask.any():
        ov = np.zeros((*mask.shape, 4))
        ov[mask, 0] = 0.18; ov[mask, 1] = 0.83; ov[mask, 2] = 0.75; ov[mask, 3] = 0.45
        ax.imshow(ov, interpolation="bilinear")
    ax.axis("off")
    fig.tight_layout(pad=0)
    out = OUTPUT_DIR / f"prostate-case-{label}-adc-slice.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="black", pad_inches=0)
    plt.close(fig)
    return out


# ── STEP 5: build inference JSON for one case ─────────────────────────────────
def build_inference(
    case_id: str,
    label: str,
    groundtruth: str,
    model: Pipeline,
    X_df: pd.DataFrame,
    merged: pd.DataFrame,
    var: pd.DataFrame,
) -> dict:
    feature_names = list(X_df.columns)

    # locate row
    idx = X_df.index[merged.loc[X_df.index, "case_id"] == case_id]
    if len(idx) == 0:
        raise ValueError(f"Case {case_id} not found in feature matrix")
    x_row = X_df.loc[idx[0]].values

    score = float(model.predict_proba(x_row.reshape(1, -1))[0, 1])

    scaler = model.named_steps["scaler"]
    clf    = model.named_steps["clf"]
    x_sc   = scaler.transform(x_row.reshape(1, -1))[0]
    imp    = clf.feature_importances_ if hasattr(clf, "feature_importances_") else clf.coef_[0]
    contribs = imp * np.abs(x_sc)
    top_idx  = np.argsort(np.abs(contribs))[::-1][:5]

    top_features = []
    for i in top_idx:
        name = (feature_names[i]
                .replace("original_", "")
                .replace("firstorder_", "fo_")
                .replace("glcm_direction_", "glcm_d")
                .replace("shape_", "sh_"))
        top_features.append({
            "name": name,
            "raw_value": round(float(x_row[i]), 4),
            "contribution": round(float(contribs[i]), 4),
        })

    var_row = var[var["case_id"] == case_id].iloc[0]
    row     = merged[merged["case_id"] == case_id].iloc[0]

    return {
        "case_id": case_id,
        "predicted_label": "Ambiguous" if score >= 0.5 else "Clear",
        "ambiguity_score": round(score, 3),
        "label_groundtruth": groundtruth,
        "dice_reader1_reader2": round(float(row["dice"]), 3),
        "dice_threshold": DICE_THRESHOLD,
        "top_features": top_features,
        "reader1_volume_voxels": int(var_row["reader1_volume_voxels"]),
        "reader2_volume_voxels": int(var_row["reader2_volume_voxels"]),
        "hausdorff_distance_voxels": round(float(var_row["hausdorff_distance_voxels"]), 2),
        "modality": "ADC",
        "features_from": "ADC",
        "dataset": "Prostate158",
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main() -> None:
    # 1. features
    feats = extract_adc_features()
    feats["case_id"] = feats["case_id"].astype(str).str.zfill(3)

    # 2. variability
    var = pd.read_csv(VARIABILITY_CSV)
    var["case_id"] = var["case_id"].astype(str).str.zfill(3)

    # 3. merge + label
    merged = feats.merge(
        var[["case_id", "dice", "reader1_volume_voxels", "reader2_volume_voxels", "hausdorff_distance_voxels"]],
        on="case_id", how="inner",
    )
    merged["ambiguous"] = (merged["dice"] < DICE_THRESHOLD).astype(int)
    log.info("Merged dataset: %d cases  |  ambiguous=%d  clear=%d",
             len(merged), merged["ambiguous"].sum(), (merged["ambiguous"] == 0).sum())

    # 4. feature matrix
    X_df = build_feature_matrix(merged)
    y    = merged.loc[X_df.index, "ambiguous"].values
    log.info("Feature matrix: %d cases × %d features", *X_df.shape)

    # 5. train
    model = train(X_df.values, y)
    log.info("Model trained.")

    # 6. pick 3 sample cases
    dice_vals = merged.loc[X_df.index, "dice"].values
    case_ids  = merged.loc[X_df.index, "case_id"].values

    # Case A — lowest Dice (most ambiguous)
    case_a = case_ids[np.argmin(dice_vals)]
    # Case C — highest Dice (most clear)
    case_c = case_ids[np.argmax(dice_vals)]
    # Case B — borderline: 0.55 ≤ Dice ≤ 0.70, low Hausdorff, exclude A and C
    var_indexed = var.set_index("case_id")
    borderline_mask = (
        (dice_vals >= 0.55) & (dice_vals <= 0.70) &
        np.array([
            var_indexed.loc[cid, "hausdorff_distance_voxels"] < 20
            if cid in var_indexed.index else False
            for cid in case_ids
        ]) &
        (case_ids != case_a) & (case_ids != case_c)
    )
    if borderline_mask.any():
        # prefer the one with Dice closest to threshold
        bl_dice = dice_vals.copy()
        bl_dice[~borderline_mask] = np.nan
        case_b = case_ids[np.nanargmin(np.abs(bl_dice - DICE_THRESHOLD))]
    else:
        # fallback: median Dice
        med_idx = np.argsort(np.abs(dice_vals - np.median(dice_vals)))[0]
        case_b = case_ids[med_idx]

    log.info("Cases selected:  A=%s  B=%s  C=%s", case_a, case_b, case_c)

    # 7. build & save outputs
    case_map = {
        "a": (case_a, "Clearly Ambiguous"),
        "b": (case_b, "Borderline"),
        "c": (case_c, "Clearly Clear"),
    }

    for letter, (cid, groundtruth) in case_map.items():
        # PNG
        png_path = render_slice(cid, letter)
        log.info("Saved slice: %s", png_path)

        # JSON
        inference = build_inference(cid, letter, groundtruth, model, X_df, merged, var)
        json_path = OUTPUT_DIR / f"prostate-case-{letter}-inference.json"
        with open(json_path, "w") as f:
            json.dump(inference, f, indent=2)
        log.info(
            "Case %s  id=%s  dice=%.3f  score=%.3f  pred=%s",
            letter.upper(), cid,
            inference["dice_reader1_reader2"],
            inference["ambiguity_score"],
            inference["predicted_label"],
        )
        print(json.dumps(inference, indent=2))

    log.info("Done. Upload the 6 files in %s to Supabase.", OUTPUT_DIR)


if __name__ == "__main__":
    main()
