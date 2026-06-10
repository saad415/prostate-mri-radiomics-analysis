"""
ADC-only pipeline — calibrated model (clean fix for small dataset).

Uses XGBoost + isotonic calibration via CalibratedClassifierCV so that
predicted probabilities are meaningful on a 67-case dataset.

Run from project root:
    .venv\Scripts\python src\adc_pipeline_calibrated.py
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))

try:
    from xgboost import XGBClassifier
    _BASE_CLF = XGBClassifier(
        n_estimators=100, max_depth=2, learning_rate=0.05,
        subsample=0.8, eval_metric="logloss", random_state=42,
    )
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    _BASE_CLF = GradientBoostingClassifier(
        n_estimators=100, max_depth=2, random_state=42
    )

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── paths ─────────────────────────────────────────────────────────────────────
DATASET_ROOT    = Path("E:/prostate158_train/prostate158_train")
RESULTS_DIR     = Path("results")
OUTPUT_DIR      = RESULTS_DIR / "sample_cases"
ADC_FEATS_CSV   = RESULTS_DIR / "adc_radiomics_features.csv"
VARIABILITY_CSV = RESULTS_DIR / "adc_inter_reader_variability.csv"
DICE_THRESHOLD  = 0.608

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── load data ─────────────────────────────────────────────────────────────────
feats = pd.read_csv(ADC_FEATS_CSV)
feats["case_id"] = feats["case_id"].astype(str).str.zfill(3)
var = pd.read_csv(VARIABILITY_CSV)
var["case_id"] = var["case_id"].astype(str).str.zfill(3)

merged = feats.merge(
    var[["case_id", "dice", "reader1_volume_voxels",
         "reader2_volume_voxels", "hausdorff_distance_voxels"]],
    on="case_id", how="inner",
)
merged["ambiguous"] = (merged["dice"] < DICE_THRESHOLD).astype(int)
log.info("Dataset: %d cases  ambiguous=%d  clear=%d",
         len(merged), merged["ambiguous"].sum(), (merged["ambiguous"] == 0).sum())

# ── feature matrix ────────────────────────────────────────────────────────────
cols = [c for c in merged.columns
        if c.startswith("original_") and merged[c].dtype != object]
X_df = merged[cols].dropna(axis=1, thresh=int(0.7 * len(merged)))
X_df = X_df.fillna(X_df.median())
X_df = X_df.loc[:, X_df.std() > 1e-6]
y    = merged.loc[X_df.index, "ambiguous"].values
log.info("Feature matrix: %d × %d", *X_df.shape)

# ── calibrated model ──────────────────────────────────────────────────────────
# CalibratedClassifierCV with cv=5 fits 5 base models and learns isotonic
# calibration maps — gives reliable probabilities on small datasets.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
calibrated = CalibratedClassifierCV(_BASE_CLF, method="isotonic", cv=cv)
model = Pipeline([("scaler", StandardScaler()), ("clf", calibrated)])
model.fit(X_df.values, y)

# show score distribution
all_scores = model.predict_proba(X_df.values)[:, 1]
log.info("Score distribution — min=%.3f  median=%.3f  max=%.3f",
         all_scores.min(), float(np.median(all_scores)), all_scores.max())

# ── pick 3 sample cases ───────────────────────────────────────────────────────
case_ids  = merged.loc[X_df.index, "case_id"].values
dice_vals = merged.loc[X_df.index, "dice"].values
hausd     = merged.loc[X_df.index, "hausdorff_distance_voxels"].values

# Case A — highest ambiguity score (most ambiguous)
case_a = case_ids[np.argmax(all_scores)]
# Case C — lowest ambiguity score (most clear)
case_c = case_ids[np.argmin(all_scores)]

# Case B — score closest to 0.5 among cases NOT A or C,
#           with Dice > threshold (truly in "clear" class → genuinely borderline)
exclude = {case_a, case_c}
mask_b  = np.array([
    cid not in exclude and dice_vals[i] > DICE_THRESHOLD
    for i, cid in enumerate(case_ids)
])
if mask_b.any():
    scores_b = all_scores.copy()
    scores_b[~mask_b] = np.nan
    case_b = case_ids[np.nanargmin(np.abs(scores_b - 0.5))]
else:
    # fallback: score closest to 0.5 excluding A and C
    scores_b = all_scores.copy()
    scores_b[np.array([cid in exclude for cid in case_ids])] = np.nan
    case_b = case_ids[np.nanargmin(np.abs(scores_b - 0.5))]

log.info("Selected  A=%s (score=%.3f)  B=%s (score=%.3f)  C=%s (score=%.3f)",
         case_a, all_scores[case_ids == case_a][0],
         case_b, all_scores[case_ids == case_b][0],
         case_c, all_scores[case_ids == case_c][0])

# ── helpers ───────────────────────────────────────────────────────────────────
def render_slice(case_id: str, letter: str) -> Path:
    case_dir  = DATASET_ROOT / "train" / case_id
    img  = nib.load(str(case_dir / "adc.nii.gz")).get_fdata()
    mid  = img.shape[2] // 2
    sl   = img[:, :, mid]
    p1, p99 = np.percentile(sl[sl > 0], [1, 99]) if (sl > 0).any() else (0, 1)
    sl   = np.clip(sl, p1, p99)
    sl   = (sl - p1) / max(p99 - p1, 1e-6)
    sl   = np.rot90(sl)
    mask = nib.load(str(case_dir / "adc_tumor_reader1.nii.gz")).get_fdata()[:, :, mid]
    mask = np.rot90(mask > 0)

    fig, ax = plt.subplots(figsize=(4, 4), facecolor="black")
    ax.imshow(sl, cmap="gray", interpolation="bilinear")
    if mask.any():
        ov = np.zeros((*mask.shape, 4))
        ov[mask, 0] = 0.18; ov[mask, 1] = 0.83; ov[mask, 2] = 0.75; ov[mask, 3] = 0.45
        ax.imshow(ov, interpolation="bilinear")
    ax.axis("off")
    fig.tight_layout(pad=0)
    out = OUTPUT_DIR / f"prostate-case-{letter}-adc-slice.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="black", pad_inches=0)
    plt.close(fig)
    return out


def build_inference(case_id: str, letter: str, groundtruth: str) -> dict:
    idx   = np.where(case_ids == case_id)[0][0]
    x_row = X_df.iloc[idx].values
    score = float(model.predict_proba(x_row.reshape(1, -1))[0, 1])

    # feature contributions via scaler + base estimator importances
    # CalibratedClassifierCV wraps multiple estimators — average their importances
    scaler    = model.named_steps["scaler"]
    cal_clf   = model.named_steps["clf"]
    x_sc      = scaler.transform(x_row.reshape(1, -1))[0]
    estimators = cal_clf.calibrated_classifiers_
    all_imp   = []
    for est in estimators:
        base = est.estimator
        if hasattr(base, "feature_importances_"):
            all_imp.append(base.feature_importances_)
        elif hasattr(base, "coef_"):
            all_imp.append(np.abs(base.coef_[0]))
    imp      = np.mean(all_imp, axis=0) if all_imp else np.ones(len(x_sc))
    contribs = imp * np.abs(x_sc)
    top_idx  = np.argsort(contribs)[::-1][:5]

    feature_names = list(X_df.columns)
    top_features  = []
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


# ── export ────────────────────────────────────────────────────────────────────
case_map = {
    "a": (case_a, "Clearly Ambiguous"),
    "b": (case_b, "Borderline"),
    "c": (case_c, "Clearly Clear"),
}

for letter, (cid, groundtruth) in case_map.items():
    png = render_slice(cid, letter)
    log.info("Saved slice: %s", png)

    inf = build_inference(cid, letter, groundtruth)
    json_path = OUTPUT_DIR / f"prostate-case-{letter}-inference.json"
    with open(json_path, "w") as f:
        json.dump(inf, f, indent=2)

    log.info("Case %s  id=%s  dice=%.3f  score=%.3f  pred=%s",
             letter.upper(), cid,
             inf["dice_reader1_reader2"],
             inf["ambiguity_score"],
             inf["predicted_label"])
    print(json.dumps(inf, indent=2))

log.info("Done. Upload 6 files from %s to Supabase (overwrite).", OUTPUT_DIR)
