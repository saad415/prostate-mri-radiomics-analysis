"""Regenerate Case B using case 090 — better borderline example."""
import json, logging
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

try:
    from xgboost import XGBClassifier
    _XGB = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                         subsample=0.8, use_label_encoder=False,
                         eval_metric="logloss", random_state=42)
except ImportError:
    _XGB = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DATASET_ROOT = Path("E:/prostate158_train/prostate158_train")
DICE_THRESHOLD = 0.608
CASE_ID = "090"
OUTPUT_DIR = Path("results/sample_cases")

def build_feature_matrix(df):
    cols = [c for c in df.columns if c.startswith("original_") and df[c].dtype != object]
    X = df[cols].dropna(axis=1, thresh=int(0.7*len(df)))
    X = X.fillna(X.median())
    return X.loc[:, X.std() > 1e-6]

feats = pd.read_csv("results/t2_radiomics_features.csv")
var   = pd.read_csv("results/adc_inter_reader_variability.csv")
feats["case_id"] = feats["case_id"].astype(str).str.zfill(3)
var["case_id"]   = var["case_id"].astype(str).str.zfill(3)

merged = feats.merge(var[["case_id","dice","reader1_volume_voxels",
                           "reader2_volume_voxels","hausdorff_distance_voxels"]],
                     on="case_id", how="inner")
merged["ambiguous"] = (merged["dice"] < DICE_THRESHOLD).astype(int)
X_df = build_feature_matrix(merged)
y    = merged.loc[X_df.index, "ambiguous"].values
X    = X_df.values
feature_names = list(X_df.columns)

model = Pipeline([("scaler", StandardScaler()), ("clf", _XGB)])
model.fit(X, y)

row     = merged[merged["case_id"] == CASE_ID].iloc[0]
var_row = var[var["case_id"] == CASE_ID].iloc[0]
x_idx   = X_df.index[merged.loc[X_df.index, "case_id"] == CASE_ID][0]
x_row   = X_df.loc[x_idx].values
score   = float(model.predict_proba(x_row.reshape(1,-1))[0,1])
dice    = float(row["dice"])

# Feature contributions
scaler = model.named_steps["scaler"]
clf    = model.named_steps["clf"]
x_sc   = scaler.transform(x_row.reshape(1,-1))[0]
imp    = clf.feature_importances_ if hasattr(clf,"feature_importances_") else clf.coef_[0]
scores = imp * np.abs(x_sc)
top_idx = np.argsort(np.abs(scores))[::-1][:5]
contribs = []
for i in top_idx:
    name = feature_names[i].replace("original_","").replace("firstorder_","fo_") \
                            .replace("glcm_direction_","glcm_d").replace("shape_","sh_")
    contribs.append({"name": name, "raw_value": round(float(x_row[i]),4),
                     "contribution": round(float(scores[i]),4)})

# Slice PNG
adc_path  = DATASET_ROOT / "train" / CASE_ID / "adc.nii.gz"
mask_path = DATASET_ROOT / "train" / CASE_ID / "adc_tumor_reader1.nii.gz"
img  = nib.load(str(adc_path)).get_fdata()
mid  = img.shape[2] // 2
sl   = img[:,:,mid]
p1,p99 = np.percentile(sl[sl>0],[1,99]) if (sl>0).any() else (0,1)
sl   = np.clip(sl,p1,p99); sl = (sl-p1)/max(p99-p1,1e-6)
sl   = np.rot90(sl)
mask = nib.load(str(mask_path)).get_fdata()[:,:,mid]
mask = np.rot90(mask > 0)

fig,ax = plt.subplots(figsize=(4,4), facecolor="black")
ax.imshow(sl, cmap="gray", interpolation="bilinear")
if mask.any():
    ov = np.zeros((*mask.shape,4))
    ov[mask,0]=0.18; ov[mask,1]=0.83; ov[mask,2]=0.75; ov[mask,3]=0.45
    ax.imshow(ov, interpolation="bilinear")
ax.axis("off"); fig.tight_layout(pad=0)
fig.savefig(OUTPUT_DIR/"prostate-case-b-adc-slice.png", dpi=150,
            bbox_inches="tight", facecolor="black", pad_inches=0)
plt.close(fig)
logging.info("Saved slice")

inference = {
    "case_id": CASE_ID,
    "predicted_label": "Ambiguous" if score >= 0.5 else "Clear",
    "ambiguity_score": round(score, 3),
    "label_groundtruth": "Borderline",
    "dice_reader1_reader2": round(dice, 3),
    "dice_threshold": DICE_THRESHOLD,
    "top_features": contribs,
    "reader1_volume_voxels": int(var_row["reader1_volume_voxels"]),
    "reader2_volume_voxels": int(var_row["reader2_volume_voxels"]),
    "hausdorff_distance_voxels": round(float(var_row["hausdorff_distance_voxels"]),2),
    "modality": "ADC", "dataset": "Prostate158",
}
with open(OUTPUT_DIR/"prostate-case-b-inference.json","w") as f:
    json.dump(inference, f, indent=2)
logging.info("Case 090  dice=%.3f  score=%.3f  label=%s", dice, score, inference["predicted_label"])
print(json.dumps(inference, indent=2))
