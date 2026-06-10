"""
Find the best borderline Case B from ADC features.
Criteria: Dice just above threshold (0.608-0.75), model score 0.25-0.75.
Run: .venv\Scripts\python src\pick_case_b_adc.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from radiomics_project.fallback_features import extract_fallback_features

try:
    from xgboost import XGBClassifier
    def make_clf():
        return XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                             subsample=0.8, eval_metric="logloss", random_state=42)
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    def make_clf():
        return GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DICE_THRESHOLD = 0.608

feats = pd.read_csv("results/adc_radiomics_features.csv")
feats["case_id"] = feats["case_id"].astype(str).str.zfill(3)
var   = pd.read_csv("results/adc_inter_reader_variability.csv")
var["case_id"] = var["case_id"].astype(str).str.zfill(3)

merged = feats.merge(
    var[["case_id","dice","reader1_volume_voxels","reader2_volume_voxels","hausdorff_distance_voxels"]],
    on="case_id", how="inner"
)
merged["ambiguous"] = (merged["dice"] < DICE_THRESHOLD).astype(int)

cols = [c for c in merged.columns if c.startswith("original_") and merged[c].dtype != object]
X_df = merged[cols].dropna(axis=1, thresh=int(0.7*len(merged))).fillna(merged[cols].median())
X_df = X_df.loc[:, X_df.std() > 1e-6]
y    = merged.loc[X_df.index, "ambiguous"].values

model = Pipeline([("scaler", StandardScaler()), ("clf", make_clf())])
model.fit(X_df.values, y)

scores = model.predict_proba(X_df.values)[:, 1]
case_ids = merged.loc[X_df.index, "case_id"].values
dice_vals = merged.loc[X_df.index, "dice"].values
hausd = merged.loc[X_df.index, "hausdorff_distance_voxels"].values

print("\nCandidates for Case B (Dice 0.61-0.75, Hausdorff<25):")
print(f"{'case':>6}  {'dice':>6}  {'score':>6}  {'hausdorff':>10}  {'r1_vol':>7}  {'r2_vol':>7}")
for i in np.argsort(np.abs(scores - 0.5)):
    cid   = case_ids[i]
    d     = dice_vals[i]
    s     = scores[i]
    h     = hausd[i]
    r1    = merged.loc[X_df.index[i], "reader1_volume_voxels"]
    r2    = merged.loc[X_df.index[i], "reader2_volume_voxels"]
    if 0.61 <= d <= 0.75 and h < 25:
        marker = " <-- GOOD" if 0.25 <= s <= 0.75 else ""
        print(f"{cid:>6}  {d:>6.3f}  {s:>6.3f}  {h:>10.1f}  {r1:>7}  {r2:>7}{marker}")
