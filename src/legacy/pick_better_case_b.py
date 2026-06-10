import pandas as pd
import nibabel as nib
import numpy as np
from pathlib import Path

var = pd.read_csv("results/adc_inter_reader_variability.csv")
var["case_id"] = var["case_id"].astype(str).str.zfill(3)
feats = pd.read_csv("results/t2_radiomics_features.csv")
feats["case_id"] = feats["case_id"].astype(str).str.zfill(3)

# Cases in merged set, borderline Dice (0.55-0.68), low Hausdorff (<40)
merged = var[var["case_id"].isin(feats["case_id"])]
borderline = merged[
    (merged["dice"] >= 0.55) & (merged["dice"] <= 0.68) &
    (merged["hausdorff_distance_voxels"] < 45)
].sort_values("dice")

ROOT = Path("E:/prostate158_train/prostate158_train/train")

for _, row in borderline.iterrows():
    cid = row["case_id"]
    mask_path = ROOT / cid / "adc_tumor_reader1.nii.gz"
    if not mask_path.exists():
        continue
    mask = nib.load(str(mask_path)).get_fdata()
    mid = mask.shape[2] // 2
    # Check mask visible within 2 slices of middle
    visible = any(mask[:,:,max(0,mid-2):mid+3].sum() > 0 for _ in [1])
    has_mask_near_mid = mask[:,:,max(0,mid-2):mid+3].sum() > 10
    print(f"case={cid}  dice={row['dice']:.3f}  hausdorff={row['hausdorff_distance_voxels']:.1f}  mask_near_mid={has_mask_near_mid}  r1={int(row['reader1_volume_voxels'])}  r2={int(row['reader2_volume_voxels'])}")
