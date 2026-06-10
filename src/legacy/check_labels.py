import nibabel as nib
import numpy as np
from pathlib import Path

case = Path("E:/prostate158_train/prostate158_train/train/025")
anat = nib.load(str(case / "t2_anatomy_reader1.nii.gz"))
data = anat.get_fdata()
print("Anatomy unique values:", np.unique(data))
print("Shape:", data.shape)

# Check tumor mask
tumor = nib.load(str(case / "t2_tumor_reader1.nii.gz"))
tdata = tumor.get_fdata()
print("Tumor unique values:", np.unique(tdata))
print("Tumor voxel count:", (tdata > 0).sum())
