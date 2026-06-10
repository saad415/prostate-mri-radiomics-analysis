# Prostate MRI Radiomics Analysis

Reproducible Python pipeline for extracting radiomic biomarkers from prostate MRI lesion segmentations and studying inter-reader segmentation variability.

## Project Goals

- Extract shape, first-order, and texture radiomic features from T2-weighted prostate MRI.
- Save lesion-level radiomics features in a clean analysis table.
- Compare feature distributions across lesions.
- Quantify segmentation variability between reader annotations using overlap and feature-stability metrics.

## Dataset

This repository does not include MRI files. Keep the downloaded dataset locally and point the scripts to it.

Expected local structure:

```text
prostate158_train/
  prostate158_train/
    train.csv
    valid.csv
    train/
      020/
        t2.nii.gz
        t2_tumor_reader1.nii.gz
        adc.nii.gz
        adc_tumor_reader1.nii.gz
        adc_tumor_reader2.nii.gz
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The default pipeline works with the fallback NumPy/SciPy extractor. To use official PyRadiomics features, install the optional dependency:

```bash
pip install -r requirements-pyradiomics.txt
```

On Windows, PyRadiomics may need Microsoft C++ Build Tools if a prebuilt wheel is unavailable.

## Quick Start

Copy the config template:

```bash
copy configs\paths.example.json configs\paths.json
```

Edit `configs/paths.json` so `dataset_root` points to your local dataset folder.

Extract T2 radiomics features:

```bash
python src\extract_t2_radiomics.py --config configs\paths.json
```

If PyRadiomics is installed, the script uses it automatically. Otherwise it computes fallback shape, first-order, and GLCM texture features.

Analyze inter-reader mask variability:

```bash
python src\inter_reader_variability.py --config configs\paths.json
```

Generated tables and figures are written to `results/`.

## Portfolio Summary

Built a reproducible radiomics workflow for prostate MRI that extracts lesion-level biomarkers from T2-weighted images, compares feature distributions, and evaluates how reader segmentation differences affect feature stability.

## Notes

- Raw medical images are intentionally ignored by git.
- Commit code, documentation, plots, and small derived tables only if allowed by the dataset license.
- If a case does not include a reader-2 T2 tumor mask, variability can be demonstrated on ADC masks or on cases where two lesion masks are available.
