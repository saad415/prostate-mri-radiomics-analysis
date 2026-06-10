# Prostate MRI Radiomics — Inter-Reader Ambiguity Prediction

A reproducible Python pipeline for radiomic feature extraction, inter-reader segmentation variability quantification, and machine-learning-based prediction of ADC tumour delineation ambiguity on the Prostate158 dataset.

## Research Question

Can ADC radiomic biomarkers — extracted from prostate tumour regions — predict which lesions will show high inter-reader segmentation disagreement between two expert radiologists?

Reader disagreement is used as an imaging-derived endpoint rather than a clinical outcome label, which is appropriate given the Prostate158 dataset does not include Gleason scores, PSA levels, or recurrence data.

```
ADC image + reader-1 tumour mask  →  ADC radiomic features (shape, first-order, GLCM)
reader-1 mask vs reader-2 mask    →  Dice / Hausdorff / volume variability
ADC radiomics  →  predict segmentation ambiguity (Dice < 0.608)
```

## Hypothesis

Tumours with heterogeneous microenvironmental phenotypes appear texturally distinct on ADC diffusion maps. This heterogeneity may cause reader disagreement at lesion boundaries. A sparse radiomic signature of ambiguity — derived entirely from ADC texture features — could serve as a non-invasive imaging proxy for such phenotypes.

## Dataset

**Prostate158** — 158 prostate MRI cases with T2-weighted and ADC volumes, and dual-reader tumour segmentations for the ADC modality.

This repository does not include raw MRI data. Download the dataset separately and configure the local path in `configs/paths.json`.

Expected local structure:

```
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

Raw medical images are excluded from version control via `.gitignore`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS
pip install -r requirements.txt
```

Copy and configure the dataset path:

```bash
copy configs\paths.example.json configs\paths.json
# Edit dataset_root in configs/paths.json to point to your local dataset
```

## Workflow

### Step 1 — T2 Lesion Characterization (anatomical reference)

Extract shape, first-order, and GLCM texture features from T2-weighted images using reader-1 tumour masks:

```bash
python src\extract_t2_radiomics.py --config configs\paths.json
```

Visualize feature distributions and inter-feature correlations:

```bash
python src\plot_feature_distributions.py \
    --features results\t2_radiomics_features.csv \
    --figures-dir results\figures
```

Output: `results/t2_radiomics_features.csv`

### Step 2 — ADC Inter-Reader Variability

Compute Dice score, relative volume difference, and Hausdorff distance between reader-1 and reader-2 ADC tumour masks for all cases with dual annotations:

```bash
python src\inter_reader_variability.py --config configs\paths.json --modality adc
```

Output: `results/adc_inter_reader_variability.csv` (67 cases with dual ADC masks)

### Step 3 — ADC Feature Extraction

Extract ADC radiomic features from `adc.nii.gz` using `adc_tumor_reader1.nii.gz`. Features are cached to CSV on first run:

```bash
python src\adc_pipeline.py
```

Output: `results/adc_radiomics_features.csv` (83 cases with ADC masks)

### Step 4 — Ambiguity Classification

Train and cross-validate three classifiers (Logistic Regression, Random Forest, XGBoost) on ADC features with the inter-reader Dice ambiguity label:

```bash
python src\train_ambiguity_classifier.py
```

The ambiguity label is defined as:

```
Dice(reader1, reader2) < 0.608  →  ambiguous  (label = 1)
Dice(reader1, reader2) ≥ 0.608  →  clear      (label = 0)
```

The threshold 0.608 is the median inter-reader Dice across the 67 dual-annotated cases, giving a balanced split (33 ambiguous, 34 clear).

Current cross-validated results (5-fold StratifiedKFold):

```
Model               AUC    Accuracy   F1
RandomForest       0.729   0.716      0.716
LogisticRegression 0.719   0.672      0.667
XGBoost            0.635   0.642      0.613
```

Output: `results/ambiguity_classification_report.csv`

### Step 5 — Feature Selection

Identify the most informative ADC radiomic features using two independent methods:

```bash
python src\feature_selection.py
```

- **LASSO** (L1 LogisticRegressionCV, C ∈ [0.001, 0.316]): selects **4 sparse features** from 168 candidates
- **Mutual information**: ranks all 168 features by non-linear dependence with the ambiguity label

The top feature by both methods is `original_firstorder_histogram_bin_26`, suggesting that the upper tail of the ADC intensity histogram is the primary discriminative signal.

Output: `results/selected_features.csv`, `results/figures/lasso_feature_importance.png`, `results/figures/mutual_information_top20.png`

### Step 6 — Sample Case Export

Export three representative cases with calibrated inference scores for visualization:

```bash
python src\adc_pipeline_calibrated.py
```

Uses XGBoost with isotonic probability calibration (`CalibratedClassifierCV`, 5-fold) to produce reliable ambiguity scores on this 67-case cohort. Selects:
- **Case A** — highest ambiguity score (Dice = 0.515, score = 1.000)
- **Case B** — score nearest to decision boundary among clear-class cases (Dice = 0.705, score = 0.362)
- **Case C** — lowest ambiguity score (Dice = 0.807, score = 0.120)

Output: `results/sample_cases/prostate-case-{a,b,c}-adc-slice.png` and `prostate-case-{a,b,c}-inference.json`

## Outputs

| File | Description |
|---|---|
| `results/t2_radiomics_features.csv` | T2 radiomic features — anatomical characterization |
| `results/adc_radiomics_features.csv` | ADC radiomic features — ambiguity prediction input |
| `results/adc_inter_reader_variability.csv` | Per-case Dice, Hausdorff, volume difference |
| `results/ambiguity_classification_report.csv` | Cross-validated AUC, accuracy, F1 per model |
| `results/selected_features.csv` | LASSO coefficients + MI scores for all 168 features |
| `results/figures/ambiguity_roc_curves.png` | ROC overlay for all three classifiers |
| `results/figures/ambiguity_lasso_features.png` | LASSO-selected feature coefficients |
| `results/figures/ambiguity_dice_distribution.png` | Inter-reader Dice distribution with threshold |
| `results/figures/mutual_information_top20.png` | Top 20 features by mutual information |
| `results/sample_cases/` | ADC slice PNGs and inference JSONs for 3 cases |

## Limitations

- **67 cases** with dual ADC annotations — sufficient for exploratory analysis, underpowered for clinical validation.
- **No clinical outcome labels** in Prostate158 (no Gleason, PSA, recurrence). Ambiguity prediction is an imaging-derived surrogate endpoint.
- **Modest AUC (0.635–0.729)** reflects the genuine difficulty of the task and honest cross-validation, not a calibration artefact.
- T2 features are extracted for anatomical characterization only and are not used in the ambiguity prediction model.

## Scientific Framing

This is an exploratory radiomics study. The predictive endpoint — inter-reader segmentation ambiguity — is derived entirely from independent expert annotations and is not circular with respect to the radiomic features used for classification. Results should be interpreted as hypothesis-generating rather than clinically actionable.

## Repository Structure

```
src/
  extract_t2_radiomics.py         T2 feature extraction
  inter_reader_variability.py     ADC inter-reader metrics
  adc_pipeline.py                 ADC feature extraction + initial training
  adc_pipeline_calibrated.py      Calibrated model + sample case export
  train_ambiguity_classifier.py   Full ML evaluation with LASSO
  feature_selection.py            Standalone LASSO + mutual information
  plot_feature_distributions.py   T2 feature visualization
  legacy/                         Deprecated scripts (retained for provenance)
  radiomics_project/              Shared utilities (feature extraction, metrics)
configs/
  paths.example.json              Config template
results/                          Generated outputs (CSV, figures, cases)
```

## Notes

- `configs/paths.json` is local-only and excluded from version control.
- Raw MRI volumes are excluded from version control.
- `src/legacy/` contains deprecated scripts kept for provenance. See `src/legacy/README.md`.
