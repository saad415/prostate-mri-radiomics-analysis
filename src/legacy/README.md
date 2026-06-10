# Legacy Scripts

These scripts are retained for provenance but are **not part of the current research workflow**.

| Script | Reason deprecated |
|---|---|
| `train_classifier.py` | Used a circular synthetic label (high volume + high entropy = high-risk). Label derived from the same features used for classification — inflated AUC (0.941). |
| `export_sample_cases.py` | Used T2 radiomics features to predict ADC inter-reader ambiguity — modality mismatch. Replaced by `adc_pipeline_calibrated.py`. |
| `regen_case_b.py` | One-off script to regenerate Case B with case 090 during early exploration. Superseded by `adc_pipeline_calibrated.py`. |
| `pick_better_case_b.py` | Interactive case selection helper used during prototyping. Superseded by `adc_pipeline_calibrated.py`. |
| `pick_case_b_adc.py` | Candidate search for Case B after switching to ADC features. Superseded by `adc_pipeline_calibrated.py`. |
| `check_labels.py` | One-off label sanity check. |

## Current workflow scripts (in `src/`)

- `src/extract_t2_radiomics.py` — T2 feature extraction for anatomical characterization
- `src/inter_reader_variability.py` — ADC inter-reader Dice / Hausdorff / volume metrics
- `src/adc_pipeline.py` — ADC feature extraction and initial model training
- `src/adc_pipeline_calibrated.py` — Calibrated model training and sample case export (canonical)
- `src/train_ambiguity_classifier.py` — Full cross-validated ML evaluation with LASSO
- `src/feature_selection.py` — Standalone LASSO + mutual information analysis
- `src/plot_feature_distributions.py` — T2 feature distribution visualization
