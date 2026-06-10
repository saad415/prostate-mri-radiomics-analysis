from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from radiomics_project.config import load_config
from radiomics_project.dataset import case_dirs, case_id, load_split_table
from radiomics_project.fallback_features import extract_fallback_features


def build_extractor():
    try:
        from radiomics import featureextractor
    except ImportError:
        return None

    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName("shape")
    extractor.enableFeatureClassByName("firstorder")
    extractor.enableFeatureClassByName("glcm")
    extractor.enableFeatureClassByName("glrlm")
    extractor.enableFeatureClassByName("glszm")
    extractor.enableFeatureClassByName("ngtdm")
    extractor.enableFeatureClassByName("gldm")
    return extractor


def extract_case_features(case_dir: Path, extractor) -> dict[str, object] | None:
    image_path = case_dir / "t2.nii.gz"
    mask_path = case_dir / "t2_tumor_reader1.nii.gz"

    if not image_path.exists() or not mask_path.exists():
        logging.warning("Skipping %s because T2 image or tumor mask is missing.", case_dir.name)
        return None

    if extractor is None:
        clean_features = extract_fallback_features(image_path, mask_path)
    else:
        features = extractor.execute(str(image_path), str(mask_path))
        clean_features = {
            key: value
            for key, value in features.items()
            if key.startswith("original_")
        }
    return {"case_id": case_id(case_dir), **clean_features}


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract T2 radiomics features from prostate MRI lesion masks.")
    parser.add_argument("--config", default="configs/paths.json", help="Path to project config JSON.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of cases for a quick test run.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(args.config)
    config.results_dir.mkdir(parents=True, exist_ok=True)

    extractor = build_extractor()
    if extractor is None:
        logging.warning("PyRadiomics is not installed. Using fallback NumPy/SciPy features.")

    cases = case_dirs(config.dataset_root, config.split)
    if args.limit:
        cases = cases[: args.limit]

    rows = []
    for folder in tqdm(cases, desc="Extracting radiomics"):
        row = extract_case_features(folder, extractor)
        if row:
            rows.append(row)

    features = pd.DataFrame(rows)
    split_table = load_split_table(config.dataset_root, config.split)
    if split_table is not None and "ID" in split_table.columns and not features.empty:
        split_table = split_table.copy()
        split_table["case_id"] = split_table["ID"].astype(int).astype(str).str.zfill(3)
        features = features.merge(split_table, on="case_id", how="left")

    output_path = config.results_dir / "t2_radiomics_features.csv"
    features.to_csv(output_path, index=False)
    logging.info("Saved %s cases and %s columns to %s", len(features), len(features.columns), output_path)


if __name__ == "__main__":
    main()
