from __future__ import annotations

import argparse
import logging

import pandas as pd
from tqdm import tqdm

from radiomics_project.config import load_config
from radiomics_project.dataset import case_dirs, case_id
from radiomics_project.metrics import (
    dice_score,
    hausdorff_distance_voxels,
    load_mask,
    relative_volume_difference,
    volume_voxels,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure inter-reader segmentation variability.")
    parser.add_argument("--config", default="configs/paths.json", help="Path to project config JSON.")
    parser.add_argument("--modality", default="adc", help="Modality prefix, for example adc or t2.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of cases for a quick test run.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(args.config)
    config.results_dir.mkdir(parents=True, exist_ok=True)

    cases = case_dirs(config.dataset_root, config.split)
    if args.limit:
        cases = cases[: args.limit]

    rows = []
    for folder in tqdm(cases, desc="Comparing masks"):
        reader1 = folder / f"{args.modality}_tumor_reader1.nii.gz"
        reader2 = folder / f"{args.modality}_tumor_reader2.nii.gz"

        if not reader1.exists() or not reader2.exists():
            logging.warning("Skipping %s because reader masks are missing for %s.", folder.name, args.modality)
            continue

        mask1 = load_mask(reader1)
        mask2 = load_mask(reader2)
        rows.append(
            {
                "case_id": case_id(folder),
                "modality": args.modality,
                "reader1_volume_voxels": volume_voxels(mask1),
                "reader2_volume_voxels": volume_voxels(mask2),
                "dice": dice_score(mask1, mask2),
                "relative_volume_difference": relative_volume_difference(mask1, mask2),
                "hausdorff_distance_voxels": hausdorff_distance_voxels(mask1, mask2),
            }
        )

    output_path = config.results_dir / f"{args.modality}_inter_reader_variability.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    logging.info("Saved variability metrics for %s cases to %s", len(rows), output_path)


if __name__ == "__main__":
    main()
