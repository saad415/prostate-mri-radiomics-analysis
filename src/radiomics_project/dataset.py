from __future__ import annotations

from pathlib import Path

import pandas as pd


def case_dirs(dataset_root: Path, split: str = "train") -> list[Path]:
    split_dir = dataset_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Could not find split directory: {split_dir}")

    return sorted(path for path in split_dir.iterdir() if path.is_dir())


def load_split_table(dataset_root: Path, split: str = "train") -> pd.DataFrame | None:
    csv_paths = [dataset_root / f"{split}.csv"]
    if split == "train":
        csv_paths.append(dataset_root / "valid.csv")

    tables = [pd.read_csv(path) for path in csv_paths if path.exists()]
    if not tables:
        return None
    return pd.concat(tables, ignore_index=True).drop_duplicates(subset=["ID"], keep="first")


def case_id(case_dir: Path) -> str:
    return case_dir.name
