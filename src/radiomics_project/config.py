from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectConfig:
    dataset_root: Path
    split: str
    results_dir: Path
    figures_dir: Path


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    return ProjectConfig(
        dataset_root=Path(raw["dataset_root"]),
        split=raw.get("split", "train"),
        results_dir=Path(raw.get("results_dir", "results")),
        figures_dir=Path(raw.get("figures_dir", "results/figures")),
    )
