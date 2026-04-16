from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from rag_eval.models.experiment_config import ExperimentConfig

class ConfigLoader:
    @staticmethod
    def load(path: str | Path) -> ExperimentConfig:
        config_path = Path(path)

        with config_path.open("r", encoding="utf-8") as file:
            raw_config = yaml.safe_load(file)

        if not isinstance(raw_config, dict):
            raise ValueError("Experiment config must be a YAML object.")

        return ConfigLoader._to_experiment_config(raw_config)

    @staticmethod
    def _to_experiment_config(raw_config: Dict[str, Any]) -> ExperimentConfig:
        ConfigLoader._validate_required_fields(raw_config)

        return ExperimentConfig(
            name=str(raw_config["name"]).strip(),
            model=str(raw_config["model"]).strip(),
            collection_id=str(raw_config["collection_id"]).strip(),
            temperature=raw_config.get("temperature"),
            max_tokens=raw_config.get("max_tokens"),
            retrieval_k=int(raw_config.get("retrieval_k", 3)),
            output_dir=str(raw_config.get("output_dir", "outputs")).strip(),
            output_tag=raw_config.get("output_tag"),
            history=raw_config.get("history", []),
            retrieval_settings=raw_config.get("retrieval_settings", {}),
        )

    @staticmethod
    def _validate_required_fields(raw_config: Dict[str, Any]) -> None:
        required_fields = ["name", "model", "collection_id"]
        missing_fields = [field for field in required_fields if field not in raw_config]

        if missing_fields:
            raise ValueError(
                f"Experiment config is missing required fields: {missing_fields}"
            )