from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from rag_eval.models.evaluation_result import EvaluationResult
from rag_eval.models.evaluation_summary import EvaluationSummary
from rag_eval.utils.serialization import to_dict

class ResultWriter:
    @staticmethod
    def write_results_json(
        results: Iterable[EvaluationResult],
        output_path: str | Path,
    ) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = [to_dict(r) for r in results]

        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def write_results_jsonl(
        results: Iterable[EvaluationResult],
        output_path: str | Path,
    ) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            for r in results:
                json.dump(to_dict(r), f, ensure_ascii=False)
                f.write("\n")

    @staticmethod
    def write_summary(
        summary: EvaluationSummary,
        output_path: str | Path,
    ) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            json.dump(to_dict(summary), f, ensure_ascii=False, indent=2)