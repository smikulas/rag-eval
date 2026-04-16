from __future__ import annotations

import json
from pathlib import Path

from rag_eval.models.sweep_report import SweepReport
from rag_eval.utils.serialization import to_dict

class ReportWriter:
    @staticmethod
    def write_json(report: SweepReport, output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as file:
            json.dump(to_dict(report), file, ensure_ascii=False, indent=2)

    @staticmethod
    def write_markdown(report: SweepReport, output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "# Experiment Comparison Report",
            "",
            f"- Best retrieval experiment: `{report.best_retrieval_experiment}`",
            f"- Best generation experiment: `{report.best_generation_experiment}`",
            f"- Fastest experiment: `{report.fastest_experiment}`",
            "",
            "| Experiment | Samples | Hit@K | Mean Rank | Exact Match | Norm. Exact | Token F1 | Faithfulness | Mean Latency (s) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]

        for record in report.records:
            mean_rank = (
                f"{record.mean_first_relevant_rank:.3f}"
                if record.mean_first_relevant_rank is not None
                else "-"
            )

            lines.append(
                "| "
                f"{record.experiment_name} | "
                f"{record.total_samples} | "
                f"{record.retrieval_hit_rate:.3f} | "
                f"{mean_rank} | "
                f"{record.exact_match_rate:.3f} | "
                f"{record.normalized_exact_match_rate:.3f} | "
                f"{record.mean_token_f1:.3f} | "
                f"{record.mean_faithfulness_overlap:.3f} | "
                f"{record.mean_latency_seconds:.3f} |"
            )

        with path.open("w", encoding="utf-8") as file:
            file.write("\n".join(lines))