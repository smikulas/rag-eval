from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt

from rag_eval.models.sweep_report import SweepReport

class PlotWriter:
    @staticmethod
    def write_all(report: SweepReport, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        PlotWriter._write_bar_chart(
            report=report,
            values=[record.retrieval_hit_rate for record in report.records],
            output_path=output_path / "retrieval_hit_rate.png",
            title="Retrieval Hit Rate by Experiment",
            ylabel="Hit Rate",
        )

        PlotWriter._write_bar_chart(
            report=report,
            values=[record.normalized_exact_match_rate for record in report.records],
            output_path=output_path / "normalized_exact_match_rate.png",
            title="Normalized Exact Match Rate by Experiment",
            ylabel="Normalized Exact Match Rate",
        )

        PlotWriter._write_bar_chart(
            report=report,
            values=[record.mean_token_f1 for record in report.records],
            output_path=output_path / "mean_token_f1.png",
            title="Mean Token F1 by Experiment",
            ylabel="Mean Token F1",
        )

        PlotWriter._write_bar_chart(
            report=report,
            values=[record.mean_faithfulness_overlap for record in report.records],
            output_path=output_path / "mean_faithfulness_overlap.png",
            title="Mean Faithfulness Overlap by Experiment",
            ylabel="Faithfulness Overlap",
        )

        PlotWriter._write_bar_chart(
            report=report,
            values=[record.mean_latency_seconds for record in report.records],
            output_path=output_path / "mean_latency_seconds.png",
            title="Mean Latency by Experiment",
            ylabel="Latency (s)",
        )

    @staticmethod
    def _write_bar_chart(
        report: SweepReport,
        values: List[float],
        output_path: Path,
        title: str,
        ylabel: str,
    ) -> None:
        experiment_names = [record.experiment_name for record in report.records]

        plt.figure(figsize=(10, 5))
        plt.bar(experiment_names, values)
        plt.title(title)
        plt.xlabel("Experiment")
        plt.ylabel(ylabel)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()