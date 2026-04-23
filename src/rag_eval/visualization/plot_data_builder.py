from __future__ import annotations

from typing import Dict, List

from rag_eval.models.sweep_report import SweepReport

class PlotDataBuilder:
    @staticmethod
    def build_metric_series(report: SweepReport) -> Dict[str, List]:
        return {
            "experiment_names": [record.experiment_name for record in report.records],
            "retrieval_hit_rate": [record.retrieval_hit_rate for record in report.records],
            "mean_first_relevant_rank": [
                record.mean_first_relevant_rank for record in report.records
            ],
            "exact_match_rate": [record.exact_match_rate for record in report.records],
            "normalized_exact_match_rate": [
                record.normalized_exact_match_rate for record in report.records
            ],
            "mean_token_f1": [record.mean_token_f1 for record in report.records],
            "mean_faithfulness_overlap": [
                record.mean_faithfulness_overlap for record in report.records
            ],
            "mean_latency_seconds": [
                record.mean_latency_seconds for record in report.records
            ],
        }