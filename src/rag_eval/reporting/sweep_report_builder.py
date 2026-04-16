from __future__ import annotations

from typing import List

from rag_eval.models.evaluation_summary import EvaluationSummary
from rag_eval.models.sweep_report import SweepReport
from rag_eval.models.sweep_summary_record import SweepSummaryRecord

class SweepReportBuilder:
    @staticmethod
    def build(summaries: List[EvaluationSummary]) -> SweepReport:
        records = [
            SweepSummaryRecord(
                experiment_name=summary.experiment_name,
                total_samples=summary.total_samples,
                retrieval_hit_rate=summary.retrieval_hit_rate,
                mean_first_relevant_rank=summary.mean_first_relevant_rank,
                exact_match_rate=summary.exact_match_rate,
                normalized_exact_match_rate=summary.normalized_exact_match_rate,
                mean_token_f1=summary.mean_token_f1,
                mean_faithfulness_overlap=summary.mean_faithfulness_overlap,
                mean_latency_seconds=summary.mean_latency_seconds,
            )
            for summary in summaries
        ]

        return SweepReport(
            records=records,
            best_retrieval_experiment=SweepReportBuilder._best_retrieval_experiment(records),
            best_generation_experiment=SweepReportBuilder._best_generation_experiment(records),
            fastest_experiment=SweepReportBuilder._fastest_experiment(records),
        )

    @staticmethod
    def _best_retrieval_experiment(records: List[SweepSummaryRecord]) -> str | None:
        if not records:
            return None

        best = max(
            records,
            key=lambda record: (
                record.retrieval_hit_rate,
                -(record.mean_first_relevant_rank or float("inf")),
            ),
        )
        return best.experiment_name

    @staticmethod
    def _best_generation_experiment(records: List[SweepSummaryRecord]) -> str | None:
        if not records:
            return None

        best = max(
            records,
            key=lambda record: (
                record.normalized_exact_match_rate,
                record.mean_token_f1,
                record.mean_faithfulness_overlap,
            ),
        )
        return best.experiment_name

    @staticmethod
    def _fastest_experiment(records: List[SweepSummaryRecord]) -> str | None:
        if not records:
            return None

        best = min(records, key=lambda record: record.mean_latency_seconds)
        return best.experiment_name