from dataclasses import dataclass, field
from typing import List

from rag_eval.models.sweep_summary_record import SweepSummaryRecord

@dataclass(slots=True)
class SweepReport:
    records: List[SweepSummaryRecord] = field(default_factory=list)
    best_retrieval_experiment: str | None = None
    best_generation_experiment: str | None = None
    fastest_experiment: str | None = None