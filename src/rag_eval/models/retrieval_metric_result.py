from dataclasses import dataclass
from typing import Optional

@dataclass(slots=True)
class RetrievalMetricResult:
    hit_at_k: float
    first_relevant_rank: Optional[int]