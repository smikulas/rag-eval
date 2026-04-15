from dataclasses import dataclass

@dataclass(slots=True)
class EvaluationSummary:
    experiment_name: str
    total_samples: int
    retrieval_hit_rate: float
    mean_first_relevant_rank: float | None
    exact_match_rate: float
    normalized_exact_match_rate: float
    mean_token_f1: float
    mean_faithfulness_overlap: float
    mean_latency_seconds: float