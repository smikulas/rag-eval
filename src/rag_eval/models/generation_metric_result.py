from dataclasses import dataclass

@dataclass(slots=True)
class GenerationMetricResult:
    exact_match: float
    normalized_exact_match: float
    token_f1: float
    faithfulness_overlap: float