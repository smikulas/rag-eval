from dataclasses import dataclass
from typing import Any, Dict

from rag_eval.models.chat_response import ChatResponse
from rag_eval.models.evaluation_sample import EvaluationSample
from rag_eval.models.generation_metric_result import GenerationMetricResult
from rag_eval.models.retrieval_metric_result import RetrievalMetricResult

@dataclass(slots=True)
class EvaluationResult:
    sample: EvaluationSample
    response: ChatResponse
    retrieval_metrics: RetrievalMetricResult
    generation_metrics: GenerationMetricResult
    latency_seconds: float
    experiment_name: str
    metadata: Dict[str, Any]