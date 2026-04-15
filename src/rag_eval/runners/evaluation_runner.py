from __future__ import annotations

import time
from dataclasses import dataclass
from statistics import mean
from typing import List, Optional

from rag_eval.clients.openwebui_client import OpenWebUIClient
from rag_eval.models.evaluation_result import EvaluationResult
from rag_eval.models.evaluation_sample import EvaluationSample
from rag_eval.models.evaluation_summary import EvaluationSummary
from rag_eval.metrics.generation_metrics import GenerationMetrics
from rag_eval.metrics.retrieval_metrics import RetrievalMetrics

DEFAULT_RETRIEVAL_K = 3

@dataclass(slots=True)
class EvaluationRunner:
    client: OpenWebUIClient
    retrieval_k: int = DEFAULT_RETRIEVAL_K

    def run_sample(
        self,
        sample: EvaluationSample,
        *,
        experiment_name: str,
        collection_id: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        history: Optional[List[dict[str, str]]] = None,
    ) -> EvaluationResult:
        start_time = time.perf_counter()

        response = self.client.ask_parsed(
            query=sample.question,
            collection_id=collection_id,
            temperature=temperature,
            max_tokens=max_tokens,
            history=history,
        )

        latency_seconds = time.perf_counter() - start_time

        retrieval_metrics = RetrievalMetrics.evaluate(
            sample=sample,
            chunks=response.chunks,
            k=self.retrieval_k,
        )

        generation_metrics = GenerationMetrics.evaluate(
            answer=response.answer,
            ground_truth=sample.ground_truth,
            chunks=response.chunks,
        )

        return EvaluationResult(
            sample=sample,
            response=response,
            retrieval_metrics=retrieval_metrics,
            generation_metrics=generation_metrics,
            latency_seconds=latency_seconds,
            experiment_name=experiment_name,
            metadata={
                "collection_id": collection_id,
                "retrieval_k": self.retrieval_k,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )

    def run_dataset(
        self,
        samples: List[EvaluationSample],
        *,
        experiment_name: str,
        collection_id: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        history: Optional[List[dict[str, str]]] = None,
    ) -> List[EvaluationResult]:
        results: List[EvaluationResult] = []

        for sample in samples:
            result = self.run_sample(
                sample=sample,
                experiment_name=experiment_name,
                collection_id=collection_id,
                temperature=temperature,
                max_tokens=max_tokens,
                history=history,
            )
            results.append(result)

        return results

    @staticmethod
    def summarize(experiment_name: str, results: List[EvaluationResult]) -> EvaluationSummary:
        if not results:
            return EvaluationSummary(
                experiment_name=experiment_name,
                total_samples=0,
                retrieval_hit_rate=0.0,
                mean_first_relevant_rank=None,
                exact_match_rate=0.0,
                normalized_exact_match_rate=0.0,
                mean_token_f1=0.0,
                mean_faithfulness_overlap=0.0,
                mean_latency_seconds=0.0,
            )

        first_relevant_ranks = [
            result.retrieval_metrics.first_relevant_rank
            for result in results
            if result.retrieval_metrics.first_relevant_rank is not None
        ]

        return EvaluationSummary(
            experiment_name=experiment_name,
            total_samples=len(results),
            retrieval_hit_rate=mean(
                result.retrieval_metrics.hit_at_k for result in results
            ),
            mean_first_relevant_rank=(
                mean(first_relevant_ranks) if first_relevant_ranks else None
            ),
            exact_match_rate=mean(
                result.generation_metrics.exact_match for result in results
            ),
            normalized_exact_match_rate=mean(
                result.generation_metrics.normalized_exact_match
                for result in results
            ),
            mean_token_f1=mean(
                result.generation_metrics.token_f1 for result in results
            ),
            mean_faithfulness_overlap=mean(
                result.generation_metrics.faithfulness_overlap
                for result in results
            ),
            mean_latency_seconds=mean(
                result.latency_seconds for result in results
            ),
        )