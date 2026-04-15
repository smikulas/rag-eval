from __future__ import annotations

from typing import List

from rag_eval.models.chunk_record import ChunkRecord
from rag_eval.models.evaluation_sample import EvaluationSample
from rag_eval.models.retrieval_metric_result import RetrievalMetricResult
from rag_eval.utils.document_keys import build_document_key_from_chunk_source

class RetrievalMetrics:
    @staticmethod
    def evaluate(
        sample: EvaluationSample,
        chunks: List[ChunkRecord],
        k: int,
    ) -> RetrievalMetricResult:
        return RetrievalMetricResult(
            hit_at_k=RetrievalMetrics.hit_at_k(sample, chunks, k),
            first_relevant_rank=RetrievalMetrics.first_relevant_rank(sample, chunks),
        )

    @staticmethod
    def hit_at_k(
        sample: EvaluationSample,
        chunks: List[ChunkRecord],
        k: int,
    ) -> float:
        if k <= 0:
            raise ValueError("k must be greater than 0.")

        expected_keys = {
            key.strip().lower()
            for key in sample.relevant_doc_keys
            if key.strip()
        }

        if not expected_keys:
            return 0.0

        for chunk in chunks[:k]:
            chunk_key = build_document_key_from_chunk_source(chunk.source)
            if chunk_key in expected_keys:
                return 1.0

        return 0.0

    @staticmethod
    def first_relevant_rank(
        sample: EvaluationSample,
        chunks: List[ChunkRecord],
    ) -> int | None:
        expected_keys = {
            key.strip().lower()
            for key in sample.relevant_doc_keys
            if key.strip()
        }

        if not expected_keys:
            return None

        for chunk in chunks:
            chunk_key = build_document_key_from_chunk_source(chunk.source)
            if chunk_key in expected_keys:
                return chunk.rank

        return None