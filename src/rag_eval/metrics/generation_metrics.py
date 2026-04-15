from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Iterable, List

from rag_eval.models.chunk_record import ChunkRecord
from rag_eval.models.generation_metric_result import GenerationMetricResult

WHITESPACE_PATTERN = re.compile(r"\s+")
PUNCTUATION_PATTERN = re.compile(r"[^\w\s]", flags=re.UNICODE)

class GenerationMetrics:
    @staticmethod
    def evaluate(
        answer: str,
        ground_truth: str,
        chunks: List[ChunkRecord],
    ) -> GenerationMetricResult:
        return GenerationMetricResult(
            exact_match=GenerationMetrics.exact_match(answer, ground_truth),
            normalized_exact_match=GenerationMetrics.normalized_exact_match(
                answer,
                ground_truth,
            ),
            token_f1=GenerationMetrics.token_f1(answer, ground_truth),
            faithfulness_overlap=GenerationMetrics.faithfulness_overlap(answer, chunks),
        )

    @staticmethod
    def exact_match(answer: str, ground_truth: str) -> float:
        return float(answer == ground_truth)

    @staticmethod
    def normalized_exact_match(answer: str, ground_truth: str) -> float:
        normalized_answer = GenerationMetrics._normalize_text(answer)
        normalized_ground_truth = GenerationMetrics._normalize_text(ground_truth)
        return float(normalized_answer == normalized_ground_truth)

    @staticmethod
    def token_f1(answer: str, ground_truth: str) -> float:
        answer_tokens = GenerationMetrics._tokenize(answer)
        ground_truth_tokens = GenerationMetrics._tokenize(ground_truth)

        if not answer_tokens and not ground_truth_tokens:
            return 1.0

        if not answer_tokens or not ground_truth_tokens:
            return 0.0

        answer_counts = Counter(answer_tokens)
        ground_truth_counts = Counter(ground_truth_tokens)

        common = answer_counts & ground_truth_counts
        num_same = sum(common.values())

        if num_same == 0:
            return 0.0

        precision = num_same / len(answer_tokens)
        recall = num_same / len(ground_truth_tokens)

        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def faithfulness_overlap(answer: str, chunks: List[ChunkRecord]) -> float:
        answer_tokens = set(GenerationMetrics._tokenize(answer))
        if not answer_tokens:
            return 0.0

        context_tokens = set(
            GenerationMetrics._tokenize(
                " ".join(chunk.text for chunk in chunks if chunk.text)
            )
        )

        if not context_tokens:
            return 0.0

        supported_tokens = answer_tokens.intersection(context_tokens)
        return len(supported_tokens) / len(answer_tokens)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        normalized = GenerationMetrics._normalize_text(text)
        if not normalized:
            return []
        return normalized.split(" ")

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = unicodedata.normalize("NFKC", text).lower()
        text = PUNCTUATION_PATTERN.sub(" ", text)
        text = WHITESPACE_PATTERN.sub(" ", text).strip()
        return text