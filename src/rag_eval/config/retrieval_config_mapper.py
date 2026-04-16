from __future__ import annotations

from typing import Any, Dict

from rag_eval.models.experiment_config import ExperimentConfig

class RetrievalConfigMapper:
    @staticmethod
    def build_update_payload(config: ExperimentConfig) -> Dict[str, Any]:
        settings = config.retrieval_settings or {}

        payload: Dict[str, Any] = {}

        supported_keys = [
            "TOP_K",
            "BYPASS_EMBEDDING_AND_RETRIEVAL",
            "RAG_FULL_CONTEXT",
            "ENABLE_RAG_HYBRID_SEARCH",
            "ENABLE_RAG_HYBRID_SEARCH_ENRICHED_TEXTS",
            "TOP_K_RERANKER",
            "RELEVANCE_THRESHOLD",
            "HYBRID_BM25_WEIGHT",
            "RAG_TEMPLATE",
            "RAG_RERANKING_MODEL",
            "RAG_RERANKING_ENGINE",
            "RAG_EXTERNAL_RERANKER_URL",
            "TEXT_SPLITTER",
            "ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER",
            "CHUNK_SIZE",
            "CHUNK_MIN_SIZE_TARGET",
            "CHUNK_OVERLAP",
            "PDF_LOADER_MODE",
        ]

        for key in supported_keys:
            if key in settings:
                payload[key] = settings[key]

        return payload