from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from rag_eval.models.evaluation_sample import EvaluationSample
from rag_eval.utils.document_keys import build_document_key_from_url

SUPPORTED_JSON_SUFFIXES = {".json", ".jsonl"}

FIELD_QUESTION_ID = "question_id"
FIELD_QUESTION = "question"
FIELD_LANGUAGE = "language"
FIELD_GROUND_TRUTH = "ground_truth"
FIELD_RELEVANT_DOCS = "relevant_docs"
RELEVANT_DOC_PREFIX = "relevant_doc_"

class DatasetLoader:
    @staticmethod
    def load(path: str | Path) -> List[EvaluationSample]:
        dataset_path = Path(path)

        if dataset_path.suffix not in SUPPORTED_JSON_SUFFIXES:
            raise ValueError(
                f"Unsupported dataset format: {dataset_path.suffix}. "
                f"Expected one of {sorted(SUPPORTED_JSON_SUFFIXES)}."
            )

        if dataset_path.suffix == ".json":
            records = DatasetLoader._load_json(dataset_path)
        else:
            records = DatasetLoader._load_jsonl(dataset_path)

        return [DatasetLoader._to_sample(record) for record in records]

    @staticmethod
    def _load_json(path: Path) -> List[Dict[str, Any]]:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            raise ValueError("JSON dataset must contain a top-level list of records.")

        return data

    @staticmethod
    def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []

        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                stripped = line.strip()
                if not stripped:
                    continue

                record = json.loads(stripped)
                if not isinstance(record, dict):
                    raise ValueError(
                        f"JSONL record at line {line_number} must be an object."
                    )

                records.append(record)

        return records

    @staticmethod
    def _to_sample(record: Dict[str, Any]) -> EvaluationSample:
        DatasetLoader._validate_required_fields(record)

        relevant_docs = DatasetLoader._extract_relevant_docs(record)
        relevant_doc_keys = [
            build_document_key_from_url(doc)
            for doc in relevant_docs
        ]

        return EvaluationSample(
            question_id=int(record[FIELD_QUESTION_ID]),
            question=str(record[FIELD_QUESTION]).strip(),
            language=str(record[FIELD_LANGUAGE]).strip(),
            ground_truth=str(record[FIELD_GROUND_TRUTH]).strip(),
            relevant_docs=relevant_docs,
            relevant_doc_keys=relevant_doc_keys
        )

    @staticmethod
    def _validate_required_fields(record: Dict[str, Any]) -> None:
        required_fields = [
            FIELD_QUESTION_ID,
            FIELD_QUESTION,
            FIELD_LANGUAGE,
            FIELD_GROUND_TRUTH,
        ]

        missing_fields = [field for field in required_fields if field not in record]
        if missing_fields:
            raise ValueError(f"Dataset record is missing required fields: {missing_fields}")

    @staticmethod
    def _extract_relevant_docs(record: Dict[str, Any]) -> List[str]:
        if FIELD_RELEVANT_DOCS in record:
            relevant_docs = record[FIELD_RELEVANT_DOCS]
            if not isinstance(relevant_docs, list):
                raise ValueError(f"Field '{FIELD_RELEVANT_DOCS}' must be a list.")
            return [str(doc).strip() for doc in relevant_docs if str(doc).strip()]

        extracted_docs: List[str] = []

        for key, value in record.items():
            if key.startswith(RELEVANT_DOC_PREFIX) and value:
                normalized_value = str(value).strip()
                if normalized_value:
                    extracted_docs.append(normalized_value)

        return extracted_docs