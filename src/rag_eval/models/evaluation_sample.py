from dataclasses import dataclass
from typing import List

@dataclass(slots=True)
class EvaluationSample:
    question_id: int
    question: str
    language: str
    ground_truth: str
    relevant_docs: List[str]
    relevant_doc_keys: List[str]