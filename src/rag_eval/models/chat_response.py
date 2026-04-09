from dataclasses import dataclass
from typing import Any, Dict, List

from rag_eval.models.chunk_record import ChunkRecord

@dataclass(slots=True)
class ChatResponse:
    answer: str
    chunks: List[ChunkRecord]
    raw: Dict[str, Any]