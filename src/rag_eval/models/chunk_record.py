from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass(slots=True)
class ChunkRecord:
    doc_id: str
    text: str
    score: Optional[float]
    source: str
    start_index: Optional[int]
    metadata: Dict[str, Any]
    rank: int