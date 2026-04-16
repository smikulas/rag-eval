from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass(slots=True)
class ExperimentConfig:
    name: str
    model: str
    collection_id: str

    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    retrieval_k: int = 3

    output_dir: str = "outputs"
    output_tag: Optional[str] = None

    history: list[dict[str, str]] = field(default_factory=list)

    retrieval_settings: Dict[str, Any] = field(default_factory=dict)