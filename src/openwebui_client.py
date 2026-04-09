import requests
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass
class OpenWebUIClient:
    base_url: str
    api_key: str
    model: str
    timeout: int = 120

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def ask(self, query: str, temperature: float = 0.0, max_tokens: int = 600) -> Dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/api/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": query}],
            "stream": False,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = requests.post(url, headers=self._headers(), json=payload, timeout=self.timeout)
        #resp.raise_for_status()
        if not resp.ok:
            raise RuntimeError(
                f"OpenWebUI request failed: {resp.status_code} {resp.text}"
            )
        return resp.json()

    def parse_answer(self, response: Dict[str, Any]) -> str:
        return response["choices"][0]["message"]["content"]
    
    def parse_chunks(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        candidates = response.get("citations") or response.get("sources") or []
        normalized: List[Dict[str, Any]] = []

        print(type(candidates), candidates[:1] if isinstance(candidates, list) else candidates)

        for i, item in enumerate(candidates, start=1):
            if not isinstance(item, dict):
                continue

            print("ITEM TYPE:", type(item), "METADATA TYPE:", type(item.get("metadata", None)) if isinstance(item, dict) else None)

            metadata = item.get("metadata", {})
            if isinstance(metadata, list):
                metadata = metadata[0] if metadata and isinstance(metadata[0], dict) else {}
            elif not isinstance(metadata, dict):
                metadata = {}

            doc_id = (
                item.get("doc_id")
                or item.get("document_id")
                or item.get("source_id")
                or metadata.get("doc_id")
                or metadata.get("document_id")
                or ""
            )

            chunk_id = (
                item.get("chunk_id")
                or metadata.get("chunk_id")
                or ""
            )

            """ text = (
                item.get("text")
                or item.get("content")
                or item.get("document")
                or metadata.get("text")
                or ""
            ) """

            raw_text = (
                item.get("text")
                or item.get("content")
                or item.get("document")
                or metadata.get("text")
                or ""
            )

            if isinstance(raw_text, list):
                text = "\n".join(str(x) for x in raw_text if x is not None)
            else:
                text = str(raw_text) if raw_text is not None else ""

            source = (
                item.get("source")
                or item.get("file_name")
                or metadata.get("source")
                or ""
            )

            normalized.append(
                {
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "text": text,
                    "score": item.get("score"),
                    "reranker_score": item.get("reranker_score"),
                    "source": source,
                    "rank": i,
                }
            )

        return normalized