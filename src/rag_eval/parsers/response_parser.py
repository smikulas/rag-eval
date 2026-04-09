from typing import Any, Dict, List

from rag_eval.models.chunk_record import ChunkRecord
from rag_eval.models.chat_response import ChatResponse

class ResponseParser:
    @staticmethod
    def parse_chat_response(response: Dict[str, Any]) -> ChatResponse:
        answer = ResponseParser._parse_answer(response)
        chunks = ResponseParser._parse_chunks(response)

        return ChatResponse(
            answer=answer,
            chunks=chunks,
            raw=response,
        )

    @staticmethod
    def _parse_answer(response: Dict[str, Any]) -> str:
        return (
            response.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

    @staticmethod
    def _parse_chunks(response: Dict[str, Any]) -> List[ChunkRecord]:
        sources = response.get("sources", [])
        results: List[ChunkRecord] = []

        rank = 1

        for source_item in sources:
            documents = source_item.get("document", [])
            metadata_list = source_item.get("metadata", [])
            distances = source_item.get("distances", [])

            for i, text in enumerate(documents):
                metadata = (
                    metadata_list[i]
                    if i < len(metadata_list) and isinstance(metadata_list[i], dict)
                    else {}
                )

                score = distances[i] if i < len(distances) else None

                results.append(
                    ChunkRecord(
                        doc_id=metadata.get("file_id", ""),
                        text=ResponseParser._normalize_text(text),
                        score=score,
                        source=metadata.get("source", ""),
                        start_index=metadata.get("start_index"),
                        metadata=metadata,
                        rank=rank,
                    )
                )

                rank += 1

        return results

    @staticmethod
    def _normalize_text(raw_text: Any) -> str:
        if isinstance(raw_text, list):
            return "\n".join(str(x) for x in raw_text if x is not None).strip()

        if raw_text is None:
            return ""

        return str(raw_text).strip()