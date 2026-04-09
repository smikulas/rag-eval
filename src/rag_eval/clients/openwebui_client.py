from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from rag_eval.models.chat_response import ChatResponse
from rag_eval.parsers.response_parser import ResponseParser

API_CHAT_COMPLETIONS_PATH = "/api/chat/completions"
HTTP_HEADER_AUTHORIZATION = "Authorization"
HTTP_HEADER_CONTENT_TYPE = "Content-Type"
HTTP_HEADER_CONTENT_TYPE_JSON = "application/json"
AUTHORIZATION_SCHEME_BEARER = "Bearer"

REQUEST_FIELD_MODEL = "model"
REQUEST_FIELD_MESSAGES = "messages"
REQUEST_FIELD_STREAM = "stream"
REQUEST_FIELD_TEMPERATURE = "temperature"
REQUEST_FIELD_MAX_TOKENS = "max_tokens"
REQUEST_FIELD_FILES = "files"

MESSAGE_ROLE_USER = "user"
FILE_TYPE_COLLECTION = "collection"

DEFAULT_TIMEOUT_SECONDS = 120

@dataclass(slots=True)
class OpenWebUIClient:
    base_url: str
    api_key: str
    model: str
    timeout: int = DEFAULT_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")

    @property
    def chat_completions_url(self) -> str:
        return f"{self.base_url}{API_CHAT_COMPLETIONS_PATH}"

    def ask(
        self,
        query: str,
        *,
        collection_id: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        payload = self._build_payload(
            query=query,
            collection_id=collection_id,
            temperature=temperature,
            max_tokens=max_tokens,
            history=history,
        )

        response = requests.post(
            self.chat_completions_url,
            headers=self._build_headers(),
            json=payload,
            timeout=self.timeout,
        )

        if not response.ok:
            raise RuntimeError(
                f"OpenWebUI request failed: {response.status_code} {response.text}"
            )

        return response.json()

    def ask_parsed(
        self,
        query: str,
        *,
        collection_id: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> ChatResponse:
        raw_response = self.ask(
            query=query,
            collection_id=collection_id,
            temperature=temperature,
            max_tokens=max_tokens,
            history=history,
        )
        return ResponseParser.parse_chat_response(raw_response)

    def _build_headers(self) -> Dict[str, str]:
        return {
            HTTP_HEADER_AUTHORIZATION: f"{AUTHORIZATION_SCHEME_BEARER} {self.api_key}",
            HTTP_HEADER_CONTENT_TYPE: HTTP_HEADER_CONTENT_TYPE_JSON,
        }

    def _build_payload(
        self,
        *,
        query: str,
        collection_id: Optional[str],
        temperature: Optional[float],
        max_tokens: Optional[int],
        history: Optional[List[Dict[str, str]]],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            REQUEST_FIELD_MODEL: self.model,
            REQUEST_FIELD_MESSAGES: self._build_messages(query=query, history=history),
            REQUEST_FIELD_STREAM: False,
        }

        if temperature is not None:
            payload[REQUEST_FIELD_TEMPERATURE] = temperature

        if max_tokens is not None:
            payload[REQUEST_FIELD_MAX_TOKENS] = max_tokens

        if collection_id:
            payload[REQUEST_FIELD_FILES] = [
                {
                    "type": FILE_TYPE_COLLECTION,
                    "id": collection_id,
                }
            ]

        return payload

    def _build_messages(
        self,
        *,
        query: str,
        history: Optional[List[Dict[str, str]]],
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []

        if history:
            messages.extend(history)

        messages.append(
            {
                "role": MESSAGE_ROLE_USER,
                "content": query,
            }
        )

        return messages