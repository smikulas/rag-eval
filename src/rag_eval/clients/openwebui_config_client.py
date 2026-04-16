from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import requests

API_RETRIEVAL_CONFIG_PATH = "/api/v1/retrieval/config/update"
HTTP_HEADER_AUTHORIZATION = "Authorization"
HTTP_HEADER_CONTENT_TYPE = "Content-Type"
HTTP_HEADER_CONTENT_TYPE_JSON = "application/json"
AUTHORIZATION_SCHEME_BEARER = "Bearer"

@dataclass(slots=True)
class OpenWebUIConfigClient:
    base_url: str
    api_key: str
    timeout: int = 120

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")

    @property
    def retrieval_config_url(self) -> str:
        return f"{self.base_url}{API_RETRIEVAL_CONFIG_PATH}"

    def get_retrieval_config(self) -> Dict[str, Any]:
        response = requests.get(
            self.retrieval_config_url,
            headers=self._build_headers(),
            timeout=self.timeout,
        )

        if not response.ok:
            raise RuntimeError(
                f"Failed to read retrieval config: {response.status_code} {response.text}"
            )

        return response.json()

    def update_retrieval_config(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post(
            self.retrieval_config_url,
            headers=self._build_headers(),
            json=updates,
            timeout=self.timeout,
        )

        if not response.ok:
            raise RuntimeError(
                f"Failed to update retrieval config: {response.status_code} {response.text}"
            )

        return response.json()

    def _build_headers(self) -> Dict[str, str]:
        return {
            HTTP_HEADER_AUTHORIZATION: f"{AUTHORIZATION_SCHEME_BEARER} {self.api_key}",
            HTTP_HEADER_CONTENT_TYPE: HTTP_HEADER_CONTENT_TYPE_JSON,
        }