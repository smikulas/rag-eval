from __future__ import annotations

import re
from urllib.parse import urlparse

DEFAULT_PATH_LEVELS = 3
CHUNK_SUFFIX_PATTERN = re.compile(r"_\d{3}$")

def build_document_key_from_url(url: str, path_levels: int = DEFAULT_PATH_LEVELS) -> str:
    parsed = urlparse(url)
    domain = parsed.netloc.strip().lower()

    path_parts = [
        part.strip().lower()
        for part in parsed.path.split("/")
        if part.strip()
    ]

    selected_parts = path_parts[:path_levels]
    return "_".join([domain, *selected_parts])

def build_document_key_from_chunk_source(source: str) -> str:
    normalized = source.strip().lower()

    if normalized.endswith(".md"):
        normalized = normalized[:-3]

    normalized = CHUNK_SUFFIX_PATTERN.sub("", normalized)
    return normalized