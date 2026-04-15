from __future__ import annotations

import re
from urllib.parse import urlparse

DEFAULT_PATH_LEVELS = 3

# removes file extensions like .html, .pdf
FILE_EXTENSION_PATTERN = re.compile(r"\.(html|pdf)$")

# remove final .md
MD_EXTENSION_PATTERN = re.compile(r"\.md$")

# removes chunk suffixes like:
# _001
# _001.chunk0002
# _12.chunk45
CHUNK_SUFFIX_PATTERN = re.compile(r"(_\d+)(\.chunk\d+)?$")

def build_document_key_from_url(url: str, path_levels: int = DEFAULT_PATH_LEVELS) -> str:
    parsed = urlparse(url)
    domain = parsed.netloc.strip().lower()

    path_parts = [
        part.strip().lower()
        for part in parsed.path.split("/")
        if part.strip()
    ]

    selected_parts = path_parts[:path_levels]

    normalized_parts = [
        FILE_EXTENSION_PATTERN.sub("", part)
        for part in selected_parts
    ]

    return "_".join([domain, *normalized_parts])

def build_document_key_from_chunk_source(source: str) -> str:
    normalized = source.strip().lower()

    # remove final .md first
    normalized = MD_EXTENSION_PATTERN.sub("", normalized)

    # remove chunk suffix (_001, _001.chunk0002, etc.)
    normalized = CHUNK_SUFFIX_PATTERN.sub("", normalized)

    # remove .html / .pdf even if in middle
    normalized = FILE_EXTENSION_PATTERN.sub("", normalized)

    return normalized