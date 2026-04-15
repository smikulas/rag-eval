# OpenWebUI RAG Evaluation Framework

Initial module for a Python-based evaluation framework targeting OpenWebUI Retrieval-Augmented Generation (RAG).

## Purpose

This module provides the first building block of the evaluation framework: a clean integration layer for calling OpenWebUI chat completions with attached knowledge collections and converting the raw API response into typed Python objects that can later be used for retrieval and generation metrics.

The goal of this first module is to make OpenWebUI interaction reproducible, structured, and easy to build on.

## Implemented Components

### `clients/openwebui_client.py`

Provides the runtime client for interacting with OpenWebUI.

Responsibilities:
- builds authenticated requests to the OpenWebUI chat completions endpoint
- sends user queries to the configured base model
- optionally attaches a knowledge collection via the `files` field
- supports optional chat history
- returns either:
  - the raw JSON response
  - or a parsed `ChatResponse` object

Design notes:
- request construction is encapsulated in helper methods
- endpoint path and request field names are declared as constants
- the client is kept focused on transport logic only
- response parsing is delegated to a dedicated parser module

### `models/chat_response.py`

Defines the structured response object returned by the parser.

Responsibilities:
- stores the final generated answer
- stores normalized retrieved chunks
- keeps the raw OpenWebUI response for debugging and traceability

This model acts as the main typed interface between the OpenWebUI client and the future evaluation pipeline.

### `models/chunk_record.py`

Defines the normalized representation of one retrieved chunk.

Responsibilities:
- stores chunk text
- stores source/document identifiers
- stores retrieval score
- stores metadata returned by OpenWebUI
- stores rank within the retrieved result set

This object provides a stable internal schema for retrieval evaluation, independent of the raw API response shape.

### `parsers/response_parser.py`

Converts raw OpenWebUI responses into typed models.

Responsibilities:
- extracts the final answer from the chat completion response
- parses retrieved sources from the OpenWebUI `sources` field
- aligns parallel lists such as:
  - `document`
  - `metadata`
  - `distances`
- creates normalized `ChunkRecord` instances
- returns a full `ChatResponse`

Design notes:
- parsing logic is separated from request logic
- the parser currently targets the observed OpenWebUI response structure
- the implementation is intentionally minimal and typed, to support later metric computation

## Current Architecture

```text
OpenWebUIClient
    -> sends request to OpenWebUI
    -> receives raw JSON response
    -> delegates parsing to ResponseParser
    -> returns ChatResponse

ResponseParser
    -> extracts answer
    -> flattens retrieved chunks
    -> builds ChunkRecord objects
    -> returns ChatResponse