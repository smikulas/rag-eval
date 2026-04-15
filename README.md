# OpenWebUI RAG Evaluation Framework

Initial modules for a Python-based evaluation framework targeting OpenWebUI Retrieval-Augmented Generation (RAG).

## Purpose

This project builds a structured and reproducible evaluation pipeline for OpenWebUI’s built-in RAG system.

The current implementation focuses on two foundational layers:

1. **OpenWebUI integration layer**
2. **Dataset modeling and loading layer**

These provide the basis for later evaluation metrics, experiment sweeps, and result analysis.

## Implemented Components

### 1. OpenWebUI Integration

#### `clients/openwebui_client.py`

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
- endpoint paths and request fields are defined as constants
- request construction is modularized
- parsing is delegated to a separate module

#### `models/chat_response.py`

Structured representation of a model response.

Responsibilities:
- stores the final generated answer
- stores normalized retrieved chunks
- keeps the raw OpenWebUI response for debugging and traceability

This model acts as the main typed interface between the OpenWebUI client and the future evaluation pipeline.

#### `models/chunk_record.py`

Defines the normalized representation of one retrieved chunk.

Responsibilities:
- stores chunk text
- stores source/document identifiers
- stores retrieval score
- stores metadata returned by OpenWebUI
- stores rank within the retrieved result set

This object provides a stable internal schema for retrieval evaluation, independent of the raw API response shape.

#### `parsers/response_parser.py`

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

### 2. Dataset Layer

#### `models/evaluation_sample.py`

Defines the structure of one evaluation sample.

Fields:
- `question_id`
- `question`
- `language`
- `ground_truth`
- `relevant_docs`
- `relevant_doc_keys`

Purpose:
- provides a typed representation of dataset entries
- prepares normalized document identifiers for retrieval evaluation

#### `datasets/dataset_loader.py`

Loads and validates evaluation datasets.

Responsibilities:
- supports `.json` and `.jsonl` formats
- validates required fields
- extracts relevant documents from:
  - `relevant_docs` (preferred)
  - or `relevant_doc_*` fields (legacy format)
- converts raw records into `EvaluationSample`
- generates normalized document keys for matching

#### `utils/document_keys.py`

Provides canonical document key generation.

Responsibilities:
- converts URLs into normalized document identifiers
- extracts base document keys from chunk filenames

Key idea:
- URLs and chunk filenames are mapped to the same canonical key
- matching is performed at **document level**, not chunk level

### Current Architecture

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

DatasetLoader
    -> loads dataset (JSON / JSONL)
    -> validates schema
    -> builds EvaluationSample
    -> generates document keys

EvaluationSample
    -> represents one evaluation case
    -> contains normalized document references