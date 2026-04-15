# OpenWebUI RAG Evaluation Framework

Initial modules for a Python-based evaluation framework targeting OpenWebUI Retrieval-Augmented Generation (RAG).

## Purpose

This project builds a structured and reproducible evaluation pipeline for OpenWebUI’s built-in RAG system.

The current implementation focuses on three foundational layers:

1. **OpenWebUI integration layer**
2. **Dataset modeling and loading layer**
3. **Metrics layer for retrieval and generation**

These provide the basis for later experiment sweeps, and result analysis.

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

## 3. Metrics Layer

This module introduces evaluation metrics for both retrieval and generation quality.

The design separates these concerns to ensure clear analysis of:
- whether the correct information was retrieved
- whether the model produced a correct and grounded answer

### Retrieval Metrics

#### `models/retrieval_metric_result.py`

Defines the structured output for retrieval evaluation.

Fields:
- `hit_at_k`: float (1.0 if relevant document is found in top-k, else 0.0)
- `first_relevant_rank`: rank of the first relevant retrieved chunk (or `None`)

#### `metrics/retrieval_metrics.py`

Provides retrieval evaluation logic.

Responsibilities:
- compares retrieved chunks against expected documents
- uses normalized document keys for matching
- computes:
  - **hit@k**: whether any relevant document appears in top-k
  - **first relevant rank**: position of the first correct chunk

Design notes:
- matching is done at **document level**, not chunk level
- relies on `document_keys.py` to ensure consistent mapping
- returns a structured `RetrievalMetricResult`

### Generation Metrics

#### `models/generation_metric_result.py`

Defines the structured output for generation evaluation.

Fields:
- `exact_match`: strict string match
- `normalized_exact_match`: match after normalization
- `token_f1`: token-level F1 score
- `faithfulness_overlap`: fraction of answer tokens supported by retrieved context

#### `metrics/generation_metrics.py`

Provides generation quality evaluation.

Responsibilities:
- compares model answer against ground truth
- evaluates lexical similarity
- measures grounding against retrieved chunks

Metrics:
- **exact match**: strict equality
- **normalized exact match**: case/format insensitive comparison
- **token F1**: overlap between answer and ground truth tokens
- **faithfulness overlap**:
  - measures how much of the answer is supported by retrieved context
  - approximates hallucination detection

Design notes:
- normalization includes lowercasing, punctuation removal, and whitespace cleanup
- tokenization is simple and language-agnostic
- metrics are intentionally lightweight and fast

## 4. Evaluation Layer

This module introduces the end-to-end evaluation pipeline, combining dataset samples, OpenWebUI responses, and metrics into structured results.

### Evaluation Runner

#### `runners/evaluation_runner.py`

EvaluationRunner
- orchestrates the full evaluation process
- executes queries against OpenWebUI
- computes retrieval and generation metrics per sample
- measures latency per request

Responsibilities:
- iterates over dataset samples
- calls `OpenWebUIClient`
- parses responses into `ChatResponse`
- computes:
  - `RetrievalMetrics`
  - `GenerationMetrics`
- produces structured `EvaluationResult` objects

Supports:
- single-sample evaluation (debugging)
- full dataset evaluation

### Evaluation Result Models

#### `models/evaluation_result.py`

EvaluationResult
- represents the result of one evaluated sample
- contains:
  - input sample
  - model response
  - retrieval metrics
  - generation metrics
  - latency
  - experiment metadata

#### `models/evaluation_summary.py`

EvaluationSummary
- represents aggregated metrics across all samples
- contains:
  - total samples
  - retrieval hit rate
  - mean first relevant rank
  - generation metric aggregates
  - latency statistics

Purpose:
- enables comparison between experiments
- provides a concise overview for reporting

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

RetrievalMetrics
    -> evaluates retrieval quality

GenerationMetrics
    -> evaluates retrieval quality

EvaluationSummary
    -> evaluation runner