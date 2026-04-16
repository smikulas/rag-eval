# OpenWebUI RAG Evaluation Framework

Initial modules for a Python-based evaluation framework targeting OpenWebUI Retrieval-Augmented Generation (RAG).

## Purpose

This project builds a structured and reproducible evaluation pipeline for OpenWebUI’s built-in RAG system.

The current implementation focuses on three foundational layers:

1. **OpenWebUI integration layer**
2. **Dataset modeling and loading layer**
3. **Metrics layer for retrieval and generation**
4. **Evaluation layer**
5. **Storage layer**
6. **Configuration layer**
7. **Experiment execution**
8. **Experiment sweep**
9. **Reporting layer**

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

## 5. Storage Layer

This module provides structured persistence for evaluation results.

### Serialization

#### `utils/serialization.py`

Provides utilities to convert dataclass-based objects into JSON-compatible dictionaries.

Responsibilities:
- converts nested dataclasses using `asdict`
- ensures compatibility with JSON/JSONL output

### Result Writer

#### `storage/result_writer.py`

ResultWriter
- persists evaluation outputs to disk
- supports:
  - JSON (full results)
  - JSONL (streaming / large datasets)
  - summary JSON

Responsibilities:
- writes per-sample `EvaluationResult`
- writes aggregated `EvaluationSummary`
- ensures output directories exist

## 6. Configuration Layer

This module introduces experiment configuration management, enabling reproducible and configurable evaluation runs.

### Experiment Configuration

#### `models/experiment_config.py`

ExperimentConfig
- represents one evaluation experiment
- defines:
  - model to use
  - collection to query
  - generation parameters (temperature, max_tokens)
  - retrieval evaluation parameter (retrieval_k)
  - output configuration (directory, tag)
  - optional chat history
  - retrieval settings for OpenWebUI

Purpose:
- provides a single source of truth for one experiment
- enables reproducibility and comparison across runs

### Config Loader

#### `config/config_loader.py`

ConfigLoader
- loads experiment configuration from YAML
- validates required fields
- converts raw YAML into `ExperimentConfig`

Supports:
- simple scalar parameters
- nested `retrieval_settings` for OpenWebUI

### Retrieval Config Mapping

#### `config/retrieval_config_mapper.py`

RetrievalConfigMapper
- converts `ExperimentConfig` into OpenWebUI retrieval config payload
- filters only supported keys
- prepares update payload for `/api/v1/retrieval/config`

Purpose:
- bridges evaluation config and OpenWebUI runtime configuration

### OpenWebUI Config Client

#### `clients/openwebui_config_client.py`

OpenWebUIConfigClient
- interacts with OpenWebUI retrieval configuration API
- supports:
  - reading current config
  - updating config via API

Purpose:
- ensures experiment settings are applied before evaluation

## 7. Experiment Execution

This module connects all components into a full evaluation workflow.

### Experiment Runner

#### `runners/run_experiment.py`

run_experiment
- executes one full experiment

Workflow:
1. load experiment config
2. load dataset
3. update OpenWebUI retrieval config
4. run evaluation over dataset
5. compute summary
6. store results and summary

## 8. Experiment Sweep Layer

This module enables running multiple experiments sequentially and comparing their results.

### Sweep Runner

#### `runners/experiment_sweep_runner.py`

run_experiment_sweep
- executes multiple experiments in sequence
- applies each experiment configuration
- runs evaluation on the same dataset
- stores per-experiment results and summaries

Workflow:
1. load dataset once
2. iterate over experiment configs
3. update OpenWebUI retrieval settings per experiment
4. run evaluation
5. store results and summary
6. collect summaries for comparison

Purpose:
- enables controlled experiments (e.g. reranker on/off, different top_k)
- ensures consistent evaluation across configurations

## 9. Reporting Layer

This module aggregates and compares experiment results.

### Sweep Summary Models

#### `models/sweep_summary_record.py`

SweepSummaryRecord
- represents summary metrics for one experiment
- contains:
  - experiment name
  - retrieval metrics
  - generation metrics
  - latency statistics

#### `models/sweep_report.py`

SweepReport
- represents comparison across multiple experiments
- contains:
  - list of experiment records
  - best retrieval experiment
  - best generation experiment
  - fastest experiment

### Report Builder

#### `reporting/sweep_report_builder.py`

SweepReportBuilder
- builds a comparison report from multiple `EvaluationSummary`
- identifies:
  - best retrieval experiment (highest hit rate / best rank)
  - best generation experiment (accuracy + quality metrics)
  - fastest experiment (lowest latency)

Purpose:
- provides structured comparison logic
- separates evaluation from reporting

### Report Writer

#### `reporting/report_writer.py`

ReportWriter
- persists sweep comparison results
- supports:
  - JSON (machine-readable)
  - Markdown (human-readable)

Outputs:
- `sweep_report.json`
- `sweep_report.md`

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

ResultWriter
    -> JSON / JSONL outputs

ConfigLoader
    -> ExperimentConfig

RetrievalConfigMapper
    -> OpenWebUI config payload

OpenWebUIConfigClient
    -> applies retrieval settings

run_experiment_sweep
    -> multiple EvaluationRunner executions
    -> multiple EvaluationSummary

SweepReportBuilder
    -> SweepReport

ReportWriter
    -> JSON / Markdown reports