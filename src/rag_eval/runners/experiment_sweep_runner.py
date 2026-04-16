from __future__ import annotations

from pathlib import Path
from typing import List

from rag_eval.clients.openwebui_client import OpenWebUIClient
from rag_eval.clients.openwebui_config_client import OpenWebUIConfigClient
from rag_eval.config.config_loader import ConfigLoader
from rag_eval.config.retrieval_config_mapper import RetrievalConfigMapper
from rag_eval.datasets.dataset_loader import DatasetLoader
from rag_eval.models.evaluation_summary import EvaluationSummary
from rag_eval.runners.evaluation_runner import EvaluationRunner
from rag_eval.storage.result_writer import ResultWriter
from rag_eval.reporting.report_writer import ReportWriter
from rag_eval.reporting.sweep_report_builder import SweepReportBuilder

def run_experiment_sweep(
    *,
    dataset_path: str,
    experiment_config_paths: List[str],
    base_url: str,
    api_key: str,
) -> List[EvaluationSummary]:
    samples = DatasetLoader.load(dataset_path)

    config_client = OpenWebUIConfigClient(
        base_url=base_url,
        api_key=api_key,
    )

    summaries: List[EvaluationSummary] = []

    for experiment_config_path in experiment_config_paths:
        experiment = ConfigLoader.load(experiment_config_path)

        retrieval_updates = RetrievalConfigMapper.build_update_payload(experiment)
        if retrieval_updates:
            config_client.update_retrieval_config(retrieval_updates)

        openwebui_client = OpenWebUIClient(
            base_url=base_url,
            api_key=api_key,
            model=experiment.model,
        )

        runner = EvaluationRunner(
            client=openwebui_client,
            retrieval_k=experiment.retrieval_k,
        )

        results = runner.run_dataset(
            samples=samples,
            experiment_name=experiment.name,
            collection_id=experiment.collection_id,
            temperature=experiment.temperature,
            max_tokens=experiment.max_tokens,
            history=experiment.history,
        )

        summary = runner.summarize(
            experiment_name=experiment.name,
            results=results,
        )
        summaries.append(summary)

        output_dir = Path(experiment.output_dir)
        suffix = f"_{experiment.output_tag}" if experiment.output_tag else ""

        ResultWriter.write_results_json(
            results,
            output_dir / f"{experiment.name}{suffix}_results.json",
        )
        ResultWriter.write_results_jsonl(
            results,
            output_dir / f"{experiment.name}{suffix}_results.jsonl",
        )
        ResultWriter.write_summary(
            summary,
            output_dir / f"{experiment.name}{suffix}_summary.json",
        )
    
    if summaries:
        report = SweepReportBuilder.build(summaries)

        base_output_dir = Path("outputs")
        ReportWriter.write_json(
            report,
            base_output_dir / "sweep_report.json",
        )
        ReportWriter.write_markdown(
            report,
            base_output_dir / "sweep_report.md",
        )

    return summaries