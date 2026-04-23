from __future__ import annotations

from datetime import datetime
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
from rag_eval.visualization.visualization_runner import VisualizationRunner

DEFAULT_SWEEP_OUTPUT_ROOT = "outputs"

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

    experiments = [
        ConfigLoader.load(experiment_config_path)
        for experiment_config_path in experiment_config_paths
    ]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sweep_name = _build_sweep_name(experiments)
    sweep_output_dir = Path(DEFAULT_SWEEP_OUTPUT_ROOT) / f"{sweep_name}_{timestamp}"
    sweep_output_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[EvaluationSummary] = []

    for experiment in experiments:
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

        suffix = f"_{experiment.output_tag}" if experiment.output_tag else ""
        experiment_output_dir = sweep_output_dir / experiment.name
        experiment_output_dir.mkdir(parents=True, exist_ok=True)

        ResultWriter.write_results_json(
            results,
            experiment_output_dir / f"{experiment.name}{suffix}_results.json",
        )
        ResultWriter.write_results_jsonl(
            results,
            experiment_output_dir / f"{experiment.name}{suffix}_results.jsonl",
        )
        ResultWriter.write_summary(
            summary,
            experiment_output_dir / f"{experiment.name}{suffix}_summary.json",
        )
    
    if summaries:
        report = SweepReportBuilder.build(summaries)

        ReportWriter.write_json(
            report,
            sweep_output_dir / "sweep_report.json",
        )
        ReportWriter.write_markdown(
            report,
            sweep_output_dir / "sweep_report.md",
        )

        VisualizationRunner.generate(
            summaries=summaries,
            output_dir=sweep_output_dir / "visualization",
        )

    return summaries


def _build_sweep_name(experiments: List) -> str:
    if len(experiments) == 1:
        return f"sweep_{experiments[0].name}"

    return "sweep_multi_experiment"