from __future__ import annotations

from pathlib import Path

from rag_eval.reporting.report_writer import ReportWriter
from rag_eval.reporting.sweep_report_builder import SweepReportBuilder
from rag_eval.models.evaluation_summary import EvaluationSummary
from rag_eval.visualization.dashboard_writer import DashboardWriter
from rag_eval.visualization.plot_writer import PlotWriter

class VisualizationRunner:
    @staticmethod
    def generate(
        summaries: list[EvaluationSummary],
        output_dir: str | Path,
    ) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        report = SweepReportBuilder.build(summaries)
        
        ReportWriter.write_markdown(report, output_path / "sweep_report.md")

        plots_dir = output_path / "plots"
        PlotWriter.write_all(report, plots_dir)

        DashboardWriter.write_html_dashboard(
            report=report,
            output_path=output_path / "dashboard.html",
            plots_dir_name="plots",
        )