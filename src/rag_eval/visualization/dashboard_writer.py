from __future__ import annotations

from pathlib import Path

from rag_eval.models.sweep_report import SweepReport

class DashboardWriter:
    @staticmethod
    def write_html_dashboard(
        report: SweepReport,
        output_path: str | Path,
        plots_dir_name: str = "plots",
    ) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for record in report.records:
            mean_rank = (
                f"{record.mean_first_relevant_rank:.3f}"
                if record.mean_first_relevant_rank is not None
                else "-"
            )

            rows.append(
                f"""
                <tr>
                    <td>{record.experiment_name}</td>
                    <td>{record.total_samples}</td>
                    <td>{record.retrieval_hit_rate:.3f}</td>
                    <td>{mean_rank}</td>
                    <td>{record.exact_match_rate:.3f}</td>
                    <td>{record.normalized_exact_match_rate:.3f}</td>
                    <td>{record.mean_token_f1:.3f}</td>
                    <td>{record.mean_faithfulness_overlap:.3f}</td>
                    <td>{record.mean_latency_seconds:.3f}</td>
                </tr>
                """
            )

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>RAG Evaluation Dashboard</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 24px;
            line-height: 1.4;
        }}
        h1, h2 {{
            margin-bottom: 8px;
        }}
        .summary {{
            margin-bottom: 24px;
            padding: 12px 16px;
            border: 1px solid #ddd;
            border-radius: 8px;
            background: #fafafa;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 32px;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px 10px;
            text-align: left;
        }}
        th {{
            background: #f3f3f3;
        }}
        .plot-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
            gap: 20px;
        }}
        .plot-card {{
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 12px;
            background: #fff;
        }}
        img {{
            max-width: 100%;
            height: auto;
            display: block;
        }}
    </style>
</head>
<body>
    <h1>RAG Evaluation Dashboard</h1>

    <div class="summary">
        <p><strong>Best retrieval experiment:</strong> {report.best_retrieval_experiment}</p>
        <p><strong>Best generation experiment:</strong> {report.best_generation_experiment}</p>
        <p><strong>Fastest experiment:</strong> {report.fastest_experiment}</p>
    </div>

    <h2>Experiment Summary</h2>
    <table>
        <thead>
            <tr>
                <th>Experiment</th>
                <th>Samples</th>
                <th>Hit@K</th>
                <th>Mean Rank</th>
                <th>Exact Match</th>
                <th>Norm. Exact</th>
                <th>Token F1</th>
                <th>Faithfulness</th>
                <th>Latency (s)</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>

    <h2>Plots</h2>
    <div class="plot-grid">
        <div class="plot-card">
            <h3>Retrieval Hit Rate</h3>
            <img src="{plots_dir_name}/retrieval_hit_rate.png" alt="Retrieval Hit Rate">
        </div>
        <div class="plot-card">
            <h3>Normalized Exact Match</h3>
            <img src="{plots_dir_name}/normalized_exact_match_rate.png" alt="Normalized Exact Match Rate">
        </div>
        <div class="plot-card">
            <h3>Mean Token F1</h3>
            <img src="{plots_dir_name}/mean_token_f1.png" alt="Mean Token F1">
        </div>
        <div class="plot-card">
            <h3>Faithfulness Overlap</h3>
            <img src="{plots_dir_name}/mean_faithfulness_overlap.png" alt="Faithfulness Overlap">
        </div>
        <div class="plot-card">
            <h3>Mean Latency</h3>
            <img src="{plots_dir_name}/mean_latency_seconds.png" alt="Mean Latency Seconds">
        </div>
    </div>
</body>
</html>
"""
        with path.open("w", encoding="utf-8") as file:
            file.write(html)