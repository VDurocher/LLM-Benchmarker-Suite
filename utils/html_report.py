"""
Visual HTML report generator for the LLM-Benchmarker Suite.

Produces a standalone HTML file (no external dependencies) including:
- Header with production verdict
- Global statistics cards
- Color-coded pass rate progress bar
- Detailed test cases table
- Per-evaluator breakdown with mini-bars
- Timestamped footer
"""

from __future__ import annotations

import html
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import PASS_RATE_TARGET, REPORT_OUTPUT_DIR
from evaluators.base_evaluator import EvaluationResult
from utils.html_primitives import (
    COLOR_ACCENT,
    COLOR_GRAY_BORDER,
    COLOR_GRAY_BG,
    COLOR_TEXT_DARK,
    COLOR_TEXT_MUTED,
    html_page_wrapper,
    pick_color,
    render_case_row,
    render_evaluator_row,
    render_progress_bar,
    render_stat_card,
)
from utils.logger import get_logger
from utils.stats import compute_evaluator_stats

logger = get_logger(__name__)


def _build_header(model_name: str, test_set: str, pass_rate: float, now: datetime) -> str:
    """Generates the report header with production verdict."""
    production_ready = pass_rate >= PASS_RATE_TARGET
    verdict_label = "PRODUCTION READY" if production_ready else "NOT PRODUCTION READY"
    verdict_color = "#10b981" if production_ready else "#ef4444"
    badge_bg = "rgba(16,185,129,.25)" if production_ready else "rgba(239,68,68,.25)"
    # XSS escaping on user-controlled fields from CLI
    safe_model_name = html.escape(model_name)
    safe_test_set = html.escape(test_set)
    return (
        f'<header style="background:linear-gradient(135deg,{COLOR_ACCENT} 0%,#4f46e5 100%);'
        f'color:#fff;padding:32px 40px;border-radius:16px;margin-bottom:28px;">'
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;'
        f'flex-wrap:wrap;gap:16px;">'
        f'<div><div style="font-size:12px;font-weight:600;opacity:.75;text-transform:uppercase;'
        f'letter-spacing:.1em;margin-bottom:6px;">LLM Benchmarker Suite</div>'
        f'<h1 style="margin:0;font-size:26px;font-weight:800;">{safe_model_name}</h1>'
        f'<div style="margin-top:6px;opacity:.8;font-size:14px;">Test set : <strong>{safe_test_set}</strong></div></div>'
        f'<div style="text-align:right;">'
        f'<span style="display:inline-block;background:{badge_bg};border:2px solid {verdict_color};'
        f'color:#fff;padding:8px 20px;border-radius:9999px;font-weight:800;font-size:15px;">'
        f"{verdict_label}</span>"
        f'<div style="margin-top:10px;opacity:.7;font-size:12px;">'
        f'{now.strftime("%Y-%m-%d %H:%M:%S")} UTC</div></div></div></header>'
    )


def _build_progress_section(pass_rate: float) -> str:
    """Generates the global pass rate progress bar section."""
    color = pick_color(pass_rate)
    bar = render_progress_bar(pass_rate, color, "14px")
    target_pct = round(PASS_RATE_TARGET * 100, 0)
    return (
        f'<section style="background:#fff;border:1px solid {COLOR_GRAY_BORDER};'
        f'border-radius:12px;padding:20px 24px;margin-bottom:28px;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">'
        f'<span style="font-weight:700;color:{COLOR_TEXT_DARK};font-size:15px;">Global Pass Rate</span>'
        f'<span style="font-weight:800;font-size:22px;color:{color};">{round(pass_rate * 100, 1)}%</span></div>'
        f"{bar}"
        f'<div style="display:flex;justify-content:space-between;margin-top:6px;'
        f'font-size:12px;color:{COLOR_TEXT_MUTED};">'
        f"<span>0%</span><span>Target: {target_pct:.0f}%</span><span>100%</span></div></section>"
    )


def _build_cases_table(case_results: list[dict[str, Any]]) -> str:
    """Generates the detailed test cases table."""
    rows = "".join(render_case_row(c) for c in case_results)
    th_style = (
        f'style="padding:12px 16px;text-align:left;font-size:12px;'
        f'color:{COLOR_TEXT_MUTED};font-weight:600;text-transform:uppercase;letter-spacing:.05em;"'
    )
    return (
        f'<section style="background:#fff;border:1px solid {COLOR_GRAY_BORDER};'
        f'border-radius:12px;margin-bottom:28px;overflow:hidden;">'
        f'<div style="padding:18px 24px;border-bottom:1px solid {COLOR_GRAY_BORDER};">'
        f'<h2 style="margin:0;font-size:17px;font-weight:700;color:{COLOR_TEXT_DARK};">'
        f"Results per test case</h2></div>"
        f'<div style="overflow-x:auto;"><table style="width:100%;border-collapse:collapse;">'
        f'<thead><tr style="background:{COLOR_GRAY_BG};border-bottom:1px solid {COLOR_GRAY_BORDER};">'
        f"<th {th_style}>Case ID</th><th {th_style}>Status</th>"
        f"<th {th_style}>Score</th><th {th_style}>Evaluators</th>"
        f"<th {th_style}>Prompt</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div></section>"
    )


def _build_breakdown_table(evaluator_stats: dict[str, dict[str, Any]]) -> str:
    """Generates the per-evaluator breakdown section."""
    rows = "".join(render_evaluator_row(name, stats) for name, stats in evaluator_stats.items())
    th_style = (
        f'style="padding:10px 16px;text-align:left;font-size:12px;'
        f'color:{COLOR_TEXT_MUTED};font-weight:600;text-transform:uppercase;letter-spacing:.05em;"'
    )
    return (
        f'<section style="background:#fff;border:1px solid {COLOR_GRAY_BORDER};'
        f'border-radius:12px;margin-bottom:28px;overflow:hidden;">'
        f'<div style="padding:18px 24px;border-bottom:1px solid {COLOR_GRAY_BORDER};">'
        f'<h2 style="margin:0;font-size:17px;font-weight:700;color:{COLOR_TEXT_DARK};">'
        f"Evaluator Breakdown</h2></div>"
        f'<div style="overflow-x:auto;"><table style="width:100%;border-collapse:collapse;">'
        f'<thead><tr style="background:{COLOR_GRAY_BG};border-bottom:1px solid {COLOR_GRAY_BORDER};">'
        f"<th {th_style}>Evaluator</th><th {th_style}>Runs</th>"
        f"<th {th_style}>Pass rate</th><th {th_style}>Avg score</th>"
        f"<th {th_style}>Avg latency</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div></section>"
    )


class HtmlReportGenerator:
    """
    Generates a standalone visual HTML report for a benchmark session.

    The produced file has no external library dependencies (inline CSS,
    minimal JavaScript). It can be opened directly in a browser or
    embedded in a CI/CD pipeline for archiving.
    """

    def __init__(self, model_name: str, test_set: str) -> None:
        self._model_name = model_name
        self._test_set = test_set
        self._session_start = datetime.now(tz=timezone.utc)
        self._case_results: list[dict[str, Any]] = []

    def add_case_result(
        self,
        case_id: str,
        prompt: str,
        expected_output: str,
        model_output: str,
        evaluation_results: list[EvaluationResult],
        composite_score: float,
        passed: bool,
    ) -> None:
        """Records the results of a test case for HTML rendering."""
        self._case_results.append(
            {
                "case_id": case_id,
                "prompt_preview": prompt[:150] + "..." if len(prompt) > 150 else prompt,
                "composite_score": round(composite_score, 4),
                "passed": passed,
                "evaluators": [
                    {
                        "name": result.evaluator_name,
                        "passed": result.passed,
                        "score": result.score,
                        "latency_ms": round(result.latency_ms, 2),
                        "error": result.error,
                    }
                    for result in evaluation_results
                ],
            }
        )

    def _build_html(self) -> str:
        """Assembles the complete HTML document."""
        now = datetime.now(tz=timezone.utc)
        total_cases = len(self._case_results)
        passed_count = sum(1 for c in self._case_results if c["passed"])
        pass_rate = passed_count / total_cases if total_cases > 0 else 0.0
        avg_score = (
            sum(c["composite_score"] for c in self._case_results) / total_cases
            if total_cases > 0
            else 0.0
        )
        total_latency = sum(
            ev["latency_ms"]
            for case in self._case_results
            for ev in case["evaluators"]
        )
        evaluator_stats = compute_evaluator_stats(self._case_results)

        cards = (
            render_stat_card("Total cases", str(total_cases), f"{passed_count} passed / {total_cases - passed_count} failed")
            + render_stat_card("Pass rate", f"{round(pass_rate * 100, 1)}%", f"Target: {round(PASS_RATE_TARGET * 100, 0):.0f}%")
            + render_stat_card("Avg score", f"{round(avg_score, 3)}", "Composite weighted")
            + render_stat_card("Total latency", f"{round(total_latency, 0):.0f} ms", "All evaluators")
        )
        cards_section = (
            f'<section style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:28px;">'
            f"{cards}</section>"
        )

        footer = (
            f'<footer style="text-align:center;padding:20px;color:{COLOR_TEXT_MUTED};'
            f'font-size:12px;border-top:1px solid {COLOR_GRAY_BORDER};margin-top:8px;">'
            f'Generated by <strong style="color:{COLOR_ACCENT};">LLM-Benchmarker-Suite</strong>'
            f' — {now.strftime("%Y-%m-%d at %H:%M:%S")} UTC</footer>'
        )

        body = (
            _build_header(self._model_name, self._test_set, pass_rate, now)
            + cards_section
            + _build_progress_section(pass_rate)
            + _build_cases_table(self._case_results)
            + _build_breakdown_table(evaluator_stats)
            + footer
        )
        return html_page_wrapper(
            title=f"Benchmark — {self._model_name} / {self._test_set}",
            body_content=body,
        )

    def save(self, output_dir: str | None = None) -> Path:
        """
        Persists the HTML report to the output directory.
        Returns the absolute path of the created file.
        """
        target_dir = Path(output_dir or REPORT_OUTPUT_DIR)
        target_dir.mkdir(parents=True, exist_ok=True)

        timestamp = self._session_start.strftime("%Y%m%d_%H%M%S")
        # Sanitise model name to prevent path traversal in filename
        safe_model = re.sub(r"[^a-zA-Z0-9_.-]", "_", self._model_name)
        filename = f"benchmark_{safe_model}_{self._test_set}_{timestamp}.html"
        output_path = target_dir / filename

        with open(output_path, "w", encoding="utf-8") as file_handle:
            file_handle.write(self._build_html())

        total_cases = len(self._case_results)
        passed_count = sum(1 for c in self._case_results if c["passed"])
        pass_rate_pct = (passed_count / total_cases * 100) if total_cases > 0 else 0.0

        logger.info(
            "HTML report saved → %s (pass rate: %.1f%%)",
            output_path,
            pass_rate_pct,
        )
        return output_path.resolve()
