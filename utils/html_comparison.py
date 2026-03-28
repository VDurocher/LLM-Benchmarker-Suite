"""
Générateur de rapport HTML pour la comparaison multi-modèles.

Produit un tableau comparatif côte à côte de plusieurs modèles LLM
avec pass rate, score moyen, latence et mise en évidence du gagnant.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone
from typing import Any

from utils.html_primitives import (
    COLOR_ACCENT,
    COLOR_GRAY_BG,
    COLOR_GRAY_BORDER,
    COLOR_TEXT_DARK,
    COLOR_TEXT_MUTED,
    html_page_wrapper,
    pick_color,
    render_progress_bar,
)


def _build_comparison_header(test_set: str, winner: str, winner_reason: str, now: datetime) -> str:
    """Génère le header du rapport de comparaison."""
    # Échappement XSS sur les champs issus des arguments CLI et des noms de modèles
    safe_test_set = html.escape(test_set)
    safe_winner = html.escape(winner)
    safe_winner_reason = html.escape(winner_reason)
    return (
        f'<header style="background:linear-gradient(135deg,{COLOR_ACCENT} 0%,#4f46e5 100%);'
        f'color:#fff;padding:32px 40px;border-radius:16px;margin-bottom:28px;">'
        f'<div style="font-size:12px;font-weight:600;opacity:.75;text-transform:uppercase;'
        f'letter-spacing:.1em;margin-bottom:6px;">LLM Benchmarker Suite — Comparaison</div>'
        f'<h1 style="margin:0;font-size:24px;font-weight:800;">Test set : {safe_test_set}</h1>'
        f'<div style="margin-top:8px;opacity:.8;font-size:14px;">'
        f"Gagnant : <strong>{safe_winner}</strong> — {safe_winner_reason}</div>"
        f'<div style="margin-top:6px;opacity:.65;font-size:12px;">'
        f'{now.strftime("%Y-%m-%d %H:%M:%S")} UTC</div></header>'
    )


def _build_comparison_table(model_stats: list[dict[str, Any]], winner: str) -> str:
    """Génère le tableau comparatif multi-modèles."""
    th_style = (
        f'style="padding:12px 20px;text-align:left;font-size:12px;'
        f'color:{COLOR_TEXT_MUTED};font-weight:600;text-transform:uppercase;letter-spacing:.05em;"'
    )
    label_style = (
        f'style="padding:12px 20px;color:{COLOR_TEXT_MUTED};font-weight:600;'
        f'background:{COLOR_GRAY_BG};"'
    )

    # En-têtes colonnes modèles — échappement XSS sur les noms de modèles (arguments CLI)
    model_headers = "".join(
        f'<th {th_style}>{html.escape(stats["model_name"])}{"  🏆" if stats["model_name"] == winner else ""}</th>'
        for stats in model_stats
    )

    # Ligne pass rate avec barres de progression
    pass_rate_cells = "".join(
        f'<td style="padding:12px 20px;">'
        f'<span style="font-weight:800;font-size:16px;color:{pick_color(stats["pass_rate"])};">'
        f'{stats["pass_rate_percent"]}%</span>'
        f'<div style="margin-top:6px;">'
        f'{render_progress_bar(stats["pass_rate"], pick_color(stats["pass_rate"]), "8px")}'
        f"</div></td>"
        for stats in model_stats
    )

    def _value_cells(values: list[str]) -> str:
        return "".join(
            f'<td style="padding:12px 20px;color:{COLOR_TEXT_DARK};font-weight:500;">{v}</td>'
            for v in values
        )

    row_border = f'style="border-bottom:1px solid {COLOR_GRAY_BORDER};"'

    return (
        f'<section style="background:#fff;border:1px solid {COLOR_GRAY_BORDER};'
        f'border-radius:12px;margin-bottom:28px;overflow:hidden;">'
        f'<div style="padding:18px 24px;border-bottom:1px solid {COLOR_GRAY_BORDER};">'
        f'<h2 style="margin:0;font-size:17px;font-weight:700;color:{COLOR_TEXT_DARK};">'
        f"Comparaison des modèles</h2></div>"
        f'<div style="overflow-x:auto;">'
        f'<table style="width:100%;border-collapse:collapse;">'
        f'<thead><tr style="background:{COLOR_GRAY_BG};border-bottom:1px solid {COLOR_GRAY_BORDER};">'
        f'<th {th_style}>Métrique</th>{model_headers}</tr></thead>'
        f"<tbody>"
        f'<tr {row_border}><td {label_style}>Pass rate</td>{pass_rate_cells}</tr>'
        f'<tr {row_border}><td {label_style}>Score moyen</td>'
        f'{_value_cells([f"{s[\"avg_score\"]:.4f}" for s in model_stats])}</tr>'
        f'<tr {row_border}><td {label_style}>Cas passés</td>'
        f'{_value_cells([f"{s[\"passed_cases\"]} / {s[\"total_cases\"]}" for s in model_stats])}</tr>'
        f'<tr {row_border}><td {label_style}>Latence totale</td>'
        f'{_value_cells([f"{s[\"total_latency_ms\"]:.0f} ms" for s in model_stats])}</tr>'
        f"</tbody></table></div></section>"
    )


def build_comparison_html(
    report: dict[str, Any],
    model_stats: list[dict[str, Any]],
) -> str:
    """
    Génère le document HTML complet du rapport de comparaison multi-modèles.

    Args:
        report: Dictionnaire du rapport de comparaison JSON.
        model_stats: Liste des statistiques par modèle dans l'ordre d'affichage.

    Returns:
        Contenu HTML complet du rapport.
    """
    now = datetime.now(tz=timezone.utc)
    winner = report["winner"]
    test_set = report["test_set"]
    winner_reason = report["winner_reason"]

    footer = (
        f'<footer style="text-align:center;padding:20px;color:{COLOR_TEXT_MUTED};'
        f'font-size:12px;border-top:1px solid {COLOR_GRAY_BORDER};">'
        f'Généré par <strong style="color:{COLOR_ACCENT};">LLM-Benchmarker-Suite</strong>'
        f' — {now.strftime("%Y-%m-%d à %H:%M:%S")} UTC</footer>'
    )

    body = (
        _build_comparison_header(test_set, winner, winner_reason, now)
        + _build_comparison_table(model_stats, winner)
        + footer
    )
    return html_page_wrapper(title=f"Comparaison — {test_set}", body_content=body)
