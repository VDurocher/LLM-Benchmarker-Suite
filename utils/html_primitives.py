"""
Primitives de rendu HTML partagées entre les générateurs de rapports.

Contient les constantes de couleur, les fonctions de rendu de composants
atomiques (barre de progression, card, ligne de tableau) utilisées dans
html_report.py et html_comparison.py.
"""

from __future__ import annotations

from typing import Any

from config import PASS_RATE_TARGET

# Palettes de couleurs du design system
COLOR_GREEN = "#10b981"
COLOR_ORANGE = "#f59e0b"
COLOR_RED = "#ef4444"
COLOR_ACCENT = "#6366f1"
COLOR_GRAY_BG = "#f8fafc"
COLOR_GRAY_BORDER = "#e2e8f0"
COLOR_TEXT_DARK = "#1e293b"
COLOR_TEXT_MUTED = "#64748b"

# Seuil orange : pass rate ≥ 70 %
PASS_RATE_ORANGE_THRESHOLD = 0.70


def pick_color(pass_rate: float) -> str:
    """Retourne la couleur CSS correspondant au niveau de pass rate."""
    if pass_rate >= PASS_RATE_TARGET:
        return COLOR_GREEN
    if pass_rate >= PASS_RATE_ORANGE_THRESHOLD:
        return COLOR_ORANGE
    return COLOR_RED


def render_progress_bar(value: float, color: str, height: str = "10px") -> str:
    """Génère une barre de progression HTML inline."""
    pct = round(value * 100, 1)
    return (
        f'<div style="background:{COLOR_GRAY_BORDER};border-radius:9999px;'
        f'height:{height};overflow:hidden;width:100%;">'
        f'<div style="background:{color};height:100%;width:{pct}%;'
        f'border-radius:9999px;transition:width .3s;"></div></div>'
    )


def render_stat_card(label: str, value: str, sub: str = "") -> str:
    """Génère une card de statistique."""
    sub_html = (
        f'<span style="font-size:12px;color:{COLOR_TEXT_MUTED};display:block;margin-top:4px;">'
        f"{sub}</span>"
        if sub
        else ""
    )
    return (
        f'<div style="background:#fff;border:1px solid {COLOR_GRAY_BORDER};border-radius:12px;'
        f'padding:20px 24px;flex:1;min-width:160px;">'
        f'<div style="font-size:13px;color:{COLOR_TEXT_MUTED};font-weight:500;'
        f'text-transform:uppercase;letter-spacing:.05em;">{label}</div>'
        f'<div style="font-size:28px;font-weight:700;color:{COLOR_TEXT_DARK};margin-top:6px;">'
        f"{value}</div>{sub_html}</div>"
    )


def render_evaluator_row(name: str, stats: dict[str, Any]) -> str:
    """Génère une ligne de breakdown pour un évaluateur."""
    pass_rate = stats["pass_rate"]
    color = pick_color(pass_rate)
    bar = render_progress_bar(pass_rate, color, height="6px")
    return (
        f"<tr>"
        f'<td style="padding:10px 16px;font-weight:500;color:{COLOR_TEXT_DARK};">{name}</td>'
        f'<td style="padding:10px 16px;color:{COLOR_TEXT_MUTED};">{stats["total_runs"]}</td>'
        f'<td style="padding:10px 16px;">'
        f'<span style="color:{color};font-weight:600;">{round(pass_rate * 100, 1)}%</span>'
        f'<div style="margin-top:4px;">{bar}</div></td>'
        f'<td style="padding:10px 16px;color:{COLOR_TEXT_MUTED};">{stats["average_score"]:.4f}</td>'
        f'<td style="padding:10px 16px;color:{COLOR_TEXT_MUTED};">'
        f'{stats["average_latency_ms"]:.1f} ms</td>'
        f"</tr>"
    )


def render_case_row(case: dict[str, Any]) -> str:
    """Génère une ligne du tableau des cas de test."""
    passed = case["passed"]
    badge_color = COLOR_GREEN if passed else COLOR_RED
    badge_label = "PASS" if passed else "FAIL"
    badge_bg = "#ecfdf5" if passed else "#fef2f2"

    evaluators_html = "".join(
        f'<span style="display:inline-block;margin:2px;padding:2px 8px;'
        f'background:{"#ecfdf5" if ev["passed"] else "#fef2f2"};'
        f'color:{COLOR_GREEN if ev["passed"] else COLOR_RED};'
        f'border-radius:9999px;font-size:11px;font-weight:600;">'
        f'{ev["name"]} {ev["score"]:.3f}</span>'
        for ev in case["evaluators"]
    )

    score_pct = round(case["composite_score"] * 100, 1)
    bar = render_progress_bar(case["composite_score"], badge_color, "5px")
    prompt_preview = case.get("prompt_preview", "")

    return (
        f'<tr style="border-bottom:1px solid {COLOR_GRAY_BORDER};">'
        f'<td style="padding:12px 16px;font-family:monospace;font-size:13px;'
        f'color:{COLOR_ACCENT};">{case["case_id"]}</td>'
        f'<td style="padding:12px 16px;">'
        f'<span style="background:{badge_bg};color:{badge_color};padding:3px 10px;'
        f'border-radius:9999px;font-size:12px;font-weight:700;">{badge_label}</span></td>'
        f'<td style="padding:12px 16px;">'
        f'<span style="font-weight:700;color:{COLOR_TEXT_DARK};">{score_pct}%</span>'
        f'<div style="margin-top:4px;">{bar}</div></td>'
        f'<td style="padding:12px 16px;">{evaluators_html}</td>'
        f'<td style="padding:12px 16px;font-size:12px;color:{COLOR_TEXT_MUTED};'
        f'max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">'
        f"{prompt_preview}</td>"
        f"</tr>"
    )


def html_page_wrapper(title: str, body_content: str) -> str:
    """Enveloppe le contenu dans un document HTML5 complet."""
    return (
        f"<!DOCTYPE html>\n<html lang='fr'>\n<head>\n"
        f'  <meta charset="UTF-8">\n'
        f'  <meta name="viewport" content="width=device-width,initial-scale=1">\n'
        f"  <title>{title}</title>\n"
        f"  <style>\n"
        f"    *, *::before, *::after {{ box-sizing: border-box; }}\n"
        f"    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "
        f"'Segoe UI', Roboto, sans-serif; background: {COLOR_GRAY_BG}; "
        f"color: {COLOR_TEXT_DARK}; line-height: 1.5; }}\n"
        f"    table {{ border-spacing: 0; }}\n"
        f"    th, td {{ white-space: nowrap; }}\n"
        f"  </style>\n</head>\n<body>\n"
        f'  <div style="max-width:1200px;margin:0 auto;padding:32px 20px;">\n'
        f"    {body_content}\n"
        f"  </div>\n</body>\n</html>"
    )
