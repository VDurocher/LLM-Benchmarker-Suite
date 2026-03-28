"""
Tests Playwright pour les rapports HTML générés.

Vérifie que les rapports HTML single-model et multi-model :
- Se chargent sans erreur JavaScript
- Contiennent les sections structurelles attendues
- Affichent les métriques correctement
- Sont visuellement cohérents (badges, barres de progression, tableaux)

Ces tests nécessitent les rapports sample pré-générés dans /reports/.
Ils tournent sans serveur web — fichiers locaux via file:// URI.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect

# Répertoire contenant les rapports générés
REPORTS_DIR = Path(__file__).parent.parent / "reports"


def _latest_report(pattern: str) -> Path | None:
    """Retourne le fichier de rapport le plus récent correspondant au pattern."""
    matches = sorted(REPORTS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def single_model_html() -> Path:
    """Chemin vers le rapport HTML single-model le plus récent."""
    report = _latest_report("benchmark_gpt-4o_all_*.html")
    if report is None:
        pytest.skip("Aucun rapport single-model trouvé — lancez main.py --format html d'abord")
    return report


@pytest.fixture(scope="session")
def comparison_html() -> Path:
    """Chemin vers le rapport HTML de comparaison le plus récent."""
    report = _latest_report("comparison_*.html")
    if report is None:
        pytest.skip("Aucun rapport de comparaison trouvé — lancez compare_runner.py --format html d'abord")
    return report


# ---------------------------------------------------------------------------
# Tests rapport single-model
# ---------------------------------------------------------------------------

class TestSingleModelReport:
    def test_page_loads_without_js_errors(self, page: Page, single_model_html: Path) -> None:
        """Le rapport HTML doit se charger sans erreur JavaScript."""
        js_errors: list[str] = []
        page.on("pageerror", lambda exc: js_errors.append(str(exc)))

        page.goto(f"file:///{single_model_html.as_posix()}")
        page.wait_for_load_state("domcontentloaded")

        assert js_errors == [], f"Erreurs JS détectées : {js_errors}"

    def test_title_contains_benchmark(self, page: Page, single_model_html: Path) -> None:
        """Le titre de la page doit mentionner LLM Benchmarker."""
        page.goto(f"file:///{single_model_html.as_posix()}")
        title = page.title().lower()
        assert "benchmark" in title or "llm" in title, (
            f"Le titre ne mentionne pas 'benchmark' ni 'llm' : '{page.title()}'"
        )

    def test_header_section_visible(self, page: Page, single_model_html: Path) -> None:
        """Le header avec le modèle et le verdict doit être visible."""
        page.goto(f"file:///{single_model_html.as_posix()}")
        header = page.locator("header")
        expect(header).to_be_visible()

    def test_verdict_badge_present(self, page: Page, single_model_html: Path) -> None:
        """Un badge PRODUCTION READY ou BELOW TARGET doit être affiché."""
        page.goto(f"file:///{single_model_html.as_posix()}")
        page_text = page.inner_text("body")
        assert "PRODUCTION READY" in page_text or "BELOW TARGET" in page_text, (
            "Aucun badge de verdict trouvé dans le rapport"
        )

    def test_stat_cards_present(self, page: Page, single_model_html: Path) -> None:
        """Les 4 cartes de statistiques doivent être présentes (total, pass rate, score, latency)."""
        page.goto(f"file:///{single_model_html.as_posix()}")
        page_text = page.inner_text("body")
        # Vérifier la présence des libellés des stat cards
        assert "pass rate" in page_text.lower() or "passed" in page_text.lower()
        assert "score" in page_text.lower()

    def test_test_cases_table_present(self, page: Page, single_model_html: Path) -> None:
        """Le tableau des cas de test doit être présent et non vide."""
        page.goto(f"file:///{single_model_html.as_posix()}")
        # Il doit y avoir au moins une ligne de tableau de cas de test
        rows = page.locator("table tbody tr")
        count = rows.count()
        assert count > 0, f"Aucune ligne dans le tableau des cas de test (count={count})"

    def test_evaluator_breakdown_section(self, page: Page, single_model_html: Path) -> None:
        """La section de breakdown par évaluateur doit être visible."""
        page.goto(f"file:///{single_model_html.as_posix()}")
        page_text = page.inner_text("body")
        # Au moins un évaluateur doit être listé
        assert any(ev in page_text for ev in [
            "similarity", "hallucination", "format", "consistency", "llm_judge"
        ]), "Aucun évaluateur trouvé dans le breakdown"

    def test_pass_fail_indicators_present(self, page: Page, single_model_html: Path) -> None:
        """Des indicateurs PASS/FAIL doivent apparaître dans le tableau."""
        page.goto(f"file:///{single_model_html.as_posix()}")
        page_text = page.inner_text("body")
        assert "PASS" in page_text or "FAIL" in page_text, (
            "Aucun indicateur PASS/FAIL trouvé"
        )

    def test_no_placeholder_text(self, page: Page, single_model_html: Path) -> None:
        """Le rapport ne doit pas contenir de placeholders non remplacés."""
        page.goto(f"file:///{single_model_html.as_posix()}")
        page_text = page.inner_text("body")
        placeholders = ["{{", "}}", "[TODO]", "undefined", "NaN"]
        for placeholder in placeholders:
            assert placeholder not in page_text, (
                f"Placeholder non remplacé trouvé : '{placeholder}'"
            )

    def test_report_is_self_contained(self, single_model_html: Path) -> None:
        """Le rapport HTML ne doit pas référencer de ressources externes (CDN, URLs)."""
        content = single_model_html.read_text(encoding="utf-8")
        # Vérification que pas de CDN externe
        external_indicators = ["https://cdn.", "http://cdn.", "https://fonts.googleapis"]
        for indicator in external_indicators:
            assert indicator not in content, (
                f"Dépendance externe détectée : '{indicator}' — le rapport doit être standalone"
            )


# ---------------------------------------------------------------------------
# Tests rapport de comparaison
# ---------------------------------------------------------------------------

class TestComparisonReport:
    def test_comparison_page_loads(self, page: Page, comparison_html: Path) -> None:
        """Le rapport de comparaison doit se charger sans erreur."""
        js_errors: list[str] = []
        page.on("pageerror", lambda exc: js_errors.append(str(exc)))

        page.goto(f"file:///{comparison_html.as_posix()}")
        page.wait_for_load_state("domcontentloaded")

        assert js_errors == [], f"Erreurs JS dans le rapport de comparaison : {js_errors}"

    def test_comparison_table_present(self, page: Page, comparison_html: Path) -> None:
        """Le tableau de comparaison multi-modèles doit être présent."""
        page.goto(f"file:///{comparison_html.as_posix()}")
        page_text = page.inner_text("body")
        assert "pass rate" in page_text.lower() or "Pass rate" in page_text

    def test_winner_highlighted(self, page: Page, comparison_html: Path) -> None:
        """Le gagnant doit être mis en évidence dans le rapport."""
        page.goto(f"file:///{comparison_html.as_posix()}")
        page_text = page.inner_text("body")
        # Le rapport de comparaison doit mentionner un gagnant
        assert "gagnant" in page_text.lower() or "winner" in page_text.lower() or "🏆" in page_text

    def test_multiple_models_in_table(self, page: Page, comparison_html: Path) -> None:
        """La table doit contenir au moins 2 colonnes modèle."""
        page.goto(f"file:///{comparison_html.as_posix()}")
        # Les en-têtes de colonnes dans thead
        headers = page.locator("thead th")
        count = headers.count()
        # Au moins 3 colonnes : Métrique + 2 modèles
        assert count >= 3, f"Pas assez de colonnes dans la table de comparaison (count={count})"
