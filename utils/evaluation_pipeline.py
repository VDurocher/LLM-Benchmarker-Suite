"""
Pipeline d'évaluation partagé entre main.py et compare_runner.py.

Fournit les fonctions de chargement des cas de test, d'instanciation
des évaluateurs, de récupération d'outputs live, et d'évaluation unitaire.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from config import (
    DEFAULT_WEIGHTS,
    LLM_JUDGE_WEIGHT,
    SUPPORTED_PROVIDERS,
    EvaluatorWeights,
)
from evaluators import (
    CodeEvaluator,
    ConsistencyEvaluator,
    EvaluationResult,
    FormatEvaluator,
    HallucinationEvaluator,
    SimilarityEvaluator,
)
from utils.logger import get_logger

logger = get_logger(__name__)

# Répertoire contenant les fichiers de cas de test
DATA_DIR = Path(__file__).parent.parent / "data"

# Ensembles de tests agrégés par "all"
_ALL_TEST_SETS = ["safety", "logic", "format", "consistency", "reasoning", "instruction_following"]


def load_test_cases(test_set: str) -> list[dict[str, Any]]:
    """Charge les cas de test depuis les fichiers JSON. 'all' agrège tous les sets."""
    if test_set == "all":
        all_cases: list[dict[str, Any]] = []
        for available_set in _ALL_TEST_SETS:
            file_path = DATA_DIR / f"test_cases_{available_set}.json"
            if file_path.exists():
                all_cases.extend(_load_single_test_set(available_set))
        logger.info("Total cas de test chargés : %d", len(all_cases))
        return all_cases
    return _load_single_test_set(test_set)


def _load_single_test_set(test_set: str) -> list[dict[str, Any]]:
    """Charge un fichier de cas de test par nom d'ensemble."""
    file_path = DATA_DIR / f"test_cases_{test_set}.json"
    if not file_path.exists():
        raise FileNotFoundError(f"Ensemble de tests non trouvé : {file_path}")
    with open(file_path, encoding="utf-8") as file_handle:
        data = json.load(file_handle)
    cases = data.get("cases", [])
    logger.info("Chargement de '%s' : %d cas de test", test_set, len(cases))
    return cases


def build_evaluators(judge_client: Any | None = None) -> dict[str, Any]:
    """
    Instancie tous les évaluateurs disponibles.

    Args:
        judge_client: Instance de LLMClient pour le juge LLM (optionnel).
                      Si None, le juge LLM n'est pas inclus dans le pipeline.

    Returns:
        Dict nom → instance d'évaluateur.
    """
    from evaluators.llm_judge_evaluator import LLMJudgeEvaluator

    evaluators: dict[str, Any] = {
        "similarity": SimilarityEvaluator(),
        "hallucination": HallucinationEvaluator(),
        "format": FormatEvaluator(),
        "code": CodeEvaluator(),
        "consistency": ConsistencyEvaluator(),
    }

    if judge_client is not None:
        evaluators["llm_judge"] = LLMJudgeEvaluator(judge_client=judge_client)
        logger.info(
            "LLM-as-a-judge activé — modèle juge : %s (%s)",
            judge_client.model,
            judge_client.provider,
        )

    return evaluators


def build_api_client(provider: str, api_key: str, model: str) -> Any:
    """
    Instancie le client d'inférence live approprié selon le provider.

    Args:
        provider: "openai" ou "anthropic".
        api_key: Clé d'API du provider.
        model: Identifiant du modèle cible.

    Returns:
        Instance de LLMClient (OpenAIClient ou AnthropicClient).

    Raises:
        ValueError: Si le provider n'est pas supporté.
    """
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Provider non supporté : '{provider}'. Valeurs acceptées : {SUPPORTED_PROVIDERS}"
        )

    if provider == "openai":
        from api.openai_client import OpenAIClient
        return OpenAIClient(api_key=api_key, model=model)

    from api.anthropic_client import AnthropicClient
    return AnthropicClient(api_key=api_key, model=model)


def fetch_live_outputs(
    test_cases: list[dict[str, Any]],
    client: Any,
    force_refresh: bool = False,
) -> list[dict[str, Any]]:
    """
    Remplace les `model_output` vides en appelant l'API live du modèle.

    En mode live, les cas de test n'ont pas besoin d'un `model_output` pré-rempli.
    Cette fonction appelle le modèle pour chaque cas et injecte la réponse.

    Args:
        test_cases: Liste des cas de test (dictionnaires).
        client: Client d'inférence LLM configuré.
        force_refresh: Si True, remplace aussi les outputs déjà remplis.

    Returns:
        Liste de cas de test enrichis avec les outputs du modèle.
    """
    enriched: list[dict[str, Any]] = []

    for index, case in enumerate(test_cases, start=1):
        case_id = case.get("id", f"case_{index:03d}")
        existing_output: str = case.get("model_output", "")

        if existing_output and not force_refresh:
            # Output déjà présent — pas d'appel API nécessaire
            enriched.append(case)
            continue

        prompt: str = case.get("prompt", "")
        logger.info("[%d/%d] Inférence live — cas : %s (modèle : %s)", index, len(test_cases), case_id, client.model)

        try:
            model_output = client.complete(prompt=prompt)
            updated_case = {**case, "model_output": model_output}
            enriched.append(updated_case)
            logger.info("  → Réponse reçue (%d caractères)", len(model_output))
        except RuntimeError as exc:
            logger.error("  → Erreur inférence pour '%s' : %s — cas ignoré", case_id, exc)
            # Conserver le cas avec output vide pour ne pas bloquer le pipeline
            enriched.append(case)

    return enriched


def resolve_api_key(provider: str, explicit_key: str | None) -> str:
    """
    Résout la clé API depuis l'argument CLI ou les variables d'environnement.

    Priorité : argument explicite > variable d'environnement.

    Args:
        provider: "openai" ou "anthropic".
        explicit_key: Clé fournie explicitement via --api-key (peut être None).

    Returns:
        La clé API résolue.

    Raises:
        ValueError: Si aucune clé n'est disponible.
    """
    if explicit_key:
        return explicit_key

    env_var = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
    env_key = os.environ.get(env_var)
    if env_key:
        logger.info("Clé API résolue depuis la variable d'environnement %s", env_var)
        return env_key

    raise ValueError(
        f"Aucune clé API trouvée pour le provider '{provider}'. "
        f"Fournissez --api-key ou définissez la variable d'environnement {env_var}."
    )


def evaluate_case(
    case: dict[str, Any],
    evaluators: dict[str, Any],
    weights: EvaluatorWeights | None = None,
    verbose: bool = False,
) -> tuple[list[EvaluationResult], float, bool]:
    """
    Évalue un cas de test avec les évaluateurs applicables.
    Retourne (résultats, score_composite, passed).

    Le score composite est normalisé par la somme des poids effectivement utilisés,
    ce qui garantit un résultat cohérent quelle que soit la combinaison d'évaluateurs.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    prompt: str = case.get("prompt", "")
    expected_output: str = case.get("expected_output", "")
    model_output: str = case.get("model_output", "")
    metadata: dict[str, Any] = case.get("metadata", {})
    applicable: list[str] = metadata.get("evaluators", ["similarity", "hallucination", "format"])

    # Pondération par évaluateur — le LLM judge utilise son propre poids fixe
    weight_map = {
        "similarity": weights.similarity,
        "hallucination": weights.hallucination,
        "format": weights.format_compliance,
        "code": weights.format_compliance,
        "consistency": weights.format_compliance,
        "llm_judge": LLM_JUDGE_WEIGHT,
    }

    results: list[EvaluationResult] = []
    total_weight = 0.0
    weighted_score = 0.0

    for ev_name in applicable:
        if ev_name not in evaluators:
            logger.warning("Évaluateur inconnu ou non configuré : '%s' — ignoré", ev_name)
            continue
        result = evaluators[ev_name].evaluate(
            prompt=prompt,
            expected_output=expected_output,
            model_output=model_output,
            metadata=metadata,
        )
        results.append(result)
        weight = weight_map.get(ev_name, 0.25)
        weighted_score += result.score * weight
        total_weight += weight

        if verbose:
            status_icon = "✓" if result.passed else "✗"
            logger.info(
                "  [%s] %s score=%.4f latency=%.1fms",
                status_icon,
                result.evaluator_name,
                result.score,
                result.latency_ms,
            )
            if result.error:
                logger.warning("    Erreur évaluateur: %s", result.error)

    composite_score = (weighted_score / total_weight) if total_weight > 0 else 0.0
    case_passed = all(result.passed for result in results if result.error is None)
    return results, round(composite_score, 4), case_passed
