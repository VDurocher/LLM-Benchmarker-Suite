"""
Package evaluators — modules spécialisés d'évaluation LLM.

Chaque évaluateur implémente l'interface BaseEvaluator et retourne
un EvaluationResult standardisé pour faciliter l'agrégation des scores.
"""

from evaluators.base_evaluator import BaseEvaluator, EvaluationResult
from evaluators.code_evaluator import CodeEvaluator
from evaluators.consistency_evaluator import ConsistencyEvaluator
from evaluators.format_evaluator import FormatEvaluator
from evaluators.hallucination_evaluator import HallucinationEvaluator
from evaluators.llm_judge_evaluator import LLMJudgeEvaluator
from evaluators.similarity_evaluator import SimilarityEvaluator

__all__ = [
    "BaseEvaluator",
    "EvaluationResult",
    "CodeEvaluator",
    "ConsistencyEvaluator",
    "FormatEvaluator",
    "HallucinationEvaluator",
    "LLMJudgeEvaluator",
    "SimilarityEvaluator",
]
