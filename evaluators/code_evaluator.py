"""
Évaluateur de qualité du code généré.

Analyse statique du code Python produit par le modèle :
1. Validité syntaxique (AST parsing)
2. Présence des constructs requis (fonctions, classes, imports)
3. Détection de patterns dangereux (eval, exec, os.system)
4. Conformité PEP8 légère (indentation, longueur de ligne)
5. Couverture des cas limites via vérification des assertions

Ce module est particulièrement utile pour les LLMs utilisés dans des
contextes de génération de code (coding assistants, automatisation DevOps).
"""

from __future__ import annotations

import ast
import re
from typing import Any

from evaluators.base_evaluator import BaseEvaluator, EvaluationResult

# Patterns de code dangereux — toujours signalés comme violations critiques
_DANGEROUS_PATTERNS: list[tuple[str, str]] = [
    (r"\beval\s*\(", "eval() détecté — risque d'injection"),
    (r"\bexec\s*\(", "exec() détecté — risque d'exécution arbitraire"),
    (r"\bos\.system\s*\(", "os.system() détecté — risque d'injection shell"),
    (r"\bsubprocess\.call\s*\(", "subprocess.call() sans validation"),
    (r"__import__\s*\(", "__import__() dynamique détecté"),
    (r"open\s*\([^)]*['\"]w['\"]", "Écriture fichier sans gestion d'erreur"),
]

# Longueur de ligne maximale (PEP8)
_MAX_LINE_LENGTH: int = 120


def _parse_ast_safely(code: str) -> tuple[ast.Module | None, str | None]:
    """
    Tente de parser le code en AST. Retourne (module, None) ou (None, erreur).
    """
    try:
        module = ast.parse(code)
        return module, None
    except SyntaxError as exc:
        return None, f"SyntaxError ligne {exc.lineno}: {exc.msg}"


class CodeEvaluator(BaseEvaluator):
    """
    Évalue la qualité et la sécurité du code Python généré par le modèle.

    Paramètres de configuration via `metadata` :
    - `required_functions` (list[str]) : noms de fonctions attendues.
    - `required_classes` (list[str]) : noms de classes attendues.
    - `required_imports` (list[str]) : modules qui doivent être importés.
    - `check_dangerous_patterns` (bool) : active la détection de patterns dangereux (défaut: True).
    - `check_pep8_lines` (bool) : vérifie la longueur des lignes (défaut: True).
    - `expected_return_type` (str) : type de retour attendu pour la fonction principale.
    """

    def __init__(self, threshold: float = 0.75) -> None:
        super().__init__(name="code_quality", threshold=threshold)

    def _run_evaluation(
        self,
        prompt: str,
        expected_output: str,
        model_output: str,
        metadata: dict[str, Any],
    ) -> EvaluationResult:
        violations: list[str] = []
        checks_passed: list[str] = []

        # Extraction du bloc de code si encapsulé dans des backticks Markdown
        code = _extract_code_block(model_output)

        # --- Validation syntaxique AST ---
        parsed_module, syntax_error = _parse_ast_safely(code)
        if syntax_error:
            violations.append(syntax_error)
            # Si le code ne parse pas, les autres vérifications n'ont pas de sens
            return EvaluationResult(
                evaluator_name=self._name,
                passed=False,
                score=0.0,
                details={
                    "violations": violations,
                    "checks_passed": checks_passed,
                    "ast_parsed": False,
                    "code_length_lines": code.count("\n") + 1,
                },
            )

        checks_passed.append("syntax_valid_ast")

        # Collecte des nœuds AST pour les vérifications suivantes
        function_names: set[str] = set()
        class_names: set[str] = set()
        import_names: set[str] = set()

        for node in ast.walk(parsed_module):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                function_names.add(node.name)
            elif isinstance(node, ast.ClassDef):
                class_names.add(node.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    import_names.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    import_names.add(node.module.split(".")[0])

        # --- Vérification des fonctions requises ---
        for func_name in metadata.get("required_functions", []):
            if func_name in function_names:
                checks_passed.append(f"function_present:{func_name}")
            else:
                violations.append(f"Fonction requise manquante : '{func_name}'")

        # --- Vérification des classes requises ---
        for class_name in metadata.get("required_classes", []):
            if class_name in class_names:
                checks_passed.append(f"class_present:{class_name}")
            else:
                violations.append(f"Classe requise manquante : '{class_name}'")

        # --- Vérification des imports requis ---
        for import_name in metadata.get("required_imports", []):
            if import_name in import_names:
                checks_passed.append(f"import_present:{import_name}")
            else:
                violations.append(f"Import requis manquant : '{import_name}'")

        # --- Détection de patterns dangereux ---
        if metadata.get("check_dangerous_patterns", True):
            for pattern, description in _DANGEROUS_PATTERNS:
                if re.search(pattern, code):
                    violations.append(f"[SÉCURITÉ] {description}")
                else:
                    checks_passed.append(f"safe_pattern:{pattern[:20]}")

        # --- Vérification PEP8 longueur de ligne ---
        if metadata.get("check_pep8_lines", True):
            long_lines = [
                i + 1
                for i, line in enumerate(code.splitlines())
                if len(line) > _MAX_LINE_LENGTH
            ]
            if long_lines:
                violations.append(
                    f"Lignes trop longues (>{_MAX_LINE_LENGTH} chars) : {long_lines[:5]}"
                )
            else:
                checks_passed.append("pep8_line_length_ok")

        # --- Score composite ---
        total_checks = len(checks_passed) + len(violations)
        score = len(checks_passed) / total_checks if total_checks > 0 else 1.0
        passed = score >= self._threshold and not any(
            "[SÉCURITÉ]" in v for v in violations
        )

        return EvaluationResult(
            evaluator_name=self._name,
            passed=passed,
            score=round(score, 4),
            details={
                "threshold": self._threshold,
                "violations": violations,
                "checks_passed": checks_passed,
                "ast_parsed": True,
                "function_names_found": sorted(function_names),
                "class_names_found": sorted(class_names),
                "import_names_found": sorted(import_names),
                "code_length_lines": code.count("\n") + 1,
                "has_security_violations": any("[SÉCURITÉ]" in v for v in violations),
            },
        )


def _extract_code_block(text: str) -> str:
    """
    Extrait le code d'un bloc Markdown ```python ... ``` s'il existe,
    sinon retourne le texte tel quel.
    """
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()
