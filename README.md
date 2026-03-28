# LLM-Benchmarker-Suite

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?logo=openai)](https://openai.com)
[![Anthropic](https://img.shields.io/badge/Anthropic-Claude-D97706)](https://anthropic.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **A production-grade LLM evaluation toolkit.** Run structured benchmarks on any model — offline or live — with 6 specialized evaluators including LLM-as-a-judge. Compare models side by side. Generate HTML reports. Built by someone who spent months evaluating LLM outputs professionally.

---

## Why This Exists

Shipping an LLM to production is not the same as shipping a deterministic microservice. A model that passes manual testing can still:

- Hallucinate critical facts under slight prompt variations
- Refuse to follow structured output instructions
- Contradict itself across a multi-turn session
- Produce code with security vulnerabilities
- Fail safety checks under adversarial rephrasing

**LLM-Benchmarker-Suite answers one question with data:** *"Can this model be trusted at 99% reliability in production?"*

Built by a former LLM Quality Analyst with experience evaluating model outputs at scale. The evaluation methodology reflects real production QA workflows, not academic benchmarks.

---

## Features

- **6 specialized evaluators** — TF-IDF similarity, hallucination detection, format compliance, code quality, consistency, and LLM-as-a-judge
- **LLM-as-a-judge** — uses GPT-4o-mini or Claude Haiku to evaluate qualitative dimensions (accuracy, completeness, safety, format) on a 0–10 scale
- **Live inference mode** — benchmark any OpenAI or Anthropic model in real-time (`--live`)
- **Offline mode** — evaluate pre-collected outputs without any API calls
- **Multi-model comparison** — run the same test set across multiple models and generate a ranked comparison report
- **6 test domains** — safety, logic, format, consistency, reasoning, instruction-following (28 curated cases)
- **HTML + JSON reports** — standalone visual reports with no external dependencies
- **CI/CD ready** — exit code 0 if ≥99% pass rate, exit code 1 otherwise

---

## Evaluators

| Evaluator | Method | Catches |
|-----------|--------|---------|
| `similarity_cosine` | TF-IDF cosine similarity | Semantic drift, off-topic responses |
| `hallucination_detector` | Keyword anchoring + contradiction detection | Fabricated facts, self-contradictions |
| `format_compliance` | JSON Schema, regex, length constraints | Structural failures, missing required patterns |
| `code_evaluator` | AST parsing, security pattern scan | Syntax errors, `eval()`/`exec()` injection risks |
| `consistency_evaluator` | Contradiction patterns, length ratios, sentence density | Internal inconsistency, inappropriately short/long responses |
| `llm_judge` | LLM-as-a-judge (GPT-4o-mini or Claude Haiku) | Qualitative correctness, reasoning quality, nuanced safety |

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/VDurocher/LLM-Benchmarker-Suite.git
cd LLM-Benchmarker-Suite

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Run offline benchmark (no API key needed)

```bash
# Evaluate pre-filled outputs in the test JSON files
python main.py --model gpt-4o --test-set safety --format html --verbose
```

### 3. Run live benchmark (real API calls)

```bash
export OPENAI_API_KEY=sk-...

# Benchmark GPT-4o on reasoning tasks
python main.py --model gpt-4o --test-set reasoning --live --provider openai --format both

# Benchmark Claude on instruction following
python main.py --model claude-3-5-sonnet-20241022 --test-set instruction_following \
  --live --provider anthropic --format both
```

### 4. Run with LLM-as-a-judge

```bash
# Adds qualitative evaluation on top of deterministic checks
python main.py --model gpt-4o --test-set all --live --judge \
  --provider openai --judge-model gpt-4o-mini --format both
```

### 5. Compare two models

```bash
python compare_runner.py \
  --models gpt-4o gpt-4o-mini \
  --test-set reasoning \
  --live --provider openai \
  --judge \
  --format both \
  --verbose
```

---

## Architecture

```
LLM-Benchmarker-Suite/
│
├── api/                              # Live inference clients
│   ├── base_client.py               # LLMClient abstract interface
│   ├── openai_client.py             # OpenAI Chat Completions (gpt-4o, gpt-4o-mini…)
│   └── anthropic_client.py          # Anthropic Messages (claude-3-5-sonnet, haiku…)
│
├── evaluators/                       # Evaluation modules
│   ├── base_evaluator.py            # BaseEvaluator + EvaluationResult (Template Method)
│   ├── similarity_evaluator.py      # TF-IDF cosine similarity
│   ├── hallucination_evaluator.py   # Keyword anchoring + contradiction detection
│   ├── format_evaluator.py          # JSON Schema, regex, length constraints
│   ├── code_evaluator.py            # AST parsing + security pattern scan
│   ├── consistency_evaluator.py     # Contradiction, length, sentence density
│   └── llm_judge_evaluator.py       # LLM-as-a-judge (0–10 scale, structured JSON)
│
├── utils/
│   ├── evaluation_pipeline.py       # Shared load, build, fetch, evaluate functions
│   ├── report_generator.py          # Versioned JSON report builder
│   ├── html_report.py               # Standalone HTML visual report
│   ├── html_comparison.py           # Multi-model comparison HTML
│   ├── html_primitives.py           # Shared rendering components
│   ├── stats.py                     # Evaluator statistics aggregation
│   └── logger.py                    # Structured logger
│
├── data/                             # Test cases (6 domains, 28 cases)
│   ├── test_cases_safety.json       # Refusals, PII, adversarial prompts
│   ├── test_cases_logic.json        # Arithmetic, SQL, probability, deduction
│   ├── test_cases_format.json       # JSON Schema, structured outputs
│   ├── test_cases_consistency.json  # Contradictions, length issues
│   ├── test_cases_reasoning.json    # Chain-of-thought, causal analysis, analogies
│   └── test_cases_instruction_following.json  # Format constraints, exclusions, multi-constraint
│
├── tests/                            # Unit tests (pytest)
│   ├── test_similarity_evaluator.py
│   ├── test_hallucination_evaluator.py
│   ├── test_format_evaluator.py
│   └── test_llm_judge_evaluator.py  # Mocked LLM client — no API calls in CI
│
├── reports/                          # Generated benchmark artifacts (gitignored)
├── main.py                           # CLI — single-model benchmark
├── compare_runner.py                 # CLI — multi-model comparison
└── config.py                         # Thresholds, weights, constants
```

**Evaluation flow:**

```
Test Cases (JSON)
      │
      │  [--live]  →  API Client  →  model_output filled in real-time
      │
      ▼
┌─────────────────────────────────────────────────┐
│  evaluate_case()                                 │
│  For each applicable evaluator in metadata:      │
│    similarity · hallucination · format           │
│    code · consistency · llm_judge               │
│  Weighted composite score (normalized)           │
└─────────────────────────────────────────────────┘
      │
      ▼
┌──────────────────────┐    ┌──────────────────────┐
│  benchmark_*.json    │    │  benchmark_*.html     │
│  comparison_*.json   │    │  comparison_*.html    │
└──────────────────────┘    └──────────────────────┘
      │
      ▼
Exit code 0 (≥99% pass rate) or 1 (below target)
```

---

## LLM-as-a-Judge

The `llm_judge` evaluator sends a structured judging prompt to a secondary LLM asking it to score the model output across four dimensions:

| Dimension | Max | Evaluates |
|-----------|-----|-----------|
| `accuracy` | 3 | Factual correctness vs. reference answer |
| `completeness` | 3 | Coverage of key points from the reference |
| `safety` | 2 | Absence of harmful or misleading content |
| `format` | 2 | Structural appropriateness for the prompt |
| **Total** | **10** | Normalized to 0.0–1.0 |

The judge responds in structured JSON, which is parsed and validated. If the judge returns malformed output, the case is marked as failed with the raw response logged.

```bash
# Use Claude Haiku as judge while benchmarking GPT-4o (cross-provider judging)
python main.py --model gpt-4o --test-set reasoning --live --provider openai \
  --judge --judge-provider anthropic --judge-model claude-haiku-4-5-20251001
```

---

## Test Domains

| Domain | Cases | Focus |
|--------|-------|-------|
| `safety` | 5 | Harmful content refusal, PII protection, adversarial prompt injection, disclaimers |
| `logic` | 5 | Arithmetic, conditional reasoning, SQL, algorithmic complexity, probability |
| `format` | 3+ | JSON Schema compliance, structured API responses, Python code generation |
| `consistency` | 5 | Self-contradictions, format mismatches, inappropriate response length |
| `reasoning` | 5 | Chain-of-thought math, logical deduction, causal analysis, counterfactuals, analogies |
| `instruction_following` | 5 | Bulleted lists, length constraints, exclusion constraints, multi-constraint, structured sections |

---

## CLI Reference

### `main.py` — single-model benchmark

```
python main.py --model <model_id> [options]

Core:
  --model         Model identifier (gpt-4o, claude-3-5-sonnet-20241022, llama-3...)
  --test-set      safety | logic | format | consistency | reasoning | instruction_following | all
  --output-dir    Output directory for reports (default: ./reports/)
  --format        json | html | both (default: json)
  --verbose       Print per-evaluator scores for each case

Live inference:
  --live          Enable live API calls (fills model_output from real model)
  --provider      openai | anthropic (default: openai)
  --api-key       API key (or set OPENAI_API_KEY / ANTHROPIC_API_KEY env var)

LLM-as-a-judge:
  --judge         Enable LLM-as-a-judge evaluator
  --judge-model   Judge model (default: gpt-4o-mini / claude-haiku-4-5-20251001)
  --judge-provider  Judge provider (default: same as --provider)
  --judge-api-key   Judge API key (default: same as --api-key)
```

### `compare_runner.py` — multi-model comparison

```
python compare_runner.py --models <m1> <m2> [...] [options]

  --models        Two or more model identifiers to compare
  (all other flags identical to main.py)
```

---

## Scoring

### Composite score

Each case produces a composite score — a normalized weighted average of the active evaluators:

| Evaluator | Default weight |
|-----------|----------------|
| `similarity` | 0.40 |
| `hallucination` | 0.20 |
| `format` / `code` / `consistency` | 0.10 |
| `llm_judge` | 0.30 (when active) |

Weights are normalized by the sum of active evaluators, so the composite is always in [0.0, 1.0].

### Pass/fail per case

A case **passes** if all active evaluators return `passed = True`. A single evaluator failure fails the case.

### Production readiness verdict

```
pass_rate = passed_cases / total_cases
✓ PRODUCTION READY     if pass_rate ≥ 99%
✗ BELOW TARGET         if pass_rate < 99%
```

Exit code 0 on success, 1 on failure — use directly in CI pipelines:

```bash
python main.py --model gpt-4o --test-set all --live --provider openai
if [ $? -ne 0 ]; then
  echo "Model did not meet production threshold. Blocking deployment."
  exit 1
fi
```

---

## Running Tests

```bash
pip install pytest pytest-cov
pytest tests/ -v --cov=evaluators --cov-report=term-missing
```

Expected output:

```
tests/test_similarity_evaluator.py::TestNormalizeText::test_lowercase_conversion PASSED
tests/test_similarity_evaluator.py::TestSimilarityEvaluator::test_identical_texts_score_one PASSED
tests/test_hallucination_evaluator.py::TestHallucinationEvaluator::test_matching_content_passes PASSED
tests/test_format_evaluator.py::TestFormatEvaluatorJsonValidation::test_valid_json_passes PASSED
tests/test_llm_judge_evaluator.py::TestLLMJudgeEvaluator::test_perfect_score_passes PASSED
tests/test_llm_judge_evaluator.py::TestLLMJudgeEvaluator::test_judge_called_with_correct_arguments PASSED
...
```

The LLM judge tests use a mocked client — no API calls are made during CI.

---

## Extending the Suite

### Add a new test domain

Create `data/test_cases_<domain>.json`:

```json
{
  "test_set": "domain_name",
  "description": "What this domain evaluates",
  "version": "1.0.0",
  "cases": [
    {
      "id": "domain_001",
      "category": "subcategory",
      "prompt": "The prompt sent to the model",
      "expected_output": "The reference answer",
      "model_output": "",
      "metadata": {
        "expect_valid_json": false,
        "required_patterns": ["pattern1|pattern2"],
        "forbidden_patterns": ["bad_word"],
        "evaluators": ["similarity", "hallucination", "format", "llm_judge"]
      }
    }
  ]
}
```

Register in `config.py` → `AVAILABLE_TEST_SETS` and in `utils/evaluation_pipeline.py` → `_ALL_TEST_SETS`.

### Add a new evaluator

```python
# evaluators/my_evaluator.py
from evaluators.base_evaluator import BaseEvaluator, EvaluationResult

class MyEvaluator(BaseEvaluator):
    def __init__(self, threshold: float = 0.7) -> None:
        super().__init__(name="my_evaluator", threshold=threshold)

    def _run_evaluation(self, prompt, expected_output, model_output, metadata):
        score = ...  # your scoring logic
        return EvaluationResult(
            evaluator_name=self._name,
            passed=score >= self._threshold,
            score=score,
            details={"custom_metric": ...},
        )
```

Register in `evaluators/__init__.py` and add to `build_evaluators()` in `utils/evaluation_pipeline.py`.

---

## Requirements

- Python 3.11+
- `scikit-learn>=1.4.0` — TF-IDF vectorization
- `jsonschema>=4.21.0` — JSON Schema validation
- `numpy>=1.26.0` — matrix operations
- `openai>=1.30.0` — OpenAI live inference or judge *(optional)*
- `anthropic>=0.18.0` — Anthropic live inference or judge *(optional)*

---

## License

MIT — see [LICENSE](LICENSE)
