# LLM-Benchmarker-Suite

> **A rigorous, production-grade evaluation toolkit for LLM outputs** — built for engineering teams who need to validate model reliability before deployment.

---

## Why This Tool Exists

Shipping an LLM to production is not the same as shipping a deterministic microservice. A model that scores 90% on academic benchmarks can still hallucinate critical facts, produce malformed JSON, or fail compliance checks in ways that are catastrophic in regulated environments (finance, healthcare, legal).

**LLM-Benchmarker-Suite was built for engineers who need to answer one question with data:**
*"Can this model be trusted at 99% reliability in production?"*

It is designed by and for QA engineers working in environments where model failures have real consequences — fraud undetected, compliance violated, incorrect financial data surfaced to users.

---

## Architecture

```
LLM-Benchmarker-Suite/
├── evaluators/                      # Specialized evaluation modules
│   ├── base_evaluator.py            # Abstract contract — Template Method pattern
│   ├── similarity_evaluator.py      # Cosine similarity (TF-IDF)
│   ├── hallucination_evaluator.py   # Keyword anchoring + contradiction detection
│   ├── format_evaluator.py          # JSON Schema + regex + length constraints
│   ├── code_evaluator.py            # AST parsing + security pattern scan
│   └── consistency_evaluator.py     # Cross-run consistency checks
├── data/                            # Test cases by domain
│   ├── test_cases_safety.json       # Refusals, PII protection, adversarial prompts
│   ├── test_cases_logic.json        # Reasoning, SQL, probabilities
│   ├── test_cases_format.json       # JSON Schema, Python code, constraints
│   └── test_cases_consistency.json  # Consistency across rephrased prompts
├── utils/
│   ├── logger.py                    # Structured logger
│   ├── report_generator.py          # Versioned JSON report generator
│   ├── html_report.py               # HTML report generator (per model)
│   ├── html_comparison.py           # HTML comparison report (multi-model)
│   ├── html_primitives.py           # Shared HTML rendering helpers
│   └── evaluation_pipeline.py      # Shared pipeline logic (load, evaluate, score)
├── reports/                         # Benchmark artifacts (gitignored)
├── config.py                        # Thresholds, weights, constants
├── main.py                          # CLI — single-model benchmark
├── compare_runner.py                # CLI — multi-model comparison
└── pyproject.toml                   # Project metadata and tool configuration
```

---

## Metrics Explained

### Correctness vs. Hallucination — What's the Difference?

| Dimension | What It Measures | Method |
|-----------|-----------------|--------|
| **Correctness (Similarity)** | Is the answer semantically aligned with the expected output? | TF-IDF cosine similarity on normalized text. Threshold: 0.72. |
| **Hallucination Score** | Does the model fabricate facts absent from the reference, or contradict it? | Keyword anchoring: key entities from the reference must appear in the output. Contradiction detection via negation patterns. |
| **Format Compliance** | Does the output match the required structure? | JSON Schema Draft-7 validation, regex pattern matching, length constraints. |
| **Code Quality** | Is generated code syntactically valid and free of security risks? | Python AST parsing, dangerous pattern detection (`eval`, `exec`, `os.system`), PEP8 line length. |

### Composite Scoring

Each test case receives a **weighted composite score** across applicable evaluators:

```
composite_score = (similarity × 0.40) + (hallucination × 0.20) + (format × 0.10) + (keyword_match × 0.30)
```

A test case **passes** only when **all its applicable evaluators pass individually**. The suite targets a **99% pass rate** for production clearance.

### Hallucination Detection Methodology

The hallucination evaluator uses a two-pass approach:

1. **Keyword Anchoring**: Extracts key terms (nouns, numbers, named entities) from the expected output after removing stop words. Calculates the fraction present in the model output. A coverage ratio below 0.60 signals a likely hallucination.

2. **Contradiction Detection**: Scans for explicit negation patterns (`"not X"`, `"never"`, `"incorrect"`, etc.) that may indicate the model is actively contradicting reference facts. When detected, a penalty weight (×0.85) is applied to the final score, and the case is automatically marked as failed.

*This approach is inspired by SelfCheckGPT and lightweight NLI-based fact verification pipelines used in RLHF data labeling workflows.*

---

## Installation

### Prerequisites

- Python 3.11+

### Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Dependencies** (`requirements.txt`):

| Package | Version | Purpose |
|---------|---------|---------|
| `scikit-learn` | >=1.4.0 | TF-IDF vectorization and cosine similarity |
| `jsonschema` | >=4.21.0 | JSON Schema Draft-7 validation |
| `numpy` | >=1.26.0 | Matrix computation (scikit-learn dependency) |

---

## CLI Usage

### Single-Model Benchmark (`main.py`)

```bash
# Evaluate a model against the safety test set
python main.py --model gpt-4o --test-set safety

# Run all test sets with verbose per-evaluator output
python main.py --model claude-3-5-sonnet --test-set all --verbose

# Run logic tests and save report to a custom directory
python main.py --model llama-3-70b --test-set logic --output-dir ./ci-reports

# Generate both JSON and HTML reports
python main.py --model gpt-4o --test-set all --format both
```

#### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | Model identifier (string label, not an API call) | *(required)* |
| `--test-set` | Test set: `safety`, `logic`, `format`, `consistency`, `all` | `all` |
| `--output-dir` | Directory to write reports | `./reports/` |
| `--format` | Report format: `json`, `html`, or `both` | `json` |
| `--verbose` | Show per-evaluator scores for each test case | `false` |

### Multi-Model Comparison (`compare_runner.py`)

Run the same test set across multiple models and generate a side-by-side comparison report. At least two models are required.

```bash
# Compare two models on the safety test set
python compare_runner.py --models gpt-4o claude-3-5-sonnet --test-set safety

# Compare three models across all test sets
python compare_runner.py --models gpt-4o claude-3-5-sonnet llama-3 --test-set all --verbose

# Generate an HTML comparison report
python compare_runner.py --models gpt-4o claude-3-5-sonnet --test-set format --format html
```

#### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--models` | Space-separated list of model identifiers (2 minimum) | *(required)* |
| `--test-set` | Test set: `safety`, `logic`, `format`, `consistency`, `all` | `all` |
| `--output-dir` | Directory to write reports | `./reports/` |
| `--format` | Report format: `json`, `html`, or `both` | `json` |
| `--verbose` | Show per-case status for each model | `false` |

The comparison runner determines a **winner** by pass rate, then by average composite score in case of a tie. It produces both individual per-model reports and a consolidated comparison report.

### Example Output

```
============================================================
LLM-Benchmarker-Suite — Démarrage
Modèle cible : gpt-4o
Ensemble de tests : safety
============================================================
[1/5] Évaluation du cas : safety_001
  [✓] similarity_cosine     score=0.8341  latency=12.3ms
  [✓] hallucination_detector score=0.9200 latency=2.1ms
  [✓] format_compliance      score=1.0000 latency=0.8ms
  → PASS (score composite: 0.8892)
...
============================================================
RÉSULTATS FINAUX
Cas traités : 5 | Passés : 4 | Échoués : 1
Pass rate : 80.0% (cible : 99.0%)
Verdict : ✗ BELOW TARGET — DO NOT DEPLOY
============================================================
```

### CI/CD Integration

The CLI exits with code `0` if the 99% target is met, `1` otherwise — making it directly usable as a quality gate in GitHub Actions or any CI pipeline:

```yaml
# .github/workflows/llm-quality-gate.yml
- name: Run LLM benchmark
  run: python main.py --model ${{ env.MODEL_ID }} --test-set all
  # Fails the pipeline if pass rate < 99%
```

---

## Report Format

Each benchmark run generates a timestamped JSON report in `/reports/`:

```json
{
  "report_version": "1.0.0",
  "generated_at": "2024-01-15T14:30:22Z",
  "session": {
    "model_name": "gpt-4o",
    "test_set": "safety"
  },
  "summary": {
    "total_cases": 5,
    "passed_cases": 4,
    "pass_rate": 0.8,
    "pass_rate_percent": 80.0,
    "production_ready": false,
    "production_ready_label": "✗ BELOW TARGET — DO NOT DEPLOY"
  },
  "evaluator_breakdown": {
    "similarity_cosine": {
      "pass_rate": 0.8,
      "average_score": 0.7821,
      "average_latency_ms": 11.4
    }
  },
  "test_cases": [...]
}
```

When using `--format html` or `--format both`, an HTML version of the same report is generated alongside the JSON file.

For `compare_runner.py`, an additional `comparison_<id>.json` (or `.html`) is produced with a side-by-side breakdown of all models and the identified winner.

---

## Adding Custom Test Cases

Extend any file in `/data/` following this schema:

```json
{
  "id": "custom_001",
  "category": "your_category",
  "prompt": "The prompt sent to the model.",
  "expected_output": "The reference output to evaluate against.",
  "model_output": "The actual output produced by your model.",
  "metadata": {
    "expect_valid_json": false,
    "required_patterns": ["regex_pattern_1"],
    "forbidden_patterns": ["dangerous_phrase"],
    "evaluators": ["similarity", "hallucination", "format"]
  }
}
```

---

## Extending with Custom Evaluators

Implement `BaseEvaluator` and register the evaluator in `main.py`:

```python
from evaluators.base_evaluator import BaseEvaluator, EvaluationResult

class ToxicityEvaluator(BaseEvaluator):
    def __init__(self) -> None:
        super().__init__(name="toxicity_check", threshold=0.95)

    def _run_evaluation(self, prompt, expected_output, model_output, metadata):
        # Your custom logic here
        score = run_toxicity_model(model_output)
        return EvaluationResult(
            evaluator_name=self.name,
            passed=score >= self.threshold,
            score=score,
        )
```

---

## Professional Disclaimer

This toolkit reflects applied experience in **Reinforcement Learning from Human Feedback (RLHF)** and **model alignment**. The evaluation strategies implemented here — keyword anchoring for hallucination detection, composite weighted scoring, and schema-based format validation — are informed by practices used in production ML systems at scale.

This tool does **not** replace human evaluation or red-teaming. It is designed as an **automated first-pass quality gate** to catch systematic failures before human reviewers spend time on clearly non-compliant outputs.

**LLM outputs evaluated by this tool are based on pre-recorded `model_output` fields in the test case JSON files.** The suite does not make live API calls to any LLM provider. To evaluate a real model, populate the `model_output` field with the model's actual responses before running the benchmark.

---

## License

MIT — Free to use, modify, and distribute with attribution.
