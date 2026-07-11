# AegisVault Stage 2 Evaluation

Stage 2 systematically evaluates AegisVault using Gemini as the external protected AI application.

Gemini is not the evaluator. AegisVault continues to use the existing Ollama-based `ScopeEvaluator` for Request Gate and Response Gate decisions. Gemini represents the application that AegisVault protects.

## Objective

The evaluation measures whether AegisVault:

- Allows valid in-domain requests.
- Blocks out-of-domain requests before Gemini is called.
- Handles ambiguous and mixed-domain requests.
- Detects generated responses that drift outside the policy.
- Adds acceptable latency compared with calling Gemini directly.
- Produces reproducible metrics and saved reports.

## Flow

Protected flow:

```text
Dataset prompt
  -> AegisVault Request Gate using Ollama
  -> Gemini external application, only if allowed
  -> AegisVault Response Gate using Ollama
  -> Final output
  -> Saved result and metrics
```

Baseline flow:

```text
Dataset prompt
  -> Gemini directly without AegisVault
  -> Baseline output and latency
```

## Datasets

Datasets are JSON Lines files under:

```text
evaluation/datasets/
```

Each domain has:

```text
requests.jsonl
responses.jsonl
```

Supported domains:

- `ecommerce`
- `hr_assistant`
- `email_assistant`

The expected labels are manually written in the dataset and are treated as the primary ground truth. Gemini is never used as the ground-truth judge.

## Policies

Policies are YAML files under:

```text
evaluation/policies/
```

All domain rules live inside YAML policies. The evaluation engine should not hardcode ecommerce, HR, or email domain rules.

## Evaluation Modes

`request`

Runs labelled request cases directly through the Request Gate. Gemini is not called.

`response`

Runs labelled response cases directly through the Response Gate. Gemini is not called.

`end-to-end`

Runs request cases through Request Gate -> Gemini -> Response Gate. Also calls Gemini directly for baseline latency and baseline output.

The direct Gemini baseline is intentionally called for every measured end-to-end request case, including cases that AegisVault later blocks in the protected flow. This keeps baseline latency comparable across the same prompt set. Protected-flow Gemini calls are tracked separately, so call-avoidance metrics only count Gemini calls skipped by AegisVault in the protected path.

## Metrics

Request metrics include:

- request accuracy
- in-domain acceptance rate
- out-of-domain detection rate
- false-positive count and rate
- false-negative count and rate
- clarify count and rate
- deterministic, Ollama, and fallback decision counts
- average confidence
- average, median, and p95 latency

Request false positive:

```text
A legitimate in-domain request was blocked or unnecessarily rejected.
```

Request false negative:

```text
An out-of-domain request was allowed.
```

Ambiguous cases are tracked separately and are not forced into binary false-positive or false-negative rates.

Response metrics include:

- response accuracy
- valid response acceptance rate
- out-of-domain response detection rate
- response false-positive and false-negative rates
- replacement rate
- average confidence
- average, median, and p95 latency

End-to-end metrics include:

- requests blocked before Gemini
- requests allowed to reach Gemini
- percentage of Gemini calls avoided
- responses blocked or replaced after Gemini
- legitimate task completion rate
- application-called accuracy
- termination source counts
- final matched rate
- direct Gemini latency
- protected-flow latency
- AegisVault absolute and percentage overhead
- API error count
- Ollama error count
- skipped case count

Metrics are reported overall, per domain, per category, per gate, and per mode.

## Latency Methodology

Latency is measured on the current machine, network, model configuration, and dataset. Results are measurements, not statistically conclusive benchmarks.

Defaults:

```text
warmup_runs: 1
measured_runs_per_case: 3
```

Overhead is calculated as:

```text
protected total latency - direct Gemini baseline latency
```

Percentage overhead is:

```text
(overhead_ms / baseline_latency_ms) * 100
```

Zero or missing baseline latency is handled safely.

## Result Folder

Each run creates a unique folder:

```text
evaluation/results/<run_id>/
```

Example contents:

```text
run_metadata.json
case_results.jsonl
request_gate_results.jsonl
response_gate_results.jsonl
baseline_results.jsonl
failures.jsonl
metrics.json
latency_summary.json
evaluation_summary.md
```

The JSONL and JSON files are the source of truth. The markdown report is for human review.

## Validate Environment

```bash
python evaluation/scripts/validate_environment.py \
  --domains ecommerce hr_assistant email_assistant
```

This checks:

- `GEMINI_API_KEY` exists
- Gemini can be reached
- the configured Gemini model responds
- Ollama is running
- the Ollama evaluator model exists
- selected policies and datasets load

## Smoke Test

Run a small 2-3 case per domain check:

```bash
python evaluation/scripts/run_evaluation.py \
  --domains ecommerce hr_assistant email_assistant \
  --modes end-to-end \
  --limit 3 \
  --runs 1 \
  --warmup-runs 0
```

The progress display shows current domain, mode, completed cases, pass/fail/skipped counts, and progress.
Terminal output uses evaluation terminology: `matched`, `mismatched`, and `skipped`.

## Full Evaluation

```bash
python evaluation/scripts/run_evaluation.py \
  --domains ecommerce hr_assistant email_assistant \
  --modes request response end-to-end \
  --runs 3 \
  --warmup-runs 1
```

## Resume

If a run is interrupted, resume with:

```bash
python evaluation/scripts/run_evaluation.py \
  --domains ecommerce hr_assistant email_assistant \
  --modes request response end-to-end \
  --runs 3 \
  --warmup-runs 1 \
  --resume evaluation/results/<run_id>
```

Completed case/run combinations are skipped.

## Regenerate Reports

```bash
python evaluation/scripts/generate_report.py evaluation/results/<run_id>
```

## Limitations

- The datasets are artificial and manually labelled.
- Gemini responses can vary across time, model versions, and network conditions.
- Ollama evaluator confidence is model-reported and not statistically calibrated.
- These results should not be described as production-grade effectiveness claims.
- Stage 2 does not add Redis, Goal Vault, Sentinel Monitor, Action Gate, embeddings, cosine similarity, LangChain, UI, API servers, deployment, or PyPI publishing.
