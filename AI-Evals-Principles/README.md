# Evaluate a routing agent — and its judges

The example is a doctor's scheduling assistant: a router maps messages to one of ten
intents. Notebook 1 evaluates those predictions. Notebook 2 asks a separate question:
**did the router misclassify this message specifically because the utterance was vague?**
It compares Jev alone, GPT-4.1-mini alone, and Jev with low-confidence cases sent to the LLM.

## Run locally

From the repository root:

```bash
cd AI-Evals-Principles
uv sync --locked
cd routing_agent_evals
```

Select the project's Python kernel and run these notebooks in order:

1. `routing_agent_evals.ipynb` — define the intents and router, evaluate saved or fresh predictions.
2. `llm_as_judge.ipynb` — System 1 motivation, Jev Choice judgments, independent LLM judgments,
   calibration, confidence-based escalation, held-out accuracy and costs, and error analysis.

**Live benchmark pending:** the existing OpenRouter evaluation credential returned HTTP
402 (insufficient credits). No new model accuracy/cost results are published yet. Both
notebooks passed full offline integration tests using temporary synthetic transport
fixtures; those fixtures are not benchmark evidence.

The router defaults to **replay**, using existing CSV predictions. The judge defaults to
replay only when a recorded manifest exists, otherwise **live**. Replay reads responses
from `data/jev_judge_run/cache` or `JUDGE_EVIDENCE_DIR`; its assertions check the
request, dataset, split, threshold and metrics against the recorded manifest. Replay makes
no model API calls. Results go to ignored `.runs/`, preserving the original datasets.

For **live** mode, set the notebook's mode selector or export `ROUTER_MODE=live` /
`JUDGE_MODE=live`. The router needs `OPENAI_API_KEY`; both judges use `OPENROUTER_API_KEY`.
Use environment variables, an ignored `.env` inside `routing_agent_evals`, Colab Secrets,
or the hidden prompt. Never put credentials in notebook cells. HTTP 401/402/403 stops the
run so keys, account credits or access can be fixed. Completed requests are cached by
endpoint and exact request hash; use a new `JUDGE_RUN_DIR` for an independent rerun. A completed live notebook writes a manifest alongside its cache;
set `JUDGE_MODE=replay` and `JUDGE_EVIDENCE_DIR` to that directory to replay it.

The comparison uses `typesafe/jev-1.13` and `openai/gpt-4.1-mini`, following the supplied
[clinic notebook](https://colab.research.google.com/drive/1FtSQhtAx8rlkWlV7GA87kIcaHwHG_mL_).
These identifiers are frozen for reproducibility, not claims about the newest models.

## Run in Colab

Open either notebook from GitHub. Its setup cell clones the teaching branch if a checkout
is not already available and installs the notebook dependencies. `REPO_REF` defaults to
`codex/jev-routing-judge` so the pull request is runnable before merge. Set
`ROUTING_EVALS_REF=main` after the change is merged, or use a specific commit for a frozen
teaching session. Use a CPU runtime; no Docker or GPU is needed.

## What the judge lab measures

- **Task:** binary agreement with the existing `Vague_Utterance_Failure` human label.
  `FALSE` includes correct routing and misclassifications caused by other failure modes.
  This is not overall intent accuracy or a new classification of the message.
- **Inputs:** both judges see the same query, candidate intent and policy. Human targets,
  comments, categories and Boolean labels are excluded. The LLM never sees Jev's answer.
- **Jev primitive:** `choice`, not `score`. The returned `TRUE`/`FALSE` label is used directly.
  Confidence and the winning-option probability are distinct quantities.
- **Split:** deterministic, label-stratified, query-grouped calibration/held-out sets.
  Only calibration labels select the threshold. The objective is TRUE-class F1, then
  accuracy, then fewer LLM calls, with lower threshold as the final tie-breaker.
- **Gate:** confidence at or above the threshold keeps Jev; otherwise the LLM judges the
  same trace independently. Invalid Jev outputs also escalate. Failed LLM outputs remain
  unresolved and count against accuracy; they are never silently accepted.
- **Metrics:** accuracy, TRUE precision/recall/F1, confusion counts, unresolved cases,
  escalation rate and per-request API cost. An always-FALSE baseline exposes imbalance.
- **Costs:** reported OpenRouter `usage.cost` in USD, with missing records flagged as
  unknown. Cascade cost includes Jev on every row plus only escalated LLM calls. The full
  benchmark additionally pays for calibration and the independent LLM comparison.
- **Mechanism:** compare high/low-confidence subsets and LLM fixes versus regressions.
  A cascade only improves quality when the second judge fixes more errors than it adds
  on the escalated subset. High-confidence errors and correlated mistakes can defeat it.

## Data limitations are part of the lesson

There are 300 judge rows: 20 TRUE and 280 FALSE. Always predicting FALSE produces 93.3%
full-dataset agreement while detecting no failures. The audit also reports 10 target-label
differences between the judge and golden CSVs, one query difference, and 248 legacy
`intent_match` cells containing prose. Those fields are not used as judge answers. Human
judge labels are preserved; inspect disagreements before treating them as unquestionable.
The notebook includes a sensitivity comparison excluding cross-file reference conflicts.

The router's 301 saved predictions contain two target labels that differ from the canonical
`golden_dataset.csv`; Notebook 1 matches IDs and text, then recomputes routing metrics
against that canonical file. New router outputs require fresh human judge labels before
being included in the frozen judge agreement benchmark.

The original `judge_alignment_1.csv` and `judge_alignment_2.csv` are retained as historical
baselines. They lack verified per-request model/usage metadata, so the notebook does not
invent historical costs or present them as the controlled Jev/LLM comparison.

## Tests and notebook execution

From `AI-Evals-Principles` after `uv sync --locked`:

```bash
uv run python -m pytest routing_agent_evals/tests -q
uv run python routing_agent_evals/run_notebooks.py
```

A funded live run or a recorded judge evidence directory is required for the full runner.
The runner executes both notebooks top to bottom in fresh kernels, writes executed copies
to `.runs/executed`, and fails on any cell error. Tests cover request leakage, malformed
outputs, threshold boundaries, query-grouped splits, calibration isolation, unresolved
judgments, fallback calls, cost accounting and offline replay of recorded evidence.

## Files

```text
routing_agent_evals/
  routing_agent_evals.ipynb   Router lesson, replay/live modes
  llm_as_judge.ipynb          Jev / LLM / confidence-gated judge lesson
  judge_lab.py               Shared policy, API client, validation and metrics
  run_notebooks.py           Execute both notebooks with clean kernels
  tests/                     Offline correctness and replay tests
  data/                      Original CSVs; optional recorded judge evidence
  prompts/                   Original router and judge policies
  images/                    Existing router diagrams
```

## References

- [TypeSafe introduction](https://docs.typesafe.ai/introduction)
- [Choice](https://docs.typesafe.ai/primitives/choice) and [confidence](https://docs.typesafe.ai/confidence)
- [Clinic comparison notebook used as inspiration](https://colab.research.google.com/drive/1FtSQhtAx8rlkWlV7GA87kIcaHwHG_mL_)
- [Evals workbook](https://docs.google.com/spreadsheets/d/1O4VwHehFm9ebc1-szXWSyOj7DI8tn5Jp7kLajBO5N_o/edit)
