# Evaluate a routing agent — and its judges

The example is a doctor's scheduling assistant: a router maps messages to one of ten
intents. Notebook 1 evaluates those predictions. The judge notebooks ask a separate question:
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
2. `jev_as_judge.ipynb` — direct TypeSafe API, Jev Choice judgments, real recorded results and token-based cost estimates.
3. `llm_as_judge.ipynb` — System 1 motivation, Jev Choice judgments, independent LLM judgments,
   calibration, confidence-based escalation, held-out accuracy and costs, and error analysis.

**Live benchmark complete:** the saved notebook outputs show 300 valid TypeSafe Jev judgments
and 300 valid independent OpenAI judgments. Raw response caches, manifests and test files
are not included in the repository. The threshold selected on 100 calibration traces is **0.70**.
On the separate 200-trace held-out set, the cascade sends 39 traces (19.5%) to the LLM:

| Judge | Human-label agreement | Failure recall | TRUE F1 | Estimated USD / 1,000 traces |
| --- | ---: | ---: | ---: | ---: |
| Always FALSE | 93.5% | 0/13 (0%) | 0.000 | 0 |
| Jev alone | 89.5% | 0/13 (0%) | 0.000 | $0.05552 |
| GPT-4.1-mini alone | 78.0% | 8/13 (61.5%) | 0.267 | $0.51179 |
| Jev then LLM | 83.5% | 5/13 (38.5%) | 0.233 | $0.15863 |

The cascade costs an estimated **69.0% less than LLM alone** on these traces and has higher
agreement, but misses three failures the LLM finds. Compared with Jev alone, it detects
five additional failures while reducing agreement. On the 39 escalated traces, the LLM
fixes six Jev errors and introduces 18, so escalation is not a universal quality improvement.
The low-confidence subset is harder, but the LLM is not a better overall judge of that subset.

The full experiment (all 600 requests, including calibration and the standalone comparisons)
has an estimated list-price cost of **$0.16995**. Costs use returned token counts and frozen
published rates, including observed OpenAI cache counts; they are not invoices or guaranteed
production costs. There were no unresolved judgments. Prompts and threshold selection were
not tuned on held-out labels. These results do not justify deployment without further rubric
and label review.

The saved outputs can be read without running code. The router defaults to replay of its
original CSV predictions. On a fresh checkout, the judge notebooks run **live** and require
provider credentials. Optional replay uses a run directory you generated locally:
`JUDGE_EVIDENCE_DIR`, or the notebook's own `.runs/` directory. It checks the request,
dataset, split, threshold and metrics against your local manifest, without new API calls.
Generated responses and reports stay in ignored `.runs/` directories.

For **live** mode, set the notebook's mode selector or export `ROUTER_MODE=live` /
`JUDGE_MODE=live`; use `JEV_MODE=live` for the Jev-only lesson. The router needs
`OPENAI_API_KEY`. Jev uses **`TYPESAFE_API_KEY`** directly by default. The independent LLM
uses **`OPENAI_API_KEY`** directly by default; set `LLM_PROVIDER=openrouter` to use
`OPENROUTER_API_KEY` instead.
Set `JEV_PROVIDER=openrouter` in the comparison if you prefer to call both stages through
OpenRouter. Credentials are scoped to each stage and never sent to the other provider.
Use environment variables, an ignored `.env` inside `routing_agent_evals`, Colab Secrets,
or the hidden prompt. Never put credentials in notebook cells. HTTP 401/402/403 stops the
run so keys, account credits or access can be fixed. Completed requests are cached by
endpoint and exact request hash; use a new `JUDGE_RUN_DIR` (or `JEV_RUN_DIR` for Jev alone)
for an independent rerun. A completed live notebook writes a manifest alongside its cache;
set `JUDGE_MODE=replay` and `JUDGE_EVIDENCE_DIR` to that directory to replay it.

The recorded comparison uses `jev-1.13.0` directly and `gpt-4.1-mini-2025-04-14` directly,
following the model families in the supplied
[clinic notebook](https://colab.research.google.com/drive/1FtSQhtAx8rlkWlV7GA87kIcaHwHG_mL_).
These identifiers are frozen for reproducibility, not claims about the newest models.

## Run in Colab

Open a notebook from GitHub. Its setup cell clones the teaching branch if a checkout
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
- **Costs:** reported `usage.cost` is preferred. Direct TypeSafe costs are explicitly
  estimated from returned tokens at the frozen $0.042/M input-token rate (output free),
  with its source and date recorded. Separate fields retain reported, estimated and unknown
  amounts; unsupported models and missing usage remain unknown. For the pinned direct
  OpenAI model, standard-tier estimates use $0.40/M input, $0.10/M cached input and
  $1.60/M output tokens, using returned cache counts. Cascade cost includes Jev on every
  row plus only escalated LLM calls. The full
  benchmark additionally pays for calibration and the independent LLM comparison.
- **Mechanism:** compare high/low-confidence subsets and LLM fixes versus regressions.
  A cascade only improves quality when the second judge fixes more errors than it adds
  on the escalated subset. High-confidence errors and correlated mistakes can defeat it.

## Why use the confidence policy?

The policy reserves the LLM for uncertain or unusable Jev judgments. Keeping a valid,
high-confidence answer avoids a second call on cases Jev handles reliably on validation
data. Escalating low-confidence cases gives an independent judge a chance to correct
them. Invalid outputs also escalate because an unusable label cannot be accepted, even
if a confidence field looks high.

If the LLM also fails, nothing has made the original rejected answer more trustworthy.
Recording an unresolved case preserves that uncertainty for human review. In this lab,
that means an explicit unresolved result in the output table; it does not dispatch a
human-review task.

For escalation fraction `r`, average per-case cost is approximately
`Jev cost + r × LLM cost`. The policy saves money when this is below the cost of asking
the LLM on every case. Quality is a separate question: the LLM must fix more errors than
it introduces on the escalated subset to improve agreement over Jev alone.

The run demonstrates the distinction: 161 accepted Jev judgments had 95.0% agreement,
and avoiding those LLM calls reduced estimated cost by 69.0% versus LLM alone. But the
LLM fixed 6 errors and introduced 18 among the 39 escalations. The policy increases
failure recall over Jev alone while lowering agreement; it is a cost/recall tradeoff
for this dataset. Confidence thresholds must be calibrated against the product's error
costs, not treated as a guarantee of correctness.

## Reference data: the supplied workshop spreadsheet

The authoritative source is **[AI Evals - Workshop](https://docs.google.com/spreadsheets/d/1DLDctkBD3dFXenVdxHPDoMsvxauGJsnFB-1yKFfbQsE/edit)**. The three local CSVs
are snapshots of its tabs read on **2026-10-23**, so the notebooks run reproducibly without
Google credentials. See [data provenance](routing_agent_evals/data/README.md).

| Local CSV | Spreadsheet tab | Used for |
| --- | --- | --- |
| `golden_dataset.csv` | Golden Dataset Full | 301 routing queries and target intents |
| `llm_as_judge_dataset.csv` | LLM as a Judge Dataset | 300 judge traces and explicit `Vague_Utterance_Failure` labels |
| `golden_dataset_with_predictions.csv` | Golden Dataset with predictions | 301 saved routing predictions |

The two judge notebooks use only **LLM as a Judge Dataset** as their reference. All 300
queries, predicted intents and human labels match the completed API experiment, so its
results are unchanged. Refreshing from the spreadsheet corrects shifted supporting
columns in the old judge CSV; `intent_match` now contains pass/fail values. That field
answers whether the intent matched, not whether a failure was caused by vagueness.

There are 20 TRUE and 280 FALSE judge labels. Always predicting FALSE produces 93.3%
full-dataset agreement while detecting no failures. “Agreement” means matching the
supplied human label; it is not a claim that every human label is beyond dispute. Neither
judge's ground truth is inferred from another tab or from `intent_match`.

For routing, the two relevant spreadsheet tabs differ on 13 target labels and on the
query text for `TEST_227`. Notebook 1 displays those differences, uses **Golden Dataset
Full** targets, and evaluates only the 300 saved predictions whose IDs and trimmed query
text match. It explicitly excludes the mismatched trace; live mode runs all 301 golden
queries. Those routing-tab differences do not filter the judge dataset. New router
predictions require new human judge labels before joining the judge benchmark.

## Notebook execution

From `AI-Evals-Principles` after `uv sync --locked`:

```bash
uv run python routing_agent_evals/run_notebooks.py --jev-only
uv run python routing_agent_evals/run_notebooks.py
```

`--jev-only` runs just the router and Jev notebook. Fresh live runs require funded provider
credentials. To use a previously generated local run, set `JUDGE_EVIDENCE_DIR` to its
directory and select replay with `JUDGE_MODE=replay` / `JEV_MODE=replay` as applicable.
The runner executes the selected notebooks top to bottom in fresh kernels, writes executed copies
to `.runs/executed`, and fails on any cell error. All three notebooks were executed after
refreshing the spreadsheet snapshots, using a verification copy of the completed API run
outside the repository. No new API calls were needed: judge inputs and labels are unchanged.

## Files

```text
routing_agent_evals/
  routing_agent_evals.ipynb   Router lesson, replay/live modes
  jev_as_judge.ipynb         Direct TypeSafe lesson and recorded Jev-only results
  llm_as_judge.ipynb          Jev / LLM / confidence-gated judge lesson
  judge_lab.py               Shared policy, API client, validation and metrics
  run_notebooks.py           Execute selected notebooks with clean kernels
  data/                      Original CSV datasets
  prompts/                   Original router and judge policies
  images/                    Existing router diagrams
```

## References

- [TypeSafe introduction](https://docs.typesafe.ai/introduction)
- [Choice](https://docs.typesafe.ai/primitives/choice) and [confidence](https://docs.typesafe.ai/confidence)
- [TypeSafe model versions and pricing](https://docs.typesafe.ai/models)
- [OpenAI GPT-4.1-mini versions and pricing](https://developers.openai.com/api/docs/models/gpt-4.1-mini)
- [Clinic comparison notebook used as inspiration](https://colab.research.google.com/drive/1FtSQhtAx8rlkWlV7GA87kIcaHwHG_mL_)
- [Evals workbook](https://docs.google.com/spreadsheets/d/1O4VwHehFm9ebc1-szXWSyOj7DI8tn5Jp7kLajBO5N_o/edit)
