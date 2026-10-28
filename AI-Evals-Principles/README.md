# Routing agents and their judges

The router maps a doctor's message to an intent. Jev answers the same failure-category
question as the original LLM judge: **Did the router fail to interpret the indirect or
underspecified scheduling request?** TRUE means this specific failure is established.
FALSE can still contain another kind of routing error.

## Run the comparison

```bash
cd AI-Evals-Principles
uv sync --locked
cd routing_agent_evals
```

Open `jev_vs_llm_judge.ipynb` with the project kernel and run the cells in order.
The notebook also works in Colab: its first cell clones `codex/jev-routing-judge`
and installs the small set of dependencies. After merge, change that branch to `main`.

`llm_as_judge.ipynb` preserves the original GPT-5/GPT-6-Astra lesson from commit
`21d0f8c2339ecf64420fb0fffd08be34e60a8b0b`. The new comparison uses an independent
GPT-4.1-mini run as its LLM-only baseline. `jev_as_judge.ipynb` runs Jev alone.

Supply credentials through environment variables, an ignored `.env` in
`routing_agent_evals/`, Colab Secrets, or the notebook's hidden input:

```text
TYPESAFE_API_KEY=...
OPENAI_API_KEY=...
```

Jev uses TypeSafe directly. The LLM uses OpenAI directly. To use an existing OpenRouter
key for the LLM, set `LLM_PROVIDER=openrouter` and supply `OPENROUTER_API_KEY` instead.
Credentials are never included in notebook code, outputs, or saved request files.

## What runs

1. Load the original **21 calibration cases (9 TRUE, 12 FALSE)** and exclude the six
   worked examples shown in the prompt.
2. Judge those cases with Jev Choice using the original failure rubric and examples.
3. Keep Jev when its answer is valid and confidence is at least **0.7**; otherwise
   call GPT-4.1-mini with the same rubric and examples. If that answer also fails,
   retain an unresolved result.
4. Complete the independent GPT-4.1-mini baseline on every case, reusing the exact
   fallback requests already cached. The LLM never receives Jev's answer or confidence.
5. Compare agreement, TRUE precision/recall/F1, false alarms, coverage, and cost.

The flow is **Jev call → validate output → confidence gate → LLM fallback → final
verdict → logging**. Quality comparisons use the same cases resolved by both judges;
coverage and unresolved counts are reported separately. Costs include unsuccessful
attempts as well as resolved answers.

The cascade calls the LLM only for cases that fail Jev's validity or confidence gate.
Completing the independent LLM baseline then adds calls for cases Jev kept; exact-request
caching reuses the fallback results. Cascade cost counts Jev for every case plus only
the fallback LLM calls. The complete comparison runs both judges on every case, so its
cost can exceed a standalone cascade's cost. The always-FALSE baseline makes the label
imbalance visible.

The prompt and labels come from main commit
`21d0f8c2339ecf64420fb0fffd08be34e60a8b0b` (October 26, 2026). Historical results for
`Vague_Utterance_Failure` do not apply to this revised task. Jev notebook outputs have
been cleared; no fresh Jev scores are claimed until a provider run completes. The
original LLM notebook retains its historical outputs unchanged.

The supplied labels are provisional. Inspect the request, prediction, explanation, and
reference together when judges disagree. The notebooks reuse the original development
and validation assignments. Set `RUN_VALIDATION = True` only after freezing the prompts
to evaluate the **150 validation cases**. If you tune on those results, use a new holdout
for a later unbiased assessment. Six prompt examples are excluded from scoring.

## Change the prompt

Edit `routing_agent_evals/prompts/llm_as_judge_prompt.py`, then restart the notebook
kernel and run all cells. The notebook imports both prompts explicitly:

- `JUDGE_RULES`: the latest shared routing policy and failure definition.
- `FEW_SHOT_EXAMPLES`: six worked examples, with their IDs excluded from scoring.
- `JEV_JUDGE_PROMPT` and `CHOICE_CRITERIA`: Jev's TRUE/FALSE Choice request.
- `INDIRECT_OR_UNDERSPECIFIED_FAILURE_PROMPT`: the LLM's Boolean label and explanation.

Only the request and predicted intent enter a case's API input. Human labels and
review notes are joined afterward. Prompt edits change the cache key; old responses
are never reused for a different request.

## Save and replay

The comparison defaults to `JUDGE_MODE=auto`: replay a completed local run or start a
live run on a fresh checkout. For changed prompts, select `JUDGE_MODE=live` to make
the new calls. Responses and reports are saved under the ignored
`.runs/indirect-judge/calibration/` or `.runs/indirect-judge/validation/` directory.
Use `JUDGE_RUN_DIR` for a separate experiment; use `JUDGE_EVIDENCE_DIR` to replay a
saved run from elsewhere. Replay checks the dataset, prompts, settings, and metrics.
Authentication or credit errors stop execution; other failed responses remain unresolved.

The Jev-only notebook uses `JEV_MODE` and `JEV_RUN_DIR`, with default directories
`.runs/indirect-jev/calibration/` and `.runs/indirect-jev/validation/`.

Reported API costs are preferred. If no charge is returned, the helper uses token usage
and dated prices recorded in `judge_lab.py`. Missing usage or unknown prices stay unknown.
The direct models are `jev-1.13.0` and `gpt-4.1-mini-2025-04-14`.

## Files

- `routing_agent_evals.ipynb`: the latest main-branch router lesson.
- `llm_as_judge.ipynb`: the unchanged original GPT-5/GPT-6-Astra judge lesson.
- `jev_vs_llm_judge.ipynb`: Jev, an independent GPT-4.1-mini baseline, and escalation.
- `jev_as_judge.ipynb`: Jev only, using the same failure category and splits.
- `prompts/llm_as_judge_prompt.py`: all judge instructions and examples.
- `judge_lab.py`: API requests, response validation, caching, and metrics.
- `notebook_utils.py`: credentials, report saving, and replay checks.
- `data/README.md`: dataset provenance.

`run_notebooks.py` executes the Jev-only and comparison notebooks in fresh kernels
and saves executed copies under `.runs/executed/`. Use `--jev-only` to skip the LLM
comparison or `--include-router` to also run the router lesson. The original LLM
lesson remains a separate notebook.

From `routing_agent_evals/`, run `python -m unittest discover -s tests` for the simulated
checks. They validate behavior without provider calls; they are not a fresh benchmark.
