# Routing agents and their judges

The router maps a doctor's message to an intent. The judge comparison asks a separate
question: **did the router fail to interpret an indirect or underspecified request?**
A FALSE judgment can still contain another kind of routing error.

## Run the comparison

```bash
cd AI-Evals-Principles
uv sync --locked
cd routing_agent_evals
```

Open `llm_as_judge.ipynb` with the project kernel and run the cells in order.
The notebook also works in Colab: its first cell clones `codex/jev-routing-judge`
and installs the small set of dependencies. After merge, change that branch to `main`.

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

1. Load the latest 300-row dataset, excluding the six examples shown in the prompt.
2. Judge the remaining **294 cases (20 TRUE, 274 FALSE)** with Jev Choice.
3. Judge those same cases independently with GPT-4.1-mini.
4. Keep Jev when its answer is valid and confidence is at least **0.7**; otherwise
   use the LLM answer. If that answer also fails, retain an unresolved result.
5. Compare agreement, TRUE precision/recall/F1, false alarms, and cost.

The cascade reuses the two standalone runs, so computing it makes no extra API calls.
Its cost counts Jev for every case plus only the LLM answers it uses. The complete
comparison calls both judges on every case and therefore costs more than a standalone
cascade. The always-FALSE baseline makes the label imbalance visible.

The latest prompt and labels come from main commit
`21d0f8c2339ecf64420fb0fffd08be34e60a8b0b` (October 26, 2026). Historical results for
`Vague_Utterance_Failure` do not apply to this revised task. Outputs have been cleared;
no scores for the new definition are claimed until a fresh run completes.

The supplied labels are provisional. These practice results do not establish production
accuracy. Inspect the request, prediction, explanation, and reference together when
judges disagree. Six prompt examples are excluded automatically; participants do not
need to prepare additional datasets or splits.

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

`mode="auto"` replays a completed local run or starts a live run on a fresh checkout.
For changed prompts, select `mode="live"` to make the new calls. A live run saves
responses and reports under the ignored `.runs/indirect-judge/` directory.
Use `JUDGE_RUN_DIR` for a separate experiment; use `JUDGE_EVIDENCE_DIR` to replay a
saved run from elsewhere. Replay checks the dataset, prompts, settings, and metrics.
Authentication or credit errors stop execution; other failed responses remain unresolved.

Reported API costs are preferred. If no charge is returned, the helper uses token usage
and dated prices recorded in `judge_lab.py`. Missing usage or unknown prices stay unknown.
The direct models are `jev-1.13.0` and `gpt-4.1-mini-2025-04-14`.

## Files

- `routing_agent_evals.ipynb`: the latest main-branch router lesson.
- `llm_as_judge.ipynb`: the short Jev, LLM, and escalation comparison.
- `jev_as_judge.ipynb`: Jev only, using the same current task and dataset.
- `prompts/llm_as_judge_prompt.py`: all judge instructions and examples.
- `judge_lab.py`: API requests, response validation, caching, and metrics.
- `notebook_utils.py`: credentials, report saving, and replay checks.
- `data/README.md`: dataset provenance.

The Jev-only notebook supports `JEV_MODE=replay` with `JEV_RUN_DIR` pointing at its
saved directory. `run_notebooks.py` executes notebooks in fresh kernels and saves
executed copies under `.runs/executed/`.
