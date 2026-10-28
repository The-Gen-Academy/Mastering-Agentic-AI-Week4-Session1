# Jev as a Judge

A doctor's scheduling assistant turns messages such as “I can work late on Friday”
into actions. This lab checks whether the assistant understood the request, using
Jev, GPT-5, and Jev with GPT-5 fallback.

## What the judges check

**Failure category:** Failure to interpret indirect or underspecified requests.

**Binary question:** Did the router fail to interpret the indirect or underspecified
scheduling request?

The scheduling policy supports viewing appointments, canceling or moving all appointments
in a date/time range, blocking or restoring availability, and changing working hours.
It also recognizes greetings, closings, unsupported requests, and irrelevant queries.
Creating an appointment or changing a specific appointment identified by patient name
or relative position is unsupported. Missing dates, times, or destinations do not make
an otherwise recognizable action unsupported. A prediction is acceptable if it has a
reasonable interpretation under the policy; judges must not invent calendar state.

Return **TRUE** only when an indirect, abbreviated, misspelled, or incomplete request
has recoverable scheduling meaning, and the predicted intent misses that meaning under
every defensible reading. Return **FALSE** when this failure is not established,
including correct routes, reasonable ambiguity, and other error categories. FALSE does
not mean the route is correct overall.

Each notebook includes the complete intent policy and six worked examples. Both judges
receive the same rules and examples. The case input contains only the doctor's message
and predicted intent; human labels are used afterward to calculate results.

## Run the lab

```bash
cd AI-Evals-Principles
uv sync --locked
cd routing_agent_evals
```

Open `jev_vs_llm_judge.ipynb` with the project kernel and run the cells in order.
For Jev alone on 21 practice cases, open `jev_as_judge.ipynb`.
The notebooks also support Colab: the setup cell clones `codex/jev-routing-judge`
and installs its dependencies. After the branch is merged, use `main` in that cell.

Supply `TYPESAFE_API_KEY` and `OPENAI_API_KEY` through environment variables, Colab
Secrets, an ignored `.env` in `routing_agent_evals/`, or the notebook's hidden input.
The Jev-only notebook needs only `TYPESAFE_API_KEY`. Keys are not saved in request logs.

The default models are TypeSafe `jev-1.13.0` and OpenAI `gpt-5-2025-08-07`, using the
providers' direct APIs. GPT-5 uses medium reasoning effort and returns a structured
Boolean verdict with a short explanation.

## What runs

1. Call Jev and GPT-5 on **21 practice cases** and display the error/cost tradeoff
   across candidate confidence thresholds.
2. Call Jev on **150 separate evaluation cases**.
3. Keep valid Jev answers at confidence **0.70 or above**. For lower confidence or
   invalid output, make a fresh GPT-5 fallback call with the same evidence and policy.
   If fallback fails, leave the final verdict unresolved.
4. Run a separate GPT-5 baseline on all 150 evaluation cases, including cases that
   already received fallback. It never sees Jev's answer or confidence.
5. Compare quality on the same cases successfully judged by every method, alongside
   coverage, unresolved counts, API cost, and an always-FALSE baseline.

Both practice and evaluation run automatically. The practice cases let you inspect
thresholds before measuring quality on different examples. The six prompt examples
are excluded from scoring. The supplied labels are provisional, so inspect disagreements
before interpreting them as model errors.

The flow is **Jev call → validate output → confidence gate → GPT-5 fallback → final
verdict → logging**. Every judge invocation sends a fresh API request. Re-running an
inference cell incurs new API usage; no local response cache or replay is used.

## Choose the confidence threshold

`THRESHOLD = 0.70` is a starting setting, not an automatically learned cutoff.
For binary Choice, native confidence is `2 × max(P(TRUE), P(FALSE)) − 1`, so 0.70
corresponds to a winning-label probability of 0.85. It does not mean 70% measured
accuracy. [TypeSafe explains the confidence calculation](https://docs.typesafe.ai/confidence).

The [CMU Jev-as-a-Judge study](https://arxiv.org/html/2609.26550v5) chooses a threshold
on local labeled examples, checks its accuracy loss against the fallback conservatively,
and evaluates it on separate cases. Its threshold uses winning probability, so it must
be converted to the native-confidence scale used here. The notebook explains that
procedure and displays a practice threshold table. Choose the setting before examining
evaluation results; the table does not change `THRESHOLD` for you.

## Inspect the run

Each execution creates a timestamped directory under `.runs/live-judge/`. Set
`JUDGE_RUN_DIR` to choose another parent directory. Individual request/response logs,
call IDs, comparison tables, decision routes, errors, and the run configuration are
saved for inspection. Saved logs are never read back to answer new requests.

Quality metrics use the common resolved cases. Coverage and unresolved counts describe
all evaluation cases. Cascade cost includes Jev calls and actual fallback calls; baseline
cost includes its separate GPT-5 calls. The complete notebook also pays for practice calls.
Costs include unsuccessful attempts, prefer reported charges, and otherwise use dated
token-price estimates. Missing usage or unknown prices remain unknown.

Edit `routing_agent_evals/prompts/llm_as_judge_prompt.py` to change the shared policy or
examples, then restart the kernel and run again. `judge_lab.py` implements requests,
validation, escalation, threshold diagnostics, and metrics. `notebook_utils.py` handles
hidden key entry and report saving. [Dataset details](routing_agent_evals/data/README.md)
describe the labels and split.

`run_notebooks.py` executes the Jev-only and comparison notebooks in fresh kernels and
saves executed copies under `.runs/executed/`. Use `--jev-only` to run only Jev, or
`--include-router` to also run the router generation lesson.

From `routing_agent_evals/`, run `python -m unittest discover -s tests` for the 30 unit
tests. These checks use simulated responses and make no provider calls.
