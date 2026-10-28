# Reference data

The judge notebooks use **llm_as_judge_indirect_underspecified.csv** from main commit
`21d0f8c2339ecf64420fb0fffd08be34e60a8b0b` (October 26, 2026).

- 300 routing traces, with 23 TRUE and 277 FALSE reference labels.
- The label is `Indirect_or_Underspecified_Failure`. FALSE does not guarantee the
  routing prediction is correct; it can contain a different kind of error.
- Six worked examples in the prompt are excluded from scoring. The default
  calibration set contains **21 cases: 9 TRUE and 12 FALSE**.
- Labels are provisional. Some capability and ambiguity boundaries still need
  human review. No label is changed to improve a model's score.
- Models receive only `USER_INPUT` and `final_intent`. Targets, comments, categories,
  and reference labels remain outside model inputs.

`judge_indirect_split.csv` preserves the original 150 development / 150 validation
assignments, paraphrase groups, calibration membership, and prompt-example membership.
The notebooks use its calibration cases by default, excluding prompt examples.
Set `RUN_VALIDATION = True` only after freezing the prompts to run the **150 validation
cases**. Do not use validation results to tune prompts and then report them as untouched
holdout performance. The original `llm_as_judge.ipynb` lesson is preserved separately;
the Jev-only and Jev/LLM comparison notebooks reuse its rubric, examples, and split.

Jev asks: **Did the router fail to interpret the indirect or underspecified scheduling
request?** TRUE and FALSE refer to that selected failure category, not overall routing
quality. Comparison metrics use paired resolved cases, with coverage and unresolved
counts reported separately. Cleared notebook outputs and simulated checks do not
constitute a fresh provider benchmark.

`llm_as_judge_dataset.csv` is the older Vague Utterance task. Its labels and recorded
scores must not be mixed with the current definition. It remains available for
historical reference, but the current judge loader does not use it.

The original workshop source is
[AI Evals - Workshop](https://docs.google.com/spreadsheets/d/1DLDctkBD3dFXenVdxHPDoMsvxauGJsnFB-1yKFfbQsE/edit).
The October 23 snapshots have 301 rows in `golden_dataset.csv` and
`golden_dataset_with_predictions.csv`, and 300 in `llm_as_judge_dataset.csv`.
The revised failure-category labels above come from the October 26 repository update.
Notebook runs do not rewrite these datasets.
