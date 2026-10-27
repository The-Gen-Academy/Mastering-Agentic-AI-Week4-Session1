# Reference data

The judge notebooks use **llm_as_judge_indirect_underspecified.csv** from main commit
`21d0f8c2339ecf64420fb0fffd08be34e60a8b0b` (October 26, 2026).

- 300 routing traces, with 23 TRUE and 277 FALSE reference labels.
- The label is `Indirect_or_Underspecified_Failure`. FALSE does not guarantee the
  routing prediction is correct; it can contain a different kind of error.
- Six worked examples in the prompt are excluded automatically, leaving
  **294 evaluation cases: 20 TRUE and 274 FALSE**.
- Labels are provisional. Some capability and ambiguity boundaries still need
  human review. No label is changed to improve a model's score.
- Models receive only `USER_INPUT` and `final_intent`. Targets, comments, categories,
  and reference labels remain outside model inputs.

`judge_indirect_split.csv` is retained from the latest main branch for provenance.
The simplified comparison uses all non-demonstration cases; it does not create or
require train/dev/test groups. It is a practice comparison, not an untouched holdout.

`llm_as_judge_dataset.csv` is the older Vague Utterance task. Its labels and recorded
scores must not be mixed with the current definition. It remains available for
historical reference, but the current judge loader does not use it.

The original workshop source is
[AI Evals - Workshop](https://docs.google.com/spreadsheets/d/1DLDctkBD3dFXenVdxHPDoMsvxauGJsnFB-1yKFfbQsE/edit).
The October 23 snapshots have 301 rows in `golden_dataset.csv` and
`golden_dataset_with_predictions.csv`, and 300 in `llm_as_judge_dataset.csv`.
The revised failure-category labels above come from the October 26 repository update.
Notebook runs do not rewrite these datasets.
