# Judge data

A doctor's scheduling assistant maps each message to an action. These labeled traces
let us check whether it missed the meaning of an indirect or underspecified request.

`llm_as_judge_indirect_underspecified.csv` contains **300 routing traces: 23 TRUE and
277 FALSE**. The label column is `Indirect_or_Underspecified_Failure`; the dataset comes
from repository commit `21d0f8c2339ecf64420fb0fffd08be34e60a8b0b` (October 26, 2026).

The judges ask: **Did the router fail to interpret the indirect or underspecified
scheduling request?** TRUE marks this specific failure. FALSE can mean a correct
prediction, defensible ambiguity, or a different routing error. Each notebook gives
the full policy and six worked examples before making any calls.

## Practice and evaluation cases

`judge_indirect_split.csv` records development/evaluation assignments, paraphrase groups,
practice membership, and prompt-example membership. It has 150 development cases and
150 separate evaluation cases, with no paraphrase group shared between them.

- **21 practice cases (9 TRUE, 12 FALSE)** support the confidence-threshold comparison.
  They are a subset of development data and exclude the six worked prompt examples.
- **150 evaluation cases** measure the final methods. The comparison notebook runs
  these automatically after the practice calls.
- The six worked prompt examples are excluded from both scoring sets.

The API helper calls the practice set `calibration` and the evaluation set `validation`.
The Jev-only notebook runs the 21 practice cases. The comparison notebook runs both sets
with live API requests and a confidence threshold of **0.70**. The practice table
compares alternative thresholds using the CMU calibration approach.

Choose the policy and threshold using practice results, then inspect the separate
evaluation results. If those results inform further changes, use new examples for a
later independent assessment.

## Inputs and scoring

Models receive only `USER_INPUT` and `final_intent`. Human labels, target intents,
reviewer comments, and other failure categories stay outside the API inputs.

Comparison quality uses the same cases successfully judged by all methods. Coverage
and unresolved counts are reported separately, and failed calls are never relabeled
FALSE. Labels are provisional: inspect capability and ambiguity boundaries when a
judge disagrees. Notebook runs do not change the source labels or datasets.

`llm_as_judge_dataset.csv` contains the older `Vague_Utterance_Failure` task. Its labels
and recorded scores must not be mixed with the current definition; the current loader
does not use it.

The workshop source is
[AI Evals - Workshop](https://docs.google.com/spreadsheets/d/1DLDctkBD3dFXenVdxHPDoMsvxauGJsnFB-1yKFfbQsE/edit).
The October 23 snapshots contain 301 rows in `golden_dataset.csv` and
`golden_dataset_with_predictions.csv`, and 300 rows in `llm_as_judge_dataset.csv`.
The revised failure-category labels above come from the October 26 repository update.
