# Reference data

Source: **[AI Evals - Workshop](https://docs.google.com/spreadsheets/d/1DLDctkBD3dFXenVdxHPDoMsvxauGJsnFB-1yKFfbQsE/edit)**.
Snapshots read on **2026-10-23** through the Google Sheets API. Row order and cell values
are preserved; empty cells become empty CSV fields. These are source datasets, not model
response caches. Re-running notebooks does not modify them.

| File | Source tab | Data rows | Reference role |
| --- | --- | ---: | --- |
| `golden_dataset.csv` | [Golden Dataset Full](https://docs.google.com/spreadsheets/d/1DLDctkBD3dFXenVdxHPDoMsvxauGJsnFB-1yKFfbQsE/edit#gid=544654879) | 301 | `USER_INPUT` and `TARGET_INTENT` for routing |
| `llm_as_judge_dataset.csv` | [LLM as a Judge Dataset](https://docs.google.com/spreadsheets/d/1DLDctkBD3dFXenVdxHPDoMsvxauGJsnFB-1yKFfbQsE/edit#gid=556727646) | 300 | `USER_INPUT`, `final_intent`, and `Vague_Utterance_Failure` for judges |
| `golden_dataset_with_predictions.csv` | [Golden Dataset with predictions](https://docs.google.com/spreadsheets/d/1DLDctkBD3dFXenVdxHPDoMsvxauGJsnFB-1yKFfbQsE/edit#gid=305434486) | 301 | Saved routing outputs for replay |

The judge loader reads only the judge CSV. Its explicit Boolean human labels are kept
as supplied; `intent_match`, target intents, comments and categories never supply the
judge answer. Compared with the previous repository CSV, 248 rows have corrected
`final_reasoning`, `intent_match` and comment columns. All 300 judge IDs, queries,
predicted intents and human labels are unchanged, preserving the completed benchmark.

Routing replay uses Golden Dataset Full targets. The prediction tab has 13 different
targets, which are displayed before recomputing matches. `TEST_227` contains “Delete my
appointment with Dr. Lee.” in Golden Dataset Full and “Delete my appointment with Lee.”
in the prediction/judge tabs. Router replay excludes that one trace from its 301-row
reference and evaluates the other 300; the judge experiment keeps its own tab's trace
and human label. No label or prediction is silently transferred to a different query.

`judge_alignment_1.csv`, `judge_alignment_2.csv`, and `golden_dataset_small_5.csv` are
legacy files retained from the original repository. They are not used by the three
evaluation notebooks.
