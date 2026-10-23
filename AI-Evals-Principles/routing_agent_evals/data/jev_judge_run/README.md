# Recorded judge evidence

`cache/jev-*.json` contains 300 genuine responses from the direct TypeSafe API using
the pinned `jev-1.13.0` model. `cache/llm-*.json` contains 300 genuine direct OpenAI
responses using `gpt-4.1-mini-2025-04-14`. Each entry records the endpoint, exact request, response,
elapsed time and UTC recording time. No authorization headers or credentials are saved.
The source queries already exist in the repository's public teaching dataset.

`jev_manifest.json` records the dataset/policy hashes, fixed calibration and held-out
IDs, Jev agreement metrics, provider/model configuration and frozen pricing source.
`manifest.json` additionally freezes the complete experiment's model/provider configuration,
calibration threshold (0.70), threshold sweep, held-out metrics and confidence bands.
Both providers returned token counts without USD charges. Cost fields therefore distinguish
the list-price estimate from any provider-reported charge. This is not an invoice.

Replay all three notebooks with no API credentials:

```bash
uv run python routing_agent_evals/run_notebooks.py
```

Run that command from `AI-Evals-Principles` after `uv sync --locked`.

The live gate requested the LLM for 39 of 200 held-out traces; the standalone LLM benchmark
then filled in the remaining judgments. A single held-out LLM smoke request preceded
calibration, as recorded in the manifest; no held-out label selected the gate. Jev was
already captured and reused without further charges. Total recorded model requests: 600.

Cost estimates use the observed cache counts and can change with production cache hit
rates, provider prices or account terms. No synthetic judge response is published here.
Integration-test fixtures live only in temporary test directories and are explicitly
labeled synthetic.
