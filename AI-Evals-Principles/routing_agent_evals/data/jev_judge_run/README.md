# Recorded judge evidence

`cache/jev-*.json` contains 300 genuine responses from the direct TypeSafe API using
the pinned `jev-1.13.0` model. Each entry records the endpoint, exact request, response,
elapsed time and UTC recording time. No authorization headers or credentials are saved.
The source queries already exist in the repository's public teaching dataset.

`jev_manifest.json` records the dataset/policy hashes, fixed calibration and held-out
IDs, Jev agreement metrics, provider/model configuration and frozen pricing source.
TypeSafe returned token counts without USD charges. Cost fields therefore distinguish
the list-price estimate from any provider-reported charge. This is not an invoice.

Replay the Jev-only notebook with no API credentials:

```bash
uv run python routing_agent_evals/run_notebooks.py --jev-only
```

Run that command from `AI-Evals-Principles` after `uv sync --locked`.

The independent LLM and confidence-gated comparison are still pending a valid funded
LLM credential. There is intentionally no complete `manifest.json` yet. No synthetic
judge response is published here. Integration-test fixtures live only in temporary
test directories and are explicitly labeled synthetic.
