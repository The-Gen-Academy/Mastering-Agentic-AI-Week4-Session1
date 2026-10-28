"""Credential setup and reproducibility bookkeeping for the comparison notebook.

The notebook keeps the judging, confidence gate, and metric calculations visible.
This module handles the parts participants do not need to change.
"""

import json
import os
from getpass import getpass
from pathlib import Path

from dotenv import load_dotenv

from judge_lab import (
    CRITERIA, DEFAULT_PROVIDERS, JUDGE_DATA_SOURCE,
    JudgeClient, digest, save_json,
)
from prompts.llm_as_judge_prompt import FAILURE_CATEGORY, FAILURE_QUESTION


def setup_run(lab, mode="auto", prompts=None, split="calibration"):
    """Return the client, output directory, and any manifest to verify on replay.

    Auto mode replays a completed local run, or runs live on a fresh checkout.
    JUDGE_MODE, JUDGE_RUN_DIR, and JUDGE_EVIDENCE_DIR remain available for scripts.
    """
    lab = Path(lab)
    if split not in {"calibration", "validation"}:
        raise ValueError("Choose calibration or validation.")
    load_dotenv(lab / ".env", override=False)
    run_dir = Path(os.environ.get("JUDGE_RUN_DIR", lab / ".runs/indirect-judge" / split))
    evidence_dir = Path(os.environ.get("JUDGE_EVIDENCE_DIR", run_dir))
    manifest_path = evidence_dir / "manifest.json"
    if mode == "auto":
        mode = os.environ.get("JUDGE_MODE", "auto")
    if mode == "auto":
        mode = "replay" if manifest_path.exists() else "live"
    if mode not in {"live", "replay"}:
        raise ValueError("Choose auto, live, or replay for mode.")
    if mode == "replay" and not manifest_path.exists():
        raise RuntimeError("No completed run to replay. Run live first, or set JUDGE_EVIDENCE_DIR to a saved run.")

    manifest = json.loads(manifest_path.read_text()) if mode == "replay" else {}
    if manifest and manifest.get("split") != split:
        raise ValueError("Saved run uses a different split; choose the matching evidence directory.")
    providers = manifest.get("providers", DEFAULT_PROVIDERS) if mode == "replay" else {
        "jev": os.environ.get("JEV_PROVIDER", "typesafe"),
        "llm": os.environ.get("LLM_PROVIDER", "openai"),
    }
    cache_dir = (evidence_dir if mode == "replay" else run_dir) / "cache"
    client = JudgeClient(cache_dir, live=(mode == "live"), providers=providers,
                         models=manifest.get("models"), pricing=manifest.get("pricing"), prompts=prompts)
    if client.live:
        keys = {}
        for stage, name in client.key_names.items():
            key = keys.get(name) or os.environ.get(name)
            if not key:
                try:
                    from google.colab import userdata
                    key = userdata.get(name)
                except Exception:
                    # Outside Colab, or no accessible Colab secret: use hidden input.
                    pass
            if not key:
                key = getpass(f"{name} (hidden, never saved): ").strip()
            keys[name] = key
            client.api_keys[stage] = key

    run_dir.mkdir(parents=True, exist_ok=True)
    print("Mode:", mode, "| providers:", client.providers, "| models:", client.models)
    print("HTTP 401/402/403 stops the run. Fix that provider's credentials/credits, then resume.")
    return client, run_dir, manifest


def records_for_json(frame):
    """Keep unavailable quality metrics JSON-null, including pandas NaN values."""
    return frame.astype(object).where(frame.notna(), None).to_dict("records")


def save_run(client, run_dir, manifest, rows, threshold, comparison, bands,
             split="calibration", score_ids=None):
    """Save reports, or verify the same data, policy, cutoff, and metrics on replay."""
    summary = {
        "split": split, "failure_category": FAILURE_CATEGORY, "question": FAILURE_QUESTION,
        "dataset_hash": digest(rows), "dataset_source": JUDGE_DATA_SOURCE,
        "policy_hash": digest([client.prompts, CRITERIA]),
        "evaluation_ids": [row["id"] for row in rows],
        "scoring_ids": list(score_ids) if score_ids is not None else [],
        "threshold": threshold, "threshold_kind": "confidence",
        **client.configuration(),
        "comparison_metrics": records_for_json(comparison.drop(
            columns=["cost_per_1000_USD", "LLM_fraction"]
        ).loc[comparison["method"] != "Always FALSE"]),
    }
    if not client.live:
        for field in ("split", "failure_category", "question", "dataset_hash", "dataset_source", "policy_hash", "evaluation_ids", "scoring_ids",
                      "threshold", "threshold_kind", "comparison_metrics"):
            if summary[field] != manifest[field]:
                raise ValueError(f"Replay does not match the saved run: {field}")
        if client.new_requests:
            raise RuntimeError("Replay must not make HTTP requests.")
    else:
        save_json(run_dir / "manifest.json", summary)
        print("Replay this run by setting JUDGE_MODE=replay and JUDGE_EVIDENCE_DIR to", run_dir)
    save_json(run_dir / "summary.json", {
        **summary, "mode": "live" if client.live else "replay",
        "comparison": records_for_json(comparison), "bands": bands,
        "new_http_attempts": client.new_requests,
    })
    comparison.to_csv(run_dir / "comparison.csv", index=False)
    print("Notebook completed. Fresh HTTP attempts:", client.new_requests)
    print("Outputs:", run_dir)
    print("Unresolved rows:", comparison.set_index("method")["unresolved"].to_dict())
