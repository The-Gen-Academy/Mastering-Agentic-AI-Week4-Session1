"""Hidden credential entry and report export for live judge experiments."""
import os
from datetime import datetime, timezone
from getpass import getpass
from pathlib import Path

from dotenv import load_dotenv
from judge_lab import CRITERIA, JUDGE_DATA_SOURCE, JudgeClient, digest, save_json
from prompts.llm_as_judge_prompt import FAILURE_CATEGORY, FAILURE_QUESTION


def setup_run(lab, prompts=None, stages=("jev", "llm")):
    """Create a live client; saved request logs are never replayed."""
    lab = Path(lab)
    load_dotenv(lab / ".env", override=False)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    parent = Path(os.environ.get("JUDGE_RUN_DIR", lab / ".runs/live-judge"))
    run_dir = parent / stamp
    client = JudgeClient(run_dir, prompts=prompts)
    for stage in stages:
        name = client.key_names[stage]
        key = os.environ.get(name)
        if not key:
            try:
                from google.colab import userdata
                key = userdata.get(name)
            except Exception:
                pass
        if not key:
            key = getpass(f"{name} (hidden, never saved): ").strip()
        if not key:
            raise ValueError(f"A {name} is required for this live run.")
        client.api_keys[stage] = key
    run_dir.mkdir(parents=True, exist_ok=True)
    print("Live models:", {stage: client.models[stage] for stage in stages})
    print("Every judge call sends an API request. Re-running inference cells incurs new usage.")
    return client, run_dir


def records_for_json(frame):
    """Keep unavailable metrics JSON-null, including pandas NaN values."""
    return frame.astype(object).where(frame.notna(), None).to_dict("records")


def save_run(client, run_dir, rows, threshold, comparison, bands, score_ids,
             calibration_rows=None, calibration=None):
    summary = {
        "mode": "live", "failure_category": FAILURE_CATEGORY, "question": FAILURE_QUESTION,
        "dataset_hash": digest(rows), "dataset_source": JUDGE_DATA_SOURCE,
        "policy_hash": digest([client.prompts, CRITERIA]),
        "evaluation_ids": [row["id"] for row in rows], "scoring_ids": list(score_ids),
        "calibration_ids": [row["id"] for row in (calibration_rows or [])],
        "threshold": threshold, "threshold_kind": "native_choice_confidence",
        **client.configuration(), "comparison": records_for_json(comparison),
        "bands": bands, "threshold_diagnostics": calibration,
        "new_http_attempts": client.new_requests,
    }
    save_json(run_dir / "summary.json", summary)
    comparison.to_csv(run_dir / "comparison.csv", index=False)
    print("Completed live HTTP attempts:", client.new_requests)
    print("Outputs:", run_dir)
    print("Unresolved cases:", comparison.set_index("method")["unresolved"].to_dict())
