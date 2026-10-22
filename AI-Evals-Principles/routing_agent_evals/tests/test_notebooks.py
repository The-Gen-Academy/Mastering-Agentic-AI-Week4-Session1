"""Execute every notebook cell with offline, synthetic transport fixtures.

These fixtures test the notebook plumbing, not model quality. They are created
only under pytest's temporary directory and never replace benchmark evidence.
"""
import json
import shutil
import sys
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from judge_lab import (JudgeClient, ENDPOINTS, POLICY, CRITERIA, digest, request_for,
                       load_dataset, split_dataset, select_threshold, run_cascade,
                       metrics, save_json)


@pytest.fixture(scope="module")
def notebook_lab(tmp_path_factory):
    target = tmp_path_factory.mktemp("synthetic-notebook-integration")
    for name in ("judge_lab.py", "llm_as_judge.ipynb", "routing_agent_evals.ipynb"):
        shutil.copy2(LAB / name, target / name)
    shutil.copytree(LAB / "prompts", target / "prompts")
    (target / "data").mkdir()
    for path in (LAB / "data").glob("*.csv"):
        shutil.copy2(path, target / "data" / path.name)
    rows = load_dataset(target / "data")
    evidence = target / "data/jev_judge_run"
    for row in rows:
        seed = int(digest(row["user_query"])[:8], 16)
        for stage in ("jev", "llm"):
            request = request_for(row, stage)
            key = digest({"endpoint": ENDPOINTS[stage], "request": request})
            if stage == "jev":
                p = (seed % 81 + 10) / 100
                raw = {"model": "SYNTHETIC_TEST_FIXTURE", "answers": {"vague_failure": {
                    "type": "choice", "choice": "TRUE" if p >= .5 else "FALSE",
                    "probabilities": {"TRUE": p, "FALSE": 1-p}, "confidence": abs(2*p-1)}}}
            else:
                raw = {"model": "SYNTHETIC_TEST_FIXTURE", "choices": [{"finish_reason": "stop",
                       "message": {"content": json.dumps({"label": "TRUE" if seed % 5 == 0 else "FALSE",
                                   "reason": "Synthetic fixture; not a model judgment."})}}]}
            raw["usage"] = {"cost": .0001 if stage == "jev" else .001,
                            "input_tokens": 100, "output_tokens": 10}
            save_json(evidence / "cache" / f"{stage}-{key}.json", {
                "endpoint": ENDPOINTS[stage], "request": request,
                "attempts": [{"response": raw, "elapsed_seconds": .1}],
                "fixture_notice": "SYNTHETIC INTEGRATION TEST ONLY"})
    client = JudgeClient(evidence / "cache")
    jev = {r["id"]: client.call(r, "jev") for r in rows}
    llm = {r["id"]: client.call(r, "llm") for r in rows}
    calibration, heldout = split_dataset(rows)
    threshold, _ = select_threshold(calibration, jev, llm)
    cascade = run_cascade(heldout, client, threshold, jev=jev)
    save_json(evidence / "manifest.json", {
        "dataset_hash": digest(rows), "policy_hash": digest([POLICY, CRITERIA]),
        "calibration_ids": [r["id"] for r in calibration],
        "heldout_ids": [r["id"] for r in heldout], "threshold": threshold,
        "heldout_metrics": [metrics(heldout, values, name) for name, values in
                            [("Jev alone", jev), ("LLM alone", llm), ("Jev then LLM", cascade)]]})
    return target


@pytest.mark.parametrize("name", ["routing_agent_evals.ipynb", "llm_as_judge.ipynb"])
def test_notebook_runs_offline_with_synthetic_responses(notebook_lab, name):
    notebook = nbformat.read(notebook_lab / name, as_version=4)
    nbformat.validate(notebook)
    notebook.cells.insert(0, nbformat.v4.new_code_cell(
        "import os, urllib.request\n"
        "os.environ['JUDGE_MODE'] = 'replay'\n"
        "os.environ['ROUTER_MODE'] = 'replay'\n"
        "os.environ.pop('JUDGE_RUN_DIR', None)\n"
        "os.environ.pop('JUDGE_EVIDENCE_DIR', None)\n"
        "def network_forbidden(*args, **kwargs):\n"
        "    raise AssertionError('Network attempted during offline integration test')\n"
        "urllib.request.urlopen = network_forbidden\n"))
    NotebookClient(notebook, timeout=180, kernel_name="python3",
                   resources={"metadata": {"path": str(notebook_lab)}}).execute()
    assert all(c.execution_count is not None for c in notebook.cells if c.cell_type == "code")
