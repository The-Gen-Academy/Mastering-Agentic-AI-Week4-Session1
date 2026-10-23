import copy
import json
import sys
from pathlib import Path
from unittest.mock import patch
from io import BytesIO

import pytest

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from judge_lab import (JudgeClient, combined, confidence_bands, digest, ENDPOINTS,
                       load_dataset, metrics, needs_llm, parse_jev, parse_llm,
                       request_for, run_cascade, select_threshold, split_dataset)


def row(identity="a", label="TRUE", query="Clear my 3 PM"):
    return {"id": identity, "user_query": query, "predicted_intent": "UNSUPPORTED_REQUEST",
            "human_label": label, "TARGET_INTENT": "SECRET_GOLD", "comment": "SECRET_COMMENT"}


def result(label="TRUE", confidence=0.7, cost=0.01, error=None):
    return dict(label=label, confidence=confidence, cost_usd=cost, error=error,
                unreported_cost_attempts=0, attempts=1, elapsed_seconds=0.1,
                input_tokens=100, output_tokens=5)


def raw_jev():
    return {"model": "jev-1.13.0", "answers": {"vague_failure": {
        "type": "choice", "choice": "TRUE", "probabilities": {"TRUE": 0.85, "FALSE": 0.15},
        "confidence": 0.7}}}


@pytest.mark.parametrize("stage", ["jev", "llm"])
def test_requests_exclude_human_labels_and_other_judges(stage):
    original = row()
    changed = dict(original, human_label="FALSE", TARGET_INTENT="OTHER_GOLD",
                   comment="OTHER_COMMENT", jev_choice="FALSE", jev_confidence=0.1)
    assert request_for(original, stage) == request_for(changed, stage)
    assert "SECRET_" not in json.dumps(request_for(original, stage))


def test_jev_choice_is_not_a_score_or_confidence_probability():
    parsed = parse_jev(raw_jev())
    assert parsed["label"] == "TRUE"
    assert parsed["confidence"] == 0.7
    assert parsed["probabilities"]["TRUE"] == 0.85


@pytest.mark.parametrize("mutation", [
    lambda a: a.update(type="score"),
    lambda a: a.update(choice="OTHER"),
    lambda a: a.update(choice="FALSE"),
    lambda a: a.update(confidence=float("nan")),
    lambda a: a.update(confidence=True),
    lambda a: a.update(probabilities={"TRUE": 0.6, "FALSE": 0.1}),
    lambda a: a.update(probabilities={"TRUE": 0.85, "FALSE": 0.15, "OTHER": 0}),
    lambda a: a.update(probabilities={"TRUE": -0.1, "FALSE": 1.1}),
])
def test_invalid_jev_outputs_rejected(mutation):
    raw = raw_jev()
    mutation(raw["answers"]["vague_failure"])
    with pytest.raises(ValueError):
        parse_jev(raw)


@pytest.mark.parametrize("finish,content", [
    ("length", '{"label":"TRUE","reason":"x"}'),
    ("stop", '{"label":true,"reason":"x"}'),
    ("stop", '{"label":"TRUE","reason":""}'),
    ("stop", '{"label":"TRUE","reason":"x","extra":1}'),
])
def test_invalid_llm_outputs_are_not_boolean_coerced(finish, content):
    with pytest.raises(ValueError):
        parse_llm({"choices": [{"finish_reason": finish, "message": {"content": content}}]})


def test_confidence_boundary_and_invalid_result_fallback():
    assert not needs_llm(result(confidence=0.7), 0.7)
    assert needs_llm(result(confidence=0.699), 0.7)
    assert needs_llm(result(label=None, confidence=1), 0.7)
    assert needs_llm(result(confidence=float("nan")), 0.7)
    assert needs_llm(result(error="invalid", confidence=1), 0.7)
    with pytest.raises(ValueError):
        needs_llm(result(), 1.01)


def test_cascade_only_calls_llm_for_low_confidence_and_invalid():
    rows = [row("high"), row("low"), row("invalid")]
    calls = []
    class Client:
        def call(self, r, stage):
            calls.append((r["id"], stage))
            return result() if stage == "llm" else {
                "high": result(confidence=0.7), "low": result(confidence=0.69),
                "invalid": result(label=None, error="bad")}[r["id"]]
    values = run_cascade(rows, Client(), 0.7)
    assert sorted(i for i, stage in calls if stage == "llm") == ["invalid", "low"]
    assert values["high"]["route"] == "Jev"
    assert values["low"]["cost_usd"] == 0.02


def test_failed_escalation_stays_unresolved_and_counts_both_costs():
    first = result(cost=0.01)
    second = result(label=None, cost=0.03, error="bad")
    second.update(attempts=2, unreported_cost_attempts=1)
    out = combined(first, second)
    assert out["label"] is None and out["error"]
    assert out["cost_usd"] == 0.04 and out["attempts"] == 3
    assert out["unreported_cost_attempts"] == 1


def test_accuracy_denominator_includes_unresolved_and_rare_class():
    rows = [row("p", "TRUE"), row("n", "FALSE"), row("u", "FALSE")]
    out = metrics(rows, {"p": result("FALSE"), "n": result("FALSE"), "u": result(None)}, "test")
    assert out["accuracy"] == 1 / 3
    assert out["recall_TRUE"] == 0 and out["f1_TRUE"] == 0 and out["unresolved"] == 1


def test_split_is_stable_and_duplicate_queries_never_cross_splits():
    rows = [row(str(i), "TRUE" if i % 3 == 0 else "FALSE", f"query {i}") for i in range(30)]
    rows.append(row("duplicate", "FALSE", "  QUERY 0  "))
    a, b = split_dataset(rows)
    c, d = split_dataset(list(reversed(rows)))
    assert {r["id"] for r in a} == {r["id"] for r in c}
    norm = lambda rows: {" ".join(r["user_query"].lower().split()) for r in rows}
    assert not norm(a) & norm(b)
    assert {r["id"] for r in a + b} == {r["id"] for r in rows}


def test_threshold_uses_only_calibration_ids():
    calibration = [row("p"), row("n", "FALSE")]
    jev = {"p": result("FALSE", 0.2), "n": result("FALSE", 0.9)}
    llm = {"p": result("TRUE"), "n": result("TRUE")}
    threshold, sweep = select_threshold(calibration, jev, llm)
    assert threshold == 0.25
    assert all(s["n"] == 2 for s in sweep)
    llm["heldout"] = result("FALSE")
    assert select_threshold(calibration, jev, llm)[0] == threshold


def test_replay_requires_exact_request_and_never_uses_network(tmp_path):
    client = JudgeClient(tmp_path)
    with patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden")):
        with pytest.raises(FileNotFoundError):
            client.call(row(), "jev")
        request = request_for(row(), "jev")
        key = digest({"endpoint": ENDPOINTS["jev"], "request": request})
        path = tmp_path / f"jev-{key}.json"
        entry = {"endpoint": ENDPOINTS["jev"], "request": request,
                 "attempts": [{"response": raw_jev(), "elapsed_seconds": 0.1}]}
        path.write_text(json.dumps(entry))
        assert client.call(row(), "jev")["unreported_cost_attempts"] == 1
        entry["request"]["state"]["user_query"] = "tampered"
        path.write_text(json.dumps(entry))
        with pytest.raises(ValueError, match="Cache request mismatch"):
            client.call(row(), "jev")


def test_source_audit_preserves_explicit_judge_references():
    rows = load_dataset(LAB / "data")
    assert len(rows) == 300 and sum(r["human_label"] == "TRUE" for r in rows) == 20
    assert sum("target_differs_between_csvs" in r["reference_issues"] for r in rows) == 10
    assert sum("query_differs_between_csvs" in r["reference_issues"] for r in rows) == 1
    assert sum("legacy_intent_match_contains_prose" in r["reference_issues"] for r in rows) == 248


def test_direct_typesafe_estimate_uses_usage_and_frozen_rate(tmp_path):
    client = JudgeClient(tmp_path)
    raw = raw_jev()
    raw["usage"] = {"input_tokens": 1_000_000, "output_tokens": 33}
    attempt = {"response": raw}
    assert client.attempt_cost(attempt, "jev") == (0.042, "estimated")
    raw["usage"]["cost"] = 0.035
    assert client.attempt_cost(attempt, "jev") == (0.035, "reported")
    del raw["usage"]["cost"]
    raw["model"] = "unknown-model"
    assert client.attempt_cost(attempt, "jev") == (None, "unknown")
    assert client.attempt_cost({"error": "HTTP 500"}, "jev") == (None, "unknown")


def test_credentials_are_scoped_to_stage_and_never_cached(tmp_path):
    client = JudgeClient(tmp_path, live=True,
                         api_keys={"jev": "test-typesafe-secret", "llm": "test-openai-secret"})
    calls = []
    def send(req, **kwargs):
        calls.append((req.full_url, req.headers["Authorization"]))
        raw = raw_jev() if "typesafe.ai" in req.full_url else {
            "choices": [{"finish_reason": "stop", "message": {
                "content": '{"label":"TRUE","reason":"test"}'}}]}
        return BytesIO(json.dumps(raw).encode())
    with patch("urllib.request.urlopen", side_effect=send):
        client.call(row(), "jev")
        client.call(row(), "llm")
    assert calls == [(ENDPOINTS["jev"], "Bearer test-typesafe-secret"),
                     (ENDPOINTS["llm"], "Bearer test-openai-secret")]
    assert "secret" not in json.dumps(client.configuration())
    assert all("secret" not in p.read_text() for p in tmp_path.glob("*.json"))


def test_openai_cost_estimate_counts_cached_tokens_at_cached_price(tmp_path):
    client = JudgeClient(tmp_path, providers={"jev": "typesafe", "llm": "openai"})
    raw = {"model": "gpt-4.1-mini-2025-04-14", "service_tier": "default",
           "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 100_000,
                     "prompt_tokens_details": {"cached_tokens": 250_000}}}
    cost, source = client.attempt_cost({"response": raw}, "llm")
    assert cost == pytest.approx(0.485) and source == "estimated"
    raw["usage"]["prompt_tokens_details"]["cached_tokens"] = 1_000_001
    assert client.attempt_cost({"response": raw}, "llm") == (None, "unknown")
    raw["usage"]["prompt_tokens_details"] = {}
    assert client.attempt_cost({"response": raw}, "llm") == (None, "unknown")
    raw["usage"]["prompt_tokens_details"]["cached_tokens"] = 0
    raw["service_tier"] = "priority"
    assert client.attempt_cost({"response": raw}, "llm") == (None, "unknown")


def test_provider_changes_cannot_reuse_another_endpoint_cache(tmp_path):
    first = JudgeClient(tmp_path, live=True, api_keys={"jev": "test-secret"})
    with patch("urllib.request.urlopen", return_value=BytesIO(json.dumps(raw_jev()).encode())):
        first.call(row(), "jev")
    second = JudgeClient(tmp_path, providers={"jev": "openrouter", "llm": "openrouter"})
    with patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden")):
        with pytest.raises(FileNotFoundError):
            second.call(row(), "jev")


def test_saved_jev_stage_replays_without_network():
    manifest_path = LAB / "data/jev_judge_run/jev_manifest.json"
    if not manifest_path.exists():
        pytest.skip("Direct TypeSafe evidence has not been captured yet")
    manifest = json.loads(manifest_path.read_text())
    rows = load_dataset(LAB / "data")
    calibration, heldout = split_dataset(rows)
    assert digest(rows) == manifest["dataset_hash"]
    client = JudgeClient(manifest_path.parent / "cache", providers=manifest["providers"],
                         models=manifest["models"], pricing=manifest["pricing"])
    with patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden")):
        jev = {r["id"]: client.call(r, "jev") for r in rows}
    assert all(r["error"] is None for r in jev.values())
    assert metrics(calibration, jev, "Jev calibration") == manifest["calibration_metrics"]
    assert metrics(heldout, jev, "Jev held out") == manifest["heldout_metrics"]
    assert client.new_requests == 0


def test_saved_run_replays_with_frozen_split_and_expected_metrics():
    manifest_path = LAB / "data/jev_judge_run/manifest.json"
    if not manifest_path.exists():
        pytest.skip("Live evidence has not been captured yet")
    manifest = json.loads(manifest_path.read_text())
    rows = load_dataset(LAB / "data")
    calibration, heldout = split_dataset(rows)
    assert digest(rows) == manifest["dataset_hash"]
    assert [r["id"] for r in calibration] == manifest["calibration_ids"]
    client = JudgeClient(manifest_path.parent / "cache", providers=manifest.get("providers"),
                         models=manifest.get("models"), pricing=manifest.get("pricing"))
    with patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden")):
        jev = {r["id"]: client.call(r, "jev") for r in rows}
        llm = {r["id"]: client.call(r, "llm") for r in rows}
        threshold, _ = select_threshold(calibration, jev, llm)
        assert threshold == manifest["threshold"]
        cascade = run_cascade(heldout, client, threshold, jev=jev)
    actual = [metrics(heldout, values, name) for name, values in
              [("Jev alone", jev), ("LLM alone", llm), ("Jev then LLM", cascade)]]
    assert actual == manifest["heldout_metrics"]
    assert client.new_requests == 0
