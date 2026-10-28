"""Offline contract tests for category-specific judging and the confidence gate.

Run with: python -m unittest discover -s routing_agent_evals/tests -v
No API credentials or third-party packages are needed.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))

import judge_lab as lab
from prompts import llm_as_judge_prompt as prompts


def row(case_id="case", human_label="TRUE"):
    return {
        "id": case_id,
        "user_query": "I can work a half-day this Saturday.",
        "predicted_intent": "UNSUPPORTED_REQUEST",
        "human_label": human_label,
    }


def jev_response(label="TRUE", confidence=0.70):
    selected = (confidence + 1) / 2
    return {
        "model": lab.JEV_MODEL,
        "answers": {"interpretation_failure": {
            "type": "choice", "choice": label, "confidence": confidence,
            "probabilities": {
                "TRUE": selected if label == "TRUE" else 1 - selected,
                "FALSE": selected if label == "FALSE" else 1 - selected,
            },
        }},
        "usage": {"input_tokens": 10, "output_tokens": 2},
    }


def llm_response(label=True):
    return {
        "model": lab.LLM_MODEL,
        "choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps({"label": label, "reasoning": "The policy supports this scheduling operation."}),
        }}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2,
                  "prompt_tokens_details": {"cached_tokens": 0}},
    }


class OfflineTestCase(unittest.TestCase):
    def setUp(self):
        # A mistaken transport call must fail immediately rather than use a key.
        self.no_network = patch("urllib.request.urlopen", side_effect=AssertionError("Tests must not call APIs"))
        self.no_network.start()
        self.addCleanup(self.no_network.stop)


class StubTransport(lab.JudgeClient):
    """Use production parsing and request logs with fabricated provider responses."""

    def __init__(self, run_dir, responses=None, **kwargs):
        super().__init__(run_dir,
                         api_keys={"jev": "offline-test-only", "llm": "offline-test-only"}, **kwargs)
        self.responses = responses or {"jev": jev_response(), "llm": llm_response()}
        self.sent = []

    def _send(self, stage, request):
        self.sent.append((stage, copy.deepcopy(request)))
        return {
            "endpoint": self.endpoints[stage], "request": request,
            "attempts": [{"response": copy.deepcopy(self.responses[stage]), "elapsed_seconds": 0.01}],
            "recorded_at": "2026-10-27T00:00:00+00:00",
        }


class PromptContractTests(OfflineTestCase):
    def test_original_llm_baseline_prompt_is_unchanged(self):
        # Hash of the supplied reference revision 21d0f8c's prompt value.
        actual = hashlib.sha256(prompts.INDIRECT_OR_UNDERSPECIFIED_FAILURE_PROMPT.encode()).hexdigest()
        self.assertEqual(actual, "5b74945ec559a19f671e5359df8fcc4aeb057aa7525381033eabde7c8844aaf0")

    def test_jev_asks_the_selected_failure_question(self):
        self.assertEqual(prompts.FAILURE_QUESTION,
                         "Did the router fail to interpret the indirect or underspecified scheduling request?")
        self.assertIn(prompts.FAILURE_QUESTION, prompts.JEV_JUDGE_PROMPT)
        self.assertIn(prompts.FAILURE_CATEGORY, prompts.JEV_JUDGE_PROMPT)
        self.assertEqual(set(prompts.CHOICE_CRITERIA), {"TRUE", "FALSE"})

    def test_judges_share_policy_and_all_six_demonstrations(self):
        self.assertEqual(len(prompts.FEW_SHOT_EXAMPLES), 6)
        self.assertEqual(len(prompts.FEW_SHOT_CASE_IDS), 6)
        for prompt in (prompts.JEV_JUDGE_PROMPT, prompts.INDIRECT_OR_UNDERSPECIFIED_FAILURE_PROMPT):
            self.assertIn(prompts.ROUTING_POLICY, prompt)
            self.assertIn(prompts.FEW_SHOT_EXAMPLES_TEXT, prompt)

    def test_false_is_category_absence_not_overall_correctness(self):
        # These demonstrations disambiguate label meaning for both judges.
        cases = {example["input"]["USER_INPUT"]: example["output"]["label"]
                 for example in prompts.FEW_SHOT_EXAMPLES}
        self.assertIs(cases["I can work a half-day this Saturday."], True)
        self.assertIs(cases["I need to reschedule."], False)
        self.assertIs(cases["I'm not working *this* Saturday."], False)
        self.assertIs(cases["Move Mr. Johnson's appointment from Monday to Tuesday."], False)
        self.assertIn("different kind of error", prompts.CHOICE_CRITERIA["FALSE"])

    def test_request_allowlist_prevents_reference_leakage(self):
        case = row()
        case.update(human_label="SECRET_REFERENCE_LABEL", target_intent="SECRET_TARGET",
                    annotations="SECRET_ANNOTATION", category="SECRET_CATEGORY")
        expected = {"USER_INPUT": case["user_query"], "final_intent": case["predicted_intent"]}
        jev = lab.request_for(case, "jev")
        llm = lab.request_for(case, "llm")
        self.assertEqual(jev["state"], expected)
        self.assertEqual(json.loads(llm["messages"][1]["content"]), expected)
        self.assertEqual(jev["questions"]["interpretation_failure"]["criteria"], prompts.CHOICE_CRITERIA)
        self.assertEqual(jev["questions"]["interpretation_failure"]["instructions"], prompts.JEV_JUDGE_PROMPT)
        self.assertEqual(llm["messages"][0]["content"], prompts.INDIRECT_OR_UNDERSPECIFIED_FAILURE_PROMPT)
        for payload in (jev, llm):
            self.assertNotIn("SECRET_", json.dumps(payload))

    def test_gpt5_request_uses_reasoning_model_parameters(self):
        request = lab.request_for(row(), "llm")
        self.assertEqual(lab.LLM_MODEL, "gpt-5-2025-08-07")
        self.assertEqual(request["model"], lab.LLM_MODEL)
        self.assertEqual(request["max_completion_tokens"], 4096)
        self.assertEqual(request["reasoning_effort"], "medium")
        self.assertNotIn("temperature", request)
        self.assertNotIn("max_tokens", request)
        schema = request["response_format"]["json_schema"]
        self.assertTrue(schema["strict"])
        self.assertEqual(schema["schema"]["properties"]["label"], {"type": "boolean"})
        self.assertEqual(set(schema["schema"]["required"]), {"label", "reasoning"})


class DatasetTests(OfflineTestCase):
    def test_reference_splits_match_notebook_and_exclude_demonstrations(self):
        calibration = lab.load_dataset(LAB / "data")
        validation = lab.load_dataset(LAB / "data", split="validation")
        self.assertEqual(len(calibration), 21)
        self.assertEqual(sum(r["human_label"] == "TRUE" for r in calibration), 9)
        self.assertEqual(len(validation), 150)
        self.assertEqual(sum(r["human_label"] == "TRUE" for r in validation), 11)
        calibration_ids = {r["id"] for r in calibration}
        validation_ids = {r["id"] for r in validation}
        self.assertFalse(calibration_ids & validation_ids)
        self.assertFalse((calibration_ids | validation_ids) & prompts.FEW_SHOT_CASE_IDS)

    def test_loader_rejects_unsupported_split(self):
        for split in ("all", "train", "test", "", None):
            with self.subTest(split=split), self.assertRaises(ValueError):
                lab.load_dataset(LAB / "data", split=split)


class ResponseValidationTests(OfflineTestCase):
    def test_valid_true_and_false_are_preserved(self):
        for label in lab.LABELS:
            with self.subTest(label=label):
                actual = lab.parse_jev(jev_response(label))
                self.assertEqual(actual["label"], label)
                self.assertEqual(actual["confidence"], 0.70)

    def test_malformed_jev_outputs_are_rejected(self):
        mutations = [
            lambda a: a.update(type="noul"),
            lambda a: a.update(choice="UNKNOWN"),
            lambda a: a.update(choice="FALSE"),
            lambda a: a.update(probabilities={"TRUE": 1.0}),
            lambda a: a.update(probabilities={"TRUE": 0.85, "FALSE": 0.15, "OTHER": 0}),
            lambda a: a.update(probabilities={"TRUE": True, "FALSE": 0}),
            lambda a: a.update(probabilities={"TRUE": "0.85", "FALSE": 0.15}),
            lambda a: a.update(probabilities={"TRUE": float("nan"), "FALSE": 0.15}),
            lambda a: a.update(probabilities={"TRUE": 0.8, "FALSE": 0.8}),
        ]
        mutations += [lambda a, value=value: a.update(confidence=value)
                      for value in (None, True, "0.7", float("nan"), float("inf"), -0.1, 1.1)]
        for index, mutate in enumerate(mutations):
            raw = jev_response()
            mutate(raw["answers"]["interpretation_failure"])
            with self.subTest(mutation=index), self.assertRaises((ValueError, TypeError, KeyError)):
                lab.parse_jev(raw)
        with self.assertRaises((ValueError, TypeError, KeyError)):
            lab.parse_jev({"model": lab.JEV_MODEL, "answers": {}})

    def test_invalid_llm_response_is_unresolved_in_client(self):
        invalid = llm_response()
        invalid["choices"][0]["message"]["content"] = '{"label":"TRUE","reasoning":"not a boolean"}'
        with tempfile.TemporaryDirectory() as folder:
            client = StubTransport(folder, {"jev": jev_response(), "llm": invalid})
            result = client.call(row(), "llm")
        self.assertIsNone(result["label"])
        self.assertIsNotNone(result["error"])

    def test_llm_refusal_and_truncation_are_rejected(self):
        for reason, refusal in (("length", None), ("stop", "refused")):
            raw = llm_response()
            raw["choices"][0]["finish_reason"] = reason
            raw["choices"][0]["message"]["refusal"] = refusal
            with self.subTest(reason=reason, refusal=refusal), self.assertRaises(ValueError):
                lab.parse_llm(raw)


class ConfidenceGateTests(OfflineTestCase):
    def test_boundary_applies_to_true_and_false(self):
        for label in lab.LABELS:
            for confidence, expected in ((0.699999, True), (0.70, False), (0.700001, False)):
                with self.subTest(label=label, confidence=confidence):
                    self.assertEqual(lab.needs_llm({"label": label, "confidence": confidence}, 0.70), expected)

    def test_gate_uses_native_confidence_not_true_probability(self):
        accepted_false = lab.parse_jev(jev_response("FALSE", 0.70))
        self.assertLess(accepted_false["probabilities"]["TRUE"], 0.70)
        self.assertFalse(lab.needs_llm(accepted_false, 0.70))
        uncertain_true = lab.parse_jev(jev_response("TRUE", 0.40))
        self.assertEqual(uncertain_true["probabilities"]["TRUE"], 0.70)
        self.assertTrue(lab.needs_llm(uncertain_true, 0.70))

    def test_invalid_result_escalates_even_with_high_confidence(self):
        for result in ({"label": None, "confidence": 1},
                       {"label": "TRUE", "confidence": None},
                       {"label": "TRUE", "confidence": 1, "error": "Invalid judge output"}):
            with self.subTest(result=result):
                self.assertTrue(lab.needs_llm(result, 0.70))

    def test_cascade_calls_fallback_only_for_escalated_rows(self):
        rows = [row(case_id) for case_id in ("accepted_true", "accepted_false", "low", "invalid")]
        jev = {
            "accepted_true": {"label": "TRUE", "confidence": 0.70, "error": None},
            "accepted_false": {"label": "FALSE", "confidence": 0.70, "error": None},
            "low": {"label": "TRUE", "confidence": 0.69, "error": None},
            "invalid": {"label": None, "error": "invalid response"},
        }

        class FakeClient:
            def __init__(self):
                self.calls = []

            def call(self, case, stage):
                self.calls.append((case["id"], stage))
                return jev[case["id"]] if stage == "jev" else {"label": "FALSE", "error": None}

        client = FakeClient()
        result = lab.run_cascade(rows, client, 0.70)
        self.assertEqual({case_id for case_id, stage in client.calls if stage == "jev"}, set(jev))
        self.assertEqual({case_id for case_id, stage in client.calls if stage == "llm"}, {"low", "invalid"})
        self.assertEqual(len(client.calls), 6)
        self.assertEqual(result["accepted_true"]["route"], "Jev")
        self.assertEqual(result["accepted_false"]["label"], "FALSE")
        self.assertEqual(result["low"]["route"], "LLM")

    def test_invalid_fallback_stays_unresolved(self):
        invalid = llm_response()
        invalid["choices"][0]["message"]["content"] = "not JSON"
        with tempfile.TemporaryDirectory() as folder:
            client = StubTransport(folder, {"jev": jev_response(confidence=0.69), "llm": invalid})
            result = lab.run_cascade([row()], client, 0.70)["case"]
        self.assertEqual(result["route"], "LLM")
        self.assertIsNone(result["label"])
        self.assertIsNotNone(result["error"])


class LiveRequestTests(OfflineTestCase):
    def test_identical_requests_are_sent_twice_and_logged_separately(self):
        with tempfile.TemporaryDirectory() as folder:
            client = StubTransport(folder)
            first = client.call(row(), "jev")
            repeated = client.call(row(), "jev")
            self.assertEqual(first["request_hash"], repeated["request_hash"])
            self.assertNotEqual(first["call_id"], repeated["call_id"])
            self.assertEqual(len(client.sent), 2)
            logs = sorted((Path(folder) / "requests").glob("jev-*.json"))
            self.assertEqual(len(logs), 2)
            entries = [json.loads(path.read_text()) for path in logs]
            self.assertEqual({entry["call_id"] for entry in entries},
                             {first["call_id"], repeated["call_id"]})
            self.assertEqual({entry["request_hash"] for entry in entries},
                             {first["request_hash"]})
            for entry in entries:
                self.assertEqual(entry["request"], client.sent[0][1])
                self.assertNotIn("offline-test-only", json.dumps(entry))

    def test_prompt_change_is_logged_without_overwriting_earlier_call(self):
        with tempfile.TemporaryDirectory() as folder:
            client = StubTransport(folder)
            first = client.call(row(), "jev")
            client.prompts["jev"] += "\nClarify the selected failure boundary."
            changed = client.call(row(), "jev")
            self.assertNotEqual(first["request_hash"], changed["request_hash"])
            self.assertNotEqual(first["call_id"], changed["call_id"])
            self.assertEqual(len(client.sent), 2)
            self.assertEqual(len(list((Path(folder) / "requests").glob("jev-*.json"))), 2)

    def test_independent_baseline_and_fallback_each_make_fresh_llm_calls(self):
        with tempfile.TemporaryDirectory() as folder:
            client = StubTransport(folder, {"jev": jev_response(confidence=0.69),
                                            "llm": llm_response()})
            baseline = client.call(row(), "llm")
            cascade = lab.run_cascade([row()], client, 0.70)["case"]
            self.assertEqual([stage for stage, request in client.sent], ["llm", "jev", "llm"])
            self.assertEqual(client.sent[0][1], client.sent[2][1])
            self.assertNotEqual(baseline["call_id"], cascade["call_id"])
            self.assertEqual(baseline["request_hash"], cascade["request_hash"])
            self.assertEqual(cascade["route"], "LLM")
            self.assertEqual(len(list((Path(folder) / "requests").glob("llm-*.json"))), 2)


class MetricTests(OfflineTestCase):
    def test_unresolved_cases_reduce_coverage_without_becoming_false_negatives(self):
        rows = [row("tp", "TRUE"), row("missing", "TRUE"), row("tn", "FALSE"), row("error", "FALSE")]
        predictions = {"tp": {"label": "TRUE"}, "tn": {"label": "FALSE"},
                       "error": {"label": "TRUE", "error": "invalid output"}}
        actual = lab.metrics(rows, predictions, "Jev")
        self.assertEqual(actual["n"], 4)
        self.assertEqual(actual["resolved_n"], 2)
        self.assertEqual(actual["scored_n"], 2)
        self.assertEqual(actual["unresolved"], 2)
        self.assertEqual(actual["coverage"], 0.5)
        self.assertEqual(actual["FN"], 0)
        self.assertEqual(actual["accuracy"], 1.0)
        self.assertEqual(actual["recall_TRUE"], 1.0)

    def test_no_resolved_cases_have_no_quality_score(self):
        actual = lab.metrics([row()], {}, "Jev")
        for key in ("accuracy", "precision_TRUE", "recall_TRUE", "f1_TRUE"):
            self.assertIsNone(actual[key])
        self.assertEqual(actual["coverage"], 0)
        self.assertEqual(actual["unresolved"], 1)

    def test_comparison_scores_all_approaches_on_the_same_resolved_rows(self):
        rows = [row("shared", "TRUE"), row("jev_only", "FALSE"), row("llm_only", "TRUE")]
        approaches = {
            "Jev": {"shared": {"label": "TRUE"}, "jev_only": {"label": "FALSE"}},
            "LLM": {"shared": {"label": "FALSE"}, "llm_only": {"label": "TRUE"}},
        }
        result = {entry["method"]: entry for entry in lab.comparison_metrics(rows, approaches)}
        self.assertEqual(set(result), set(approaches))
        for entry in result.values():
            self.assertEqual(entry["n"], 3)
            self.assertEqual(entry["resolved_n"], 2)
            self.assertEqual(entry["scored_n"], 1)
            self.assertAlmostEqual(entry["coverage"], 2 / 3)
        self.assertEqual(result["Jev"]["accuracy"], 1.0)
        self.assertEqual(result["LLM"]["accuracy"], 0.0)

    def test_metrics_reject_unknown_requested_score_ids(self):
        with self.assertRaises(ValueError):
            lab.metrics([row()], {"case": {"label": "TRUE"}}, "Jev", score_ids={"unknown"})

    def test_confidence_band_fixes_and_regressions_require_paired_resolution(self):
        rows = [row(case_id) for case_id in ("fix", "regression", "jev_error", "llm_error")]
        jev = {
            "fix": {"label": "FALSE", "confidence": 0.6},
            "regression": {"label": "TRUE", "confidence": 0.6},
            "jev_error": {"label": None, "error": "bad"},
            "llm_error": {"label": "FALSE", "confidence": 0.6},
        }
        llm = {
            "fix": {"label": "TRUE"}, "regression": {"label": "FALSE"},
            "jev_error": {"label": "TRUE"}, "llm_error": {"label": None, "error": "bad"},
        }
        result = lab.confidence_bands(rows, jev, llm, 0.7)
        self.assertEqual(len(result), 1)
        band = result[0]
        self.assertEqual(band["n"], 4)
        self.assertEqual(band["paired_n"], 2)
        self.assertEqual(band["Jev_unresolved"], 1)
        self.assertEqual(band["LLM_unresolved"], 1)
        self.assertEqual(band["LLM_fixes"], 1)
        self.assertEqual(band["LLM_regressions"], 1)
        self.assertEqual(band["Jev_accuracy"], 0.5)
        self.assertEqual(band["LLM_accuracy"], 0.5)


class ThresholdDiagnosticTests(OfflineTestCase):
    def test_paired_lower_bound_and_costs_describe_each_candidate(self):
        rows = [row(case_id) for case_id in ("loss", "gain", "same", "escalated")]
        jev = {
            "loss": {"label": "FALSE", "confidence": 0.9, "cost_usd": 0.01},
            "gain": {"label": "TRUE", "confidence": 0.9, "cost_usd": 0.01},
            "same": {"label": "TRUE", "confidence": 0.9, "cost_usd": 0.01},
            "escalated": {"label": "FALSE", "confidence": 0.6, "cost_usd": 0.01},
        }
        llm = {case_id: {"label": "FALSE" if case_id == "gain" else "TRUE", "cost_usd": 0.1}
               for case_id in jev}
        original = copy.deepcopy((jev, llm))
        original_threshold = lab.CONFIDENCE_THRESHOLD
        results = lab.threshold_sweep(rows, jev, llm, [0.70, 1.0])
        self.assertEqual([entry["threshold"] for entry in results], [0.70, 1.0])
        candidate, all_fallback = results
        self.assertEqual(candidate["n"], 4)
        self.assertEqual(candidate["paired_n"], 4)
        self.assertEqual(candidate["accepted_n"], 3)
        self.assertEqual(candidate["llm_routes"], 1)
        self.assertEqual(candidate["accepted_fraction"], 0.75)
        self.assertEqual(candidate["cascade_accuracy"], 0.75)
        self.assertEqual(candidate["llm_accuracy"], 0.75)
        self.assertEqual(candidate["accuracy_difference"], 0.0)
        # The paired differences are [-1, +1, 0, 0], with sample SD sqrt(2/3).
        expected_lower_bound = -1.645 * math.sqrt(2 / 3) / math.sqrt(4)
        self.assertAlmostEqual(candidate["accuracy_difference_lcb_95"], expected_lower_bound)
        self.assertEqual(candidate["noninferiority_margin"], -0.02)
        self.assertIs(candidate["noninferiority_pass"], False)
        self.assertAlmostEqual(candidate["cascade_cost_usd"], 0.14)
        self.assertAlmostEqual(candidate["llm_cost_usd"], 0.4)
        self.assertAlmostEqual(candidate["cost_savings_fraction"], 0.65)
        self.assertEqual(all_fallback["accepted_n"], 0)
        self.assertEqual(all_fallback["llm_routes"], 4)
        self.assertEqual(all_fallback["accuracy_difference_lcb_95"], 0.0)
        self.assertIs(all_fallback["noninferiority_pass"], True)
        self.assertAlmostEqual(all_fallback["cascade_cost_usd"], 0.44)
        self.assertAlmostEqual(all_fallback["cost_savings_fraction"], -0.1)
        self.assertEqual((jev, llm), original)
        self.assertEqual(lab.CONFIDENCE_THRESHOLD, original_threshold)

    def test_missing_or_invalid_decisions_cannot_certify_a_threshold(self):
        rows = [row("complete"), row("unresolved")]
        for unresolved_stage in ("jev", "llm"):
            for bad_result in ({}, {"label": None}, {"label": "TRUE", "error": "bad output"}):
                jev = {case["id"]: {"label": "TRUE", "confidence": 0.9} for case in rows}
                llm = {case["id"]: {"label": "TRUE"} for case in rows}
                (jev if unresolved_stage == "jev" else llm)["unresolved"] = bad_result
                with self.subTest(stage=unresolved_stage, result=bad_result):
                    result = lab.threshold_sweep(rows, jev, llm, [0.70])[0]
                    self.assertEqual(result["paired_n"], 1)
                    self.assertEqual(result["status"], "insufficient_evidence")
                    self.assertIsNone(result["noninferiority_pass"])
                    self.assertIsNone(result["accuracy_difference_lcb_95"])

    def test_single_example_cannot_supply_a_sample_variance_bound(self):
        result = lab.threshold_sweep(
            [row()], {"case": {"label": "TRUE", "confidence": 0.9}},
            {"case": {"label": "TRUE"}}, [0.70])[0]
        self.assertEqual(result["paired_n"], 1)
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertIsNone(result["noninferiority_pass"])
        self.assertIsNone(result["accuracy_difference_lcb_95"])

    def test_good_resolved_subset_cannot_certify_an_incomplete_panel(self):
        rows = [row(case_id) for case_id in ("one", "two", "missing")]
        jev = {case_id: {"label": "TRUE", "confidence": 0.9} for case_id in ("one", "two")}
        llm = {case["id"]: {"label": "TRUE"} for case in rows}
        result = lab.threshold_sweep(rows, jev, llm, [0.70])[0]
        self.assertEqual(result["paired_n"], 2)
        self.assertEqual(result["accuracy_difference_lcb_95"], 0.0)
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertIsNone(result["noninferiority_pass"])

    def test_unknown_provider_cost_is_not_reported_as_zero_or_savings(self):
        rows = [row(case_id) for case_id in ("one", "two")]
        for unknown_stage in ("jev", "llm"):
            jev = {case["id"]: {"label": "TRUE", "confidence": 0.6, "cost_usd": 0.01}
                   for case in rows}
            llm = {case["id"]: {"label": "TRUE", "cost_usd": 0.1} for case in rows}
            (jev if unknown_stage == "jev" else llm)["one"]["unknown_cost_attempts"] = 1
            with self.subTest(stage=unknown_stage):
                result = lab.threshold_sweep(rows, jev, llm, [0.70])[0]
                self.assertEqual(result["unknown_cost_attempts"], 1)
                self.assertIsNone(result["cascade_cost_usd"])
                self.assertIsNone(result["cost_savings_fraction"])
                if unknown_stage == "llm":
                    self.assertIsNone(result["llm_cost_usd"])
                else:
                    self.assertAlmostEqual(result["llm_cost_usd"], 0.2)


if __name__ == "__main__":
    unittest.main()
