"""Live Jev/LLM judge comparison with an audit log for every call.

Only the public user query and candidate intent reach the judges. Human labels
are used by the metric functions, never by request builders.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from prompts.llm_as_judge_prompt import (
    CHOICE_CRITERIA, FEW_SHOT_CASE_IDS, INDIRECT_OR_UNDERSPECIFIED_FAILURE_PROMPT,
    JEV_JUDGE_PROMPT,
)

JEV_MODEL = "jev-1.13.0"
LLM_MODEL = "gpt-5-2025-08-07"
ENDPOINTS = {"jev": "https://api.typesafe.ai/v1/systemone",
             "llm": "https://api.openai.com/v1/chat/completions"}
DEFAULT_PROVIDERS = {"jev": "typesafe", "llm": "openai"}
PROVIDER_CONFIG = {
    ("jev", "typesafe"): (ENDPOINTS["jev"], JEV_MODEL, "TYPESAFE_API_KEY"),
    ("jev", "openrouter"): ("https://openrouter.ai/api/v1/systemone", "typesafe/jev-1.13", "OPENROUTER_API_KEY"),
    ("llm", "openrouter"): ("https://openrouter.ai/api/v1/chat/completions", "openai/gpt-5", "OPENROUTER_API_KEY"),
    ("llm", "openai"): (ENDPOINTS["llm"], LLM_MODEL, "OPENAI_API_KEY"),
}
# Frozen list-price estimate, not a reported charge. Unknown models stay unknown.
DEFAULT_PRICING = {"jev": {"provider": "typesafe", "model": "jev-1.13.0",
    "input_usd_per_million": 0.042, "output_usd_per_million": 0.0,
    "source": "https://docs.typesafe.ai/models", "checked_on": "2026-10-22"},
    "llm": {"provider": "openai", "model": "gpt-5-2025-08-07",
    "input_usd_per_million": 1.25, "cached_input_usd_per_million": 0.125,
    "output_usd_per_million": 10.0,
    "source": "https://developers.openai.com/api/docs/models/gpt-5",
    "checked_on": "2026-10-27"}}
LABELS = ("TRUE", "FALSE")
JUDGE_DATA_SOURCE = {
    "file": "llm_as_judge_indirect_underspecified.csv",
    "split_file": "judge_indirect_split.csv",
    "revision": "21d0f8c2339ecf64420fb0fffd08be34e60a8b0b",
    "label_column": "Indirect_or_Underspecified_Failure",
    "excluded_prompt_examples": sorted(FEW_SHOT_CASE_IDS),
}
CONFIDENCE_THRESHOLD = 0.7  # Fixed confidence cutoff; not selected from human labels.
POLICY = JEV_JUDGE_PROMPT
CRITERIA = CHOICE_CRITERIA
DEFAULT_PROMPTS = {"jev": JEV_JUDGE_PROMPT, "llm": INDIRECT_OR_UNDERSPECIFIED_FAILURE_PROMPT}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def load_dataset(data_dir, split="calibration"):
    """Load either the practice panel or the separate evaluation panel.

    Validate the complete source and split before selecting rows. All six worked
    examples remain in development and can never enter an evaluation panel.
    """
    if split not in {"calibration", "validation"}:
        raise ValueError("Choose calibration for practice or validation for evaluation.")
    data_dir = Path(data_dir)
    raw = read_csv(data_dir / JUDGE_DATA_SOURCE["file"])
    assignments = read_csv(data_dir / JUDGE_DATA_SOURCE["split_file"])
    expected_ids = {f"TEST_{number:03d}" for number in range(1, 301)}
    for name, records in (("reference dataset", raw), ("split file", assignments)):
        ids = [row.get("TEST_ID") for row in records]
        if len(ids) != len(set(ids)) or set(ids) != expected_ids:
            raise ValueError(f"The {name} must contain all 300 unique TEST_ID values.")

    references = {}
    for row in raw:
        label = row.get(JUDGE_DATA_SOURCE["label_column"], "").strip().upper()
        if (label not in LABELS or not row.get("USER_INPUT", "").strip()
                or not row.get("final_intent", "").strip()):
            raise ValueError(f"Invalid judge reference: {row['TEST_ID']}")
        references[row["TEST_ID"]] = {
            "id": row["TEST_ID"], "user_query": row["USER_INPUT"],
            "predicted_intent": row["final_intent"], "human_label": label,
        }

    panels = {"calibration": set(), "validation": set(), "development": set(), "prompt": set()}
    groups = {"development": set(), "validation": set()}
    for row in assignments:
        case_id = row["TEST_ID"]
        partition = row.get("split")
        flags = [row.get(field, "").strip().lower() for field in ("in_calibration", "in_prompt")]
        group = row.get("paraphrase_group", "").strip()
        if partition not in groups or any(flag not in {"true", "false"} for flag in flags) or not group:
            raise ValueError(f"Invalid split assignment: {case_id}")
        in_calibration, in_prompt = [flag == "true" for flag in flags]
        if in_prompt != (case_id in FEW_SHOT_CASE_IDS):
            raise ValueError("Prompt-example IDs differ from the frozen rubric.")
        if (in_prompt or in_calibration) and partition != "development":
            raise ValueError("Calibration and prompt examples must remain in development.")
        if in_prompt and in_calibration:
            raise ValueError("Prompt examples cannot be scored as calibration cases.")
        panels[partition].add(case_id)
        groups[partition].add(group)
        if in_calibration:
            panels["calibration"].add(case_id)
        if in_prompt:
            panels["prompt"].add(case_id)

    if (len(panels["development"]) != 150 or len(panels["validation"]) != 150
            or len(panels["calibration"]) != 21
            or panels["prompt"] != set(FEW_SHOT_CASE_IDS)
            or not groups["development"].isdisjoint(groups["validation"])):
        raise ValueError("The frozen split requires 150 development, 150 validation and 21 calibration cases, with disjoint paraphrase groups.")
    selected_ids = panels[split]
    # Keep the source order stable for manifests and row-level comparisons.
    return [row for case_id, row in references.items() if case_id in selected_ids]


def request_for(row, stage, model=None, prompt=None):
    """Explicit allowlist prevents human-label or reviewer-answer leakage."""
    state = {"USER_INPUT": row["user_query"], "final_intent": row["predicted_intent"]}
    if stage == "jev":
        return {"model": model or JEV_MODEL, "state": state,
                "questions": {"interpretation_failure": {"type": "choice", "instructions": prompt or DEFAULT_PROMPTS[stage],
                                                  "criteria": CRITERIA}}}
    if stage != "llm":
        raise ValueError("Unknown judge stage")
    return {"model": model or LLM_MODEL, "max_completion_tokens": 4096,
            "reasoning_effort": "medium",
            "messages": [{"role": "system", "content": prompt or DEFAULT_PROMPTS[stage]},
                         {"role": "user", "content": json.dumps(state, ensure_ascii=False)}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "interpretation_failure_judgment", "strict": True,
                "schema": {"type": "object", "additionalProperties": False,
                           "properties": {"label": {"type": "boolean"},
                                          "reasoning": {"type": "string"}},
                           "required": ["label", "reasoning"]}}}}


def finite_number(value, low=0, high=1):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and low <= value <= high)


def parse_jev(raw):
    answer = raw["answers"]["interpretation_failure"]
    probabilities = answer["probabilities"]
    if (not raw.get("model") or answer["type"] != "choice"
            or not isinstance(probabilities, dict) or set(probabilities) != set(LABELS)
            or not all(finite_number(p) for p in probabilities.values())
            or abs(sum(probabilities.values()) - 1) > 0.010000001
            or answer["choice"] not in LABELS
            or probabilities[answer["choice"]] < max(probabilities.values())
            or not finite_number(answer["confidence"])):
        raise ValueError("Invalid Jev choice response")
    return {"label": answer["choice"], "confidence": answer["confidence"],
            "probabilities": probabilities}


def parse_llm(raw):
    choice = raw["choices"][0]
    if choice["finish_reason"] != "stop" or choice["message"].get("refusal"):
        raise ValueError("Incomplete or refused LLM response")
    answer = json.loads(choice["message"]["content"])
    if (set(answer) != {"label", "reasoning"} or type(answer["label"]) is not bool
            or not isinstance(answer["reasoning"], str) or not answer["reasoning"].strip()):
        raise ValueError("Invalid LLM judgment")
    return {"label": "TRUE" if answer["label"] else "FALSE", "reason": answer["reasoning"]}


class JudgeClient:
    """Send every judgment live and write a separate audit record for each call.

    Records include each HTTP attempt, reported usage and elapsed time. They are
    never read as responses. Authorization headers are never written to disk or
    included in exceptions.
    """
    def __init__(self, run_dir, *, api_keys=None, providers=None,
                 models=None, pricing=None, prompts=None):
        self.run_dir = Path(run_dir)
        self.api_keys = api_keys or {}
        self.providers = dict(DEFAULT_PROVIDERS if providers is None else providers)
        config = {stage: PROVIDER_CONFIG[(stage, self.providers[stage])] for stage in ("jev", "llm")}
        self.endpoints = {stage: c[0] for stage, c in config.items()}
        self.models = models or {stage: c[1] for stage, c in config.items()}
        self.key_names = {stage: c[2] for stage, c in config.items()}
        self.pricing = DEFAULT_PRICING if pricing is None else pricing
        self.prompts = dict(DEFAULT_PROMPTS if prompts is None else prompts)
        self.new_requests = 0

    def configuration(self):
        """Public experiment metadata; never include API keys."""
        return {"providers": self.providers, "models": self.models, "pricing": self.pricing}

    def attempt_cost(self, attempt, stage):
        raw = attempt.get("response", {})
        usage = raw.get("usage", {})
        cost = usage.get("cost")
        if finite_number(cost, high=float("inf")):
            return cost, "reported"
        price = self.pricing.get(stage, {})
        if self.providers[stage] == "openai" and raw.get("service_tier") not in {None, "default"}:
            return None, "unknown"
        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
        output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
        if (price.get("provider") == self.providers[stage]
                and price.get("model") == self.models[stage] == raw.get("model")
                and finite_number(input_tokens, high=float("inf"))
                and finite_number(output_tokens, high=float("inf"))):
            cached = 0
            if "cached_input_usd_per_million" in price:
                cached = usage.get("prompt_tokens_details", {}).get("cached_tokens")
                if not finite_number(cached, high=input_tokens):
                    return None, "unknown"
            cost = ((input_tokens - cached) * price["input_usd_per_million"]
                    + cached * price.get("cached_input_usd_per_million", 0)
                    + output_tokens * price["output_usd_per_million"]) / 1_000_000
            return cost, "estimated"
        return None, "unknown"

    def call(self, row, stage):
        request = request_for(row, stage, self.models[stage], self.prompts[stage])
        request_hash = digest({"endpoint": self.endpoints[stage], "request": request})
        if not self.api_keys.get(stage):
            raise RuntimeError(f"Live {stage} requires {self.key_names[stage]}")
        call_id = str(uuid.uuid4())
        path = self.run_dir / "requests" / f"{stage}-{call_id}.json"
        audit = {"call_id": call_id, "request_hash": request_hash,
                 "id": row["id"], "stage": stage}
        try:
            entry = self._send(stage, request)
        except RuntimeError as exc:
            # Authentication/billing failures stop the run, but the failed live
            # request still has an audit record. _send exposes no provider body.
            save_json(path, {**audit, "endpoint": self.endpoints[stage],
                             "request": request, "error": str(exc),
                             "recorded_at": datetime.now(timezone.utc).isoformat()})
            raise
        entry.update(audit)
        save_json(path, entry)
        attempts = entry["attempts"]
        costs = [self.attempt_cost(a, stage) for a in attempts]
        result = {"id": row["id"], "stage": stage, "call_id": call_id,
                  "request_hash": request_hash,
                  "cost_usd": sum(c for c, source in costs if c is not None),
                  "reported_cost_usd": sum(c for c, source in costs if source == "reported"),
                  "estimated_cost_usd": sum(c for c, source in costs if source == "estimated"),
                  "estimated_cost_attempts": sum(source == "estimated" for c, source in costs),
                  "unknown_cost_attempts": sum(source == "unknown" for c, source in costs),
                  "unreported_cost_attempts": sum(source != "reported" for c, source in costs),
                  "attempts": len(attempts), "elapsed_seconds": sum(a["elapsed_seconds"] for a in attempts),
                  "input_tokens": sum(a.get("response", {}).get("usage", {}).get("prompt_tokens",
                                      a.get("response", {}).get("usage", {}).get("input_tokens", 0)) for a in attempts),
                  "output_tokens": sum(a.get("response", {}).get("usage", {}).get("completion_tokens",
                                       a.get("response", {}).get("usage", {}).get("output_tokens", 0)) for a in attempts),
                  "label": None, "error": None}
        raw = attempts[-1].get("response")
        if raw is None:
            result["error"] = attempts[-1]["error"]
            return result
        try:
            result.update(parse_jev(raw) if stage == "jev" else parse_llm(raw))
            result["model"] = raw.get("model")
        except (KeyError, IndexError, TypeError, ValueError):
            result["error"] = "Invalid judge output; unresolved"
        return result

    def _send(self, stage, request):
        attempts = []
        for attempt in range(3):
            req = urllib.request.Request(self.endpoints[stage], method="POST",
                  data=json.dumps(request).encode(), headers={"Authorization": "Bearer " + self.api_keys[stage],
                                                            "Content-Type": "application/json"})
            started = time.monotonic()
            self.new_requests += 1
            retry = False
            try:
                with urllib.request.urlopen(req, timeout=120) as response:
                    raw = json.load(response)
                if "error" in raw:
                    attempts.append({"error": "Provider error envelope", "response": raw,
                                     "elapsed_seconds": time.monotonic() - started})
                else:
                    attempts.append({"response": raw, "elapsed_seconds": time.monotonic() - started})
            except urllib.error.HTTPError as exc:
                if exc.code in {401, 402, 403}:
                    raise RuntimeError(f"HTTP {exc.code}: check {self.providers[stage]} key, credits and model access.") from None
                attempts.append({"error": f"HTTP {exc.code}", "elapsed_seconds": time.monotonic() - started})
                retry = exc.code == 429 or exc.code >= 500
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                attempts.append({"error": "Network error or invalid JSON", "elapsed_seconds": time.monotonic() - started})
                retry = True
            if not retry or attempt == 2:
                break
            time.sleep(2 ** attempt)
        return {"endpoint": self.endpoints[stage], "request": request, "attempts": attempts,
                "recorded_at": datetime.now(timezone.utc).isoformat()}


def batch(rows, function, workers=4):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return {row["id"]: value for row, value in zip(rows, pool.map(function, rows))}


def needs_llm(jev_result, threshold):
    if not finite_number(threshold):
        raise ValueError("Confidence threshold must be between 0 and 1")
    return (jev_result.get("error") is not None or jev_result.get("label") not in LABELS
            or not finite_number(jev_result.get("confidence"))
            or jev_result["confidence"] < threshold)


def combined(jev_result, llm_result=None):
    selected = llm_result if llm_result is not None else jev_result
    result = dict(selected, route="LLM" if llm_result is not None else "Jev")
    if llm_result is not None:
        for key in ("cost_usd", "unreported_cost_attempts", "attempts", "elapsed_seconds",
                    "input_tokens", "output_tokens", "reported_cost_usd", "estimated_cost_usd",
                    "estimated_cost_attempts", "unknown_cost_attempts"):
            result[key] = jev_result.get(key, 0) + llm_result.get(key, 0)
    return result


def run_cascade(rows, client, threshold, jev=None):
    """Actually call the LLM only for uncertain/invalid Jev judgments."""
    def one(row):
        first = jev[row["id"]] if jev is not None else client.call(row, "jev")
        return combined(first, client.call(row, "llm") if needs_llm(first, threshold) else None)
    return batch(rows, one)


def resolved(result):
    """A provider error or missing decision is not a FALSE judgment."""
    return result.get("label") in LABELS and not result.get("error")


def metrics(rows, predictions, method, score_ids=None):
    """Score resolved decisions; retain coverage and costs for every requested row.

    ``score_ids`` restricts quality metrics to a common panel for comparisons.
    Unresolved calls never become false negatives. With no scored cases, quality
    percentages are unavailable (None), while coverage and accounting remain valid.
    """
    if not rows:
        raise ValueError("Cannot score an empty dataset")
    requested_ids = {row["id"] for row in rows}
    if len(requested_ids) != len(rows) or any(row["human_label"] not in LABELS for row in rows):
        raise ValueError("Scoring requires unique IDs and TRUE/FALSE reference labels.")
    allowed_ids = requested_ids if score_ids is None else set(score_ids)
    if not allowed_ids.issubset(requested_ids):
        raise ValueError("Scoring IDs must belong to the requested dataset.")
    resolved_ids = {row["id"] for row in rows if resolved(predictions.get(row["id"], {}))}
    scored = [row for row in rows if row["id"] in resolved_ids & allowed_ids]
    truth = [row["human_label"] for row in scored]
    guesses = [predictions[row["id"]]["label"] for row in scored]
    tp = sum(t == "TRUE" and p == "TRUE" for t, p in zip(truth, guesses))
    fp = sum(t == "FALSE" and p == "TRUE" for t, p in zip(truth, guesses))
    fn = sum(t == "TRUE" and p == "FALSE" for t, p in zip(truth, guesses))
    tn = sum(t == "FALSE" and p == "FALSE" for t, p in zip(truth, guesses))
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    results = [predictions.get(row["id"], {}) for row in rows]
    return {"method": method, "n": len(rows),
            "positive_references": sum(row["human_label"] == "TRUE" for row in rows),
            "scored_n": len(scored), "scored_positive_references": truth.count("TRUE"),
            "resolved_n": len(resolved_ids), "coverage": len(resolved_ids) / len(rows),
            "accuracy": sum(t == p for t, p in zip(truth, guesses)) / len(scored) if scored else None,
            "precision_TRUE": precision if scored else None,
            "recall_TRUE": recall if scored else None, "f1_TRUE": f1 if scored else None,
            "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "unresolved": len(rows) - len(resolved_ids),
            "cost_usd": sum(r.get("cost_usd", 0) for r in results),
            "reported_cost_usd": sum(r.get("reported_cost_usd", 0) for r in results),
            "estimated_cost_usd": sum(r.get("estimated_cost_usd", 0) for r in results),
            "estimated_cost_attempts": sum(r.get("estimated_cost_attempts", 0) for r in results),
            "unknown_cost_attempts": sum(r.get("unknown_cost_attempts", 0) for r in results),
            "unreported_cost_attempts": sum(r.get("unreported_cost_attempts", 0) for r in results),
            "llm_routes": sum(r.get("route") == "LLM" for r in results)}


def comparison_metrics(rows, approaches):
    """Compare each approach on exactly the same successfully judged cases."""
    if not approaches:
        raise ValueError("Provide at least one judge approach to compare.")
    paired_ids = {row["id"] for row in rows}
    for predictions in approaches.values():
        paired_ids &= {row["id"] for row in rows if resolved(predictions.get(row["id"], {}))}
    return [metrics(rows, predictions, name, score_ids=paired_ids)
            for name, predictions in approaches.items()]


def confidence_bands(rows, jev, llm, threshold):
    """Measure fixes/regressions only where both judges returned valid decisions."""
    records = []
    for escalated, label in [(False, "Kept by Jev"), (True, "Sent to LLM")]:
        group = [r for r in rows if needs_llm(jev.get(r["id"], {}), threshold) == escalated]
        if not group:
            continue
        paired = [r for r in group if resolved(jev.get(r["id"], {})) and resolved(llm.get(r["id"], {}))]
        paired_ids = {r["id"] for r in paired}
        jev_summary = metrics(group, jev, "Jev", score_ids=paired_ids)
        llm_summary = metrics(group, llm, "LLM", score_ids=paired_ids)
        fixes = sum(jev[r["id"]]["label"] != r["human_label"] and llm[r["id"]]["label"] == r["human_label"] for r in paired)
        regressions = sum(jev[r["id"]]["label"] == r["human_label"] and llm[r["id"]]["label"] != r["human_label"] for r in paired)
        records.append({"group": label, "n": len(group), "paired_n": len(paired),
                        "Jev_unresolved": jev_summary["unresolved"],
                        "LLM_unresolved": llm_summary["unresolved"],
                        "Jev_accuracy": jev_summary["accuracy"],
                        "LLM_accuracy": llm_summary["accuracy"],
                        "LLM_fixes": fixes, "LLM_regressions": regressions})
    return records


def threshold_sweep(rows, jev, llm, thresholds):
    """Inspect candidate confidence gates on a labeled calibration panel.

    This diagnostic uses the already collected independent decisions to estimate
    each gate's accuracy and request cost; it never calls a provider or chooses
    the notebook's threshold. The live cascade separately makes its fallback
    requests. The paired correctness difference is cascade minus LLM accuracy.

    Following the paper's paired comparison, report a one-sided 95% lower bound
    mean(d) - 1.645 * sample_sd(d) / sqrt(n) and compare it with a -0.02 margin.
    This small-panel approximation is diagnostic, not a performance guarantee.
    A threshold is eligible only when every Jev and LLM decision is resolved and
    at least two cases are available. The paper gates on winning probability;
    these candidates gate on Jev's returned binary Choice confidence instead.
    """
    if not rows:
        raise ValueError("Cannot calibrate on an empty dataset")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)) or any(row["human_label"] not in LABELS for row in rows):
        raise ValueError("Calibration requires unique IDs and TRUE/FALSE reference labels.")
    candidates = list(thresholds)
    if any(not finite_number(threshold) for threshold in candidates):
        raise ValueError("Confidence thresholds must be between 0 and 1")
    paired = [row for row in rows if resolved(jev.get(row["id"], {}))
              and resolved(llm.get(row["id"], {}))]
    paired_ids = {row["id"] for row in paired}
    llm_summary = metrics(rows, llm, "LLM", score_ids=paired_ids)
    jev_resolved_n = sum(resolved(jev.get(case_id, {})) for case_id in ids)
    records = []
    for threshold in candidates:
        escalated = {case_id: needs_llm(jev.get(case_id, {}), threshold) for case_id in ids}
        predictions = {case_id: combined(jev.get(case_id, {}), llm.get(case_id, {})
                                         if escalated[case_id] else None) for case_id in ids}
        summary = metrics(rows, predictions, "Cascade", score_ids=paired_ids)
        differences = [int(predictions[row["id"]]["label"] == row["human_label"])
                       - int(llm[row["id"]]["label"] == row["human_label"])
                       for row in paired]
        mean = sum(differences) / len(differences) if differences else None
        lower_bound = None
        if len(differences) >= 2:
            sample_variance = sum((difference - mean) ** 2 for difference in differences) / (len(differences) - 1)
            lower_bound = mean - 1.645 * math.sqrt(sample_variance / len(differences))
        complete = len(paired) == len(rows) and len(paired) >= 2
        passes = lower_bound >= -0.02 if complete else None
        cascade_cost = summary["cost_usd"] if not summary["unknown_cost_attempts"] else None
        llm_cost = llm_summary["cost_usd"] if not llm_summary["unknown_cost_attempts"] else None
        savings = (1 - cascade_cost / llm_cost
                   if cascade_cost is not None and llm_cost is not None and llm_cost > 0 else None)
        accepted_n = sum(not value for value in escalated.values())
        records.append({
            "threshold": threshold, "n": len(rows), "accepted_n": accepted_n,
            "accepted_fraction": accepted_n / len(rows), "llm_routes": len(rows) - accepted_n,
            "jev_resolved_n": jev_resolved_n, "llm_resolved_n": llm_summary["resolved_n"],
            "paired_n": len(paired), "cascade_accuracy": summary["accuracy"],
            "precision_TRUE": summary["precision_TRUE"], "recall_TRUE": summary["recall_TRUE"],
            "FP": summary["FP"], "FN": summary["FN"],
            "llm_accuracy": llm_summary["accuracy"], "accuracy_difference": mean,
            "accuracy_difference_lcb_95": lower_bound, "noninferiority_margin": -0.02,
            "noninferiority_pass": passes, "cascade_cost_usd": cascade_cost,
            "llm_cost_usd": llm_cost, "cost_savings_fraction": savings,
            "unknown_cost_attempts": summary["unknown_cost_attempts"],
            "status": ("meets_margin" if passes else "below_margin") if complete else "insufficient_evidence",
        })
    return records


def always_false(rows):
    return {r["id"]: {"label": "FALSE"} for r in rows}
