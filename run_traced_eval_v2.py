"""
Enhanced Traced Evaluation Runner with Proper LangSmith Metadata
Uses client.create_run() to explicitly set trace metadata.
"""

import os
import sys
import json
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

# ── LangSmith Configuration ────────────────────────────────────────────────
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "CardWise_Eval"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"

sys.path.insert(0, str(Path(__file__).parent))

from src.graph import CardWiseGraph
from langsmith import Client

client = Client()
app = CardWiseGraph()


@dataclass
class EvaluationResult:
    case_id: str
    query: str
    intent: str
    card_filter: Optional[str]
    retrieved_topics: List[str]
    final_answer: str
    financial_data: Optional[Dict[str, Any]]
    latency_ms: float
    error: Optional[str] = None


def evaluate_single_case_with_metadata(
    case: Dict[str, Any],
    run_name: Optional[str] = None
) -> tuple[EvaluationResult, str]:
    """
    Evaluate a single case and create a LangSmith run with proper metadata.
    Returns (result, run_id).
    """
    case_id = case["case_id"]
    query = case["query"]
    spend_profile = case.get("spend_profile", {})
    expected_intent = case.get("expected_intent", "")
    expected_filter = case.get("expected_card_filter")
    
    if expected_filter in ("", "null"):
        expected_filter = None
    
    run_name = run_name or f"eval_{case_id}"
    run_id = uuid.uuid4()
    
    start_time = time.time()
    error = None
    result = None
    
    try:
        # Run the agent
        agent_result = app.run(query, spend_profile)
        
        retrieved_topics = [
            doc.metadata.get("topic", "")
            for doc in agent_result.get("retrieved_docs", [])
        ]
        
        latency_ms = (time.time() - start_time) * 1000
        
        actual_intent = agent_result.get("intent", "")
        actual_filter = agent_result.get("card_filter")
        norm_actual_filter = actual_filter if actual_filter not in ("", "null") else None
        
        intent_correct = (actual_intent == expected_intent)
        filter_correct = (norm_actual_filter == expected_filter)
        
        result = EvaluationResult(
            case_id=case_id,
            query=query,
            intent=actual_intent,
            card_filter=actual_filter,
            retrieved_topics=retrieved_topics,
            final_answer=agent_result.get("final_answer", ""),
            financial_data=agent_result.get("financial_data"),
            latency_ms=latency_ms,
            error=None
        )
        
        # Create LangSmith run with metadata
        metadata = {
            "case_id": case_id,
            "run_name": run_name,
            "prompt_version": "v2.0",
            "true_category": expected_intent,
            "predicted_category": actual_intent,
            "true_filter": expected_filter,
            "predicted_filter": actual_filter,
            "correctness": {
                "intent_match": intent_correct,
                "filter_match": filter_correct,
                "overall": intent_correct and filter_correct
            },
            "timing": {
                "latency_ms": latency_ms,
                "start_time_unix": start_time
            }
        }
        
        # Create the run in LangSmith
        client.create_run(
            id=run_id,
            name=run_name,
            run_type="chain",
            project_name=os.environ["LANGSMITH_PROJECT"],
            inputs={"query": query, "spend_profile": spend_profile},
            outputs={
                "intent": actual_intent,
                "card_filter": actual_filter,
                "retrieved_topics": retrieved_topics,
                "final_answer": agent_result.get("final_answer", "")[:500]  # Truncate for display
            },
            extra={"metadata": metadata},
            start_time=start_time,
            end_time=time.time(),
            error=None
        )
        
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        error_msg = str(e)
        
        result = EvaluationResult(
            case_id=case_id,
            query=query,
            intent="error",
            card_filter=None,
            retrieved_topics=[],
            final_answer="",
            financial_data=None,
            latency_ms=latency_ms,
            error=error_msg
        )
        
        # Create error run in LangSmith
        metadata = {
            "case_id": case_id,
            "run_name": run_name,
            "prompt_version": "v2.0",
            "true_category": expected_intent,
            "predicted_category": "error",
            "correctness": {"intent_match": False, "filter_match": False, "overall": False},
            "timing": {"latency_ms": latency_ms, "start_time_unix": start_time},
            "error": error_msg
        }
        
        client.create_run(
            id=run_id,
            name=run_name,
            run_type="chain",
            project_name=os.environ["LANGSMITH_PROJECT"],
            inputs={"query": query, "spend_profile": spend_profile},
            outputs={"error": error_msg},
            extra={"metadata": metadata},
            start_time=start_time,
            end_time=time.time(),
            error=error_msg
        )
    
    return result, str(run_id)


def run_traced_evaluation(
    dataset_path: Optional[Path] = None,
    limit: Optional[int] = None,
    experiment_name: str = "baseline",
    experiment_description: Optional[str] = None
):
    """Run evaluation with LangSmith tracing and metadata in CardWise_Eval project."""
    if dataset_path is None:
        dataset_path = Path(__file__).parent / "data" / "golden_dataset.json"
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        golden_cases = json.load(f)
    
    if limit:
        golden_cases = golden_cases[:limit]
    
    print("=" * 70)
    print(f"CardWise AI — Traced Evaluation with Metadata")
    print(f"Project: {os.environ['LANGSMITH_PROJECT']}")
    print(f"Experiment: {experiment_name}")
    print(f"Cases: {len(golden_cases)}")
    print("=" * 70)
    
    results = []
    run_ids = []
    
    for i, case in enumerate(golden_cases):
        case_id = case["id"]
        
        case_data = {
            "case_id": case_id,
            "query": case["query"],
            "spend_profile": case.get("expected_spend_profile", {}),
            "expected_intent": case.get("expected_intent", ""),
            "expected_card_filter": case.get("expected_card_filter")
        }
        
        result, run_id = evaluate_single_case_with_metadata(case_data, run_name=f"{experiment_name}_{case_id}")
        results.append(result)
        run_ids.append(run_id)
        
        status = "ERROR" if result.error else ("PASS" if result.intent == case_data["expected_intent"] else "FAIL")
        print(f"[{i+1}/{len(golden_cases)}] {case_id}: {status} ({result.latency_ms:.0f}ms) | run_id={run_id[:8]}...")
    
    # Summary
    correct = sum(1 for r, c in zip(results, golden_cases) 
                  if r.intent == c.get("expected_intent") and not r.error)
    
    print("\n" + "=" * 70)
    print("TRACED EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Total cases: {len(results)}")
    print(f"Correct: {correct}/{len(results)} ({correct/len(results)*100:.1f}%)")
    print(f"Errors: {sum(1 for r in results if r.error)}")
    print(f"Avg latency: {sum(r.latency_ms for r in results)/len(results):.0f}ms")
    print(f"\nView all experiments in project at:")
    print(f"https://smith.langchain.com/o/a9c4699a-b822-4097-b8a3-86a04ebb31d3/projects/p/{os.environ['LANGSMITH_PROJECT']}")
    print("=" * 70)
    
    return results, run_ids


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of cases")
    parser.add_argument("--dataset", type=str, default=None, help="Path to dataset JSON")
    parser.add_argument("--name", type=str, default="baseline", help="Experiment name (e.g., 'baseline-v1', 'improved-retrieval')")
    parser.add_argument("--description", type=str, default=None, help="Experiment description")
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset) if args.dataset else None
    run_traced_evaluation(dataset_path, args.limit, args.name, args.description)
