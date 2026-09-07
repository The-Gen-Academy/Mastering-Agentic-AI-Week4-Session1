"""
Traced Evaluation Runner for CardWise AI
Runs the agent with LangSmith tracing, collects metadata, and executes evaluators.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# ── LangSmith Configuration ────────────────────────────────────────────────
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "CardWise_Eval"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.graph import CardWiseGraph
from src.prompts import ROUTER_SYSTEM_PROMPT, ADVISOR_SYSTEM_PROMPT

# LangSmith imports
from langsmith import Client
from langsmith.run_helpers import traceable

# Initialize
client = Client()
app = CardWiseGraph()


@dataclass
class EvaluationResult:
    """Result from a single evaluation run."""
    case_id: str
    query: str
    intent: str
    card_filter: Optional[str]
    retrieved_topics: List[str]
    final_answer: str
    financial_data: Optional[Dict[str, Any]]
    latency_ms: float
    error: Optional[str] = None


@traceable(name="cardwise_agent_run", run_type="chain")
def run_agent_with_tracing(query: str, spend_profile: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Run the CardWise agent with full LangSmith tracing.
    Returns the complete state including all intermediate outputs.
    """
    return app.run(query, spend_profile=spend_profile)


@traceable(name="cardwise_eval_case", run_type="chain")
def evaluate_single_case(
    case: Dict[str, Any],
    run_name: Optional[str] = None
) -> EvaluationResult:
    """
    Evaluate a single golden dataset case with tracing.
    Collects all required metadata for LangSmith.
    """
    case_id = case["case_id"]
    query = case["query"]
    spend_profile = case.get("spend_profile", {})
    expected_intent = case.get("expected_intent", "")
    expected_filter = case.get("expected_card_filter")
    
    # Normalize expected filter
    if expected_filter in ("", "null"):
        expected_filter = None
    
    start_time = time.time()
    error = None
    
    try:
        # Run agent with tracing
        result = run_agent_with_tracing(query, spend_profile)
        
        # Extract topics from retrieved docs
        retrieved_topics = [
            doc.metadata.get("topic", "")
            for doc in result.get("retrieved_docs", [])
        ]
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Calculate correctness
        actual_intent = result.get("intent", "")
        actual_filter = result.get("card_filter")
        
        # Normalize actual filter for comparison
        norm_actual_filter = actual_filter if actual_filter not in ("", "null") else None
        
        intent_correct = (actual_intent == expected_intent)
        filter_correct = (norm_actual_filter == expected_filter)
        
        eval_result = EvaluationResult(
            case_id=case_id,
            query=query,
            intent=actual_intent,
            card_filter=actual_filter,
            retrieved_topics=retrieved_topics,
            final_answer=result.get("final_answer", ""),
            financial_data=result.get("financial_data"),
            latency_ms=latency_ms,
            error=None
        )
        
        # Return with metadata for LangSmith trace
        return {
            "result": eval_result,
            "metadata": {
                "case_id": case_id,
                "run_name": run_name or f"eval_{case_id}",
                "prompt_version": "v2.0",
                "true_category": expected_intent,
                "predicted_category": actual_intent,
                "correctness": {
                    "intent_match": intent_correct,
                    "filter_match": filter_correct,
                    "overall": intent_correct and filter_correct
                },
                "timing": {
                    "latency_ms": latency_ms,
                    "start_time": start_time
                },
                "error": None
            }
        }
        
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        error_result = EvaluationResult(
            case_id=case_id,
            query=query,
            intent="error",
            card_filter=None,
            retrieved_topics=[],
            final_answer="",
            financial_data=None,
            latency_ms=latency_ms,
            error=str(e)
        )
        
        return {
            "result": error_result,
            "metadata": {
                "case_id": case_id,
                "run_name": run_name or f"eval_{case_id}",
                "prompt_version": "v2.0",
                "true_category": expected_intent,
                "predicted_category": "error",
                "correctness": {
                    "intent_match": False,
                    "filter_match": False,
                    "overall": False
                },
                "timing": {
                    "latency_ms": latency_ms,
                    "start_time": start_time
                },
                "error": str(e)
            }
        }


def run_traced_evaluation(dataset_path: Optional[Path] = None, limit: Optional[int] = None):
    """
    Run evaluation on golden dataset with full LangSmith tracing.
    """
    # Load golden dataset
    if dataset_path is None:
        dataset_path = Path(__file__).parent / "data" / "golden_dataset.json"
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        golden_cases = json.load(f)
    
    if limit:
        golden_cases = golden_cases[:limit]
    
    print("=" * 70)
    print(f"CardWise AI — Traced Evaluation")
    print(f"Project: {os.environ['LANGSMITH_PROJECT']}")
    print(f"Cases: {len(golden_cases)}")
    print("=" * 70)
    
    results = []
    
    for i, case in enumerate(golden_cases):
        case_id = case["id"]
        
        # Prepare case data with expected values for metadata
        case_data = {
            "case_id": case_id,
            "query": case["query"],
            "spend_profile": case.get("expected_spend_profile", {}),
            "expected_intent": case.get("expected_intent", ""),
            "expected_card_filter": case.get("expected_card_filter")
        }
        
        # Run with tracing
        eval_output = evaluate_single_case(case_data, run_name=f"eval_{case_id}")
        result = eval_output["result"]
        metadata = eval_output["metadata"]
        results.append(result)
        
        # Log progress with correctness
        status = "ERROR" if result.error else ("PASS" if metadata["correctness"]["overall"] else "FAIL")
        print(f"[{i+1}/{len(golden_cases)}] {case_id}: {status} ({result.latency_ms:.0f}ms)")
        print(f"    Intent: {result.intent} (expected: {metadata['true_category']}) | "
              f"Filter: {result.card_filter} | "
              f"Correct: {metadata['correctness']['overall']}")
    
    # Summary
    print("\n" + "=" * 70)
    print("TRACED EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Total cases: {len(results)}")
    print(f"Successful: {sum(1 for r in results if not r.error)}")
    print(f"Errors: {sum(1 for r in results if r.error)}")
    print(f"Avg latency: {sum(r.latency_ms for r in results)/len(results):.0f}ms")
    print(f"\nView traces at: https://smith.langchain.com/")
    print(f"Project: {os.environ['LANGSMITH_PROJECT']}")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of cases to run")
    parser.add_argument("--dataset", type=str, default=None, help="Path to dataset JSON")
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset) if args.dataset else None
    run_traced_evaluation(dataset_path, args.limit)
