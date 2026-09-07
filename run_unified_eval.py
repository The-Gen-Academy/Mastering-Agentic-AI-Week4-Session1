"""
Unified Evaluation Runner for CardWise AI
Ensures all 100 runs get feedback with consistent scoring logic matching local evaluation.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

# ── LangSmith Configuration ────────────────────────────────────────────────
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "CardWise_Eval"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"

sys.path.insert(0, str(Path(__file__).parent))

from src.graph import CardWiseGraph
from langsmith import Client
from langsmith.evaluation import EvaluationResult

client = Client()
app = CardWiseGraph()

PROJECT_NAME = "CardWise_Eval"
DATASET_NAME = "CardWise_Golden_Dataset"


def cardwise_agent_target(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Target function for evaluation."""
    query = inputs["query"]
    spend_profile = inputs.get("spend_profile", {})
    result = app.run(query, spend_profile=spend_profile)
    
    return {
        "intent": result.get("intent", ""),
        "card_filter": result.get("card_filter"),
        "retrieved_topics": [doc.metadata.get("topic", "") for doc in result.get("retrieved_docs", [])],
        "retrieved_context": result.get("retrieved_context", ""),
        "final_answer": result.get("final_answer", ""),
        "financial_data": result.get("financial_data"),
        "latency_ms": result.get("latency_ms", 0)
    }


# ── Evaluators with Same Logic as Local Evaluation ────────────────────────

def intent_accuracy_evaluator(run, example) -> EvaluationResult:
    """
    Intent accuracy — same as local evaluation.
    Score: 1.0 if match, 0.0 if mismatch.
    """
    predicted = run.outputs.get("intent", "")
    expected = example.outputs.get("expected_intent", "")
    is_correct = predicted == expected
    
    return EvaluationResult(
        key="intent_accuracy",
        score=1.0 if is_correct else 0.0,
        comment=f"Predicted: {predicted}, Expected: {expected}",
        metadata={
            "case_id": example.metadata.get("case_id", "unknown"),
            "predicted_intent": predicted,
            "expected_intent": expected
        }
    )


def card_filter_accuracy_evaluator(run, example) -> EvaluationResult:
    """
    Card filter accuracy — same as local evaluation.
    Normalizes None/"" as equivalent.
    """
    predicted = run.outputs.get("card_filter")
    expected = example.outputs.get("expected_card_filter")
    
    # Normalize: None and "" both mean "no specific card filter"
    norm_pred = predicted if predicted not in (None, "", "null") else None
    norm_exp = expected if expected not in (None, "", "null") else None
    is_correct = norm_pred == norm_exp
    
    return EvaluationResult(
        key="card_filter_accuracy",
        score=1.0 if is_correct else 0.0,
        comment=f"Predicted: {predicted}, Expected: {expected}",
        metadata={
            "case_id": example.metadata.get("case_id", "unknown"),
            "predicted_filter": predicted,
            "expected_filter": expected,
            "normalized_predicted": norm_pred,
            "normalized_expected": norm_exp
        }
    )


def context_recall_evaluator(run, example) -> EvaluationResult:
    """
    Context recall — same as local evaluation.
    Measures % of expected topics found in retrieved topics or context.
    """
    expected_topics = example.outputs.get("expected_retrieved_topics", [])
    retrieved_topics = run.outputs.get("retrieved_topics", [])
    retrieved_context = run.outputs.get("retrieved_context", "").lower()
    
    if not expected_topics:
        score = 1.0 if not retrieved_topics else 0.5
        comment = "No expected topics"
    else:
        found_topics = []
        for exp_topic in expected_topics:
            exp_lower = exp_topic.lower()
            in_topics = any(exp_lower in t.lower() for t in retrieved_topics)
            in_context = exp_lower in retrieved_context
            if in_topics or in_context:
                found_topics.append(exp_topic)
        
        score = len(found_topics) / len(expected_topics)
        comment = f"Found {len(found_topics)}/{len(expected_topics)} expected topics"
    
    return EvaluationResult(
        key="context_recall",
        score=score,
        comment=comment,
        metadata={
            "case_id": example.metadata.get("case_id", "unknown"),
            "expected_topics": expected_topics,
            "found_topics": found_topics if expected_topics else []
        }
    )


def answer_relevance_evaluator(run, example) -> EvaluationResult:
    """
    Answer relevance — same as local evaluation.
    Measures entity overlap and fact coverage.
    """
    answer = run.outputs.get("final_answer", "")
    query = example.inputs.get("query", "")
    reference_facts = example.outputs.get("reference_facts", [])
    expected_intent = example.outputs.get("expected_intent", "")
    
    if not answer or not answer.strip():
        return EvaluationResult(
            key="answer_relevance",
            score=0.0,
            comment="Empty answer",
            metadata={"case_id": example.metadata.get("case_id", "unknown")}
        )
    
    # Check for refusal patterns
    refusal_patterns = [
        "i cannot", "i can't", "i'm unable", "not permitted",
        "not allowed", "against my guidelines", "out of scope",
        "outside my domain", "cannot assist", "guardrail", "adversarial"
    ]
    is_refusal = any(p in answer.lower() for p in refusal_patterns)
    
    # Entity overlap
    card_words = ["amex", "platinum", "chase", "sapphire", "csr", "lounge", "hotel", "credit", "fee", "benefit"]
    query_entities = set(w for w in card_words if w in query.lower())
    answer_entities = set(w for w in card_words if w in answer.lower())
    entity_overlap = len(query_entities & answer_entities) / max(len(query_entities), 1)
    
    # Fact coverage
    facts_found = 0
    for fact in reference_facts:
        fact_keywords = [w.lower() for w in fact.split() if len(w) > 3]
        if any(kw in answer.lower() for kw in fact_keywords):
            facts_found += 1
    fact_coverage = facts_found / max(len(reference_facts), 1)
    
    # Score calculation
    if is_refusal:
        if expected_intent in ["adversarial", "out_of_scope"]:
            score = 0.8
            comment = "Appropriate refusal for out-of-scope/adversarial query"
        else:
            score = 0.2
            comment = "Inappropriate refusal for in-scope query"
    else:
        score = (entity_overlap * 0.4) + (fact_coverage * 0.6)
        comment = f"Entity overlap: {entity_overlap:.1%}, Fact coverage: {fact_coverage:.1%}"
    
    return EvaluationResult(
        key="answer_relevance",
        score=min(1.0, max(0.0, score)),
        comment=comment,
        metadata={
            "case_id": example.metadata.get("case_id", "unknown"),
            "is_refusal": is_refusal,
            "entity_overlap": entity_overlap,
            "fact_coverage": fact_coverage
        }
    )


def refusal_rate_evaluator(run, example) -> EvaluationResult:
    """
    Refusal rate — same as local evaluation.
    Measures appropriate refusal behavior for out-of-scope/adversarial queries.
    """
    answer = run.outputs.get("final_answer", "")
    expected_intent = example.outputs.get("expected_intent", "")
    
    refusal_patterns = [
        "i cannot", "i can't", "i'm unable", "not permitted",
        "not allowed", "against my guidelines", "out of scope",
        "outside my domain", "cannot assist", "guardrail", "adversarial"
    ]
    is_refusal = any(p in answer.lower() for p in refusal_patterns)
    should_refuse = expected_intent in ["adversarial", "out_of_scope"]
    
    if should_refuse and is_refusal:
        score = 1.0
        comment = "Correctly refused out-of-scope/adversarial query"
    elif should_refuse and not is_refusal:
        score = 0.0
        comment = "Failed to refuse adversarial/out-of-scope query"
    elif not should_refuse and is_refusal:
        score = 0.0
        comment = "Incorrectly refused in-scope query"
    else:
        score = 1.0
        comment = "Correctly answered in-scope query"
    
    return EvaluationResult(
        key="refusal_rate",
        score=score,
        comment=comment,
        metadata={
            "case_id": example.metadata.get("case_id", "unknown"),
            "is_refusal": is_refusal,
            "should_refuse": should_refuse
        }
    )


def token_usage_evaluator(run, example) -> EvaluationResult:
    """
    Token usage — same as local evaluation.
    Measures efficiency of token usage in response.
    """
    answer = run.outputs.get("final_answer", "")
    retrieved_context = run.outputs.get("retrieved_context", "")
    query = example.inputs.get("query", "")
    
    query_tokens = len(query) // 4
    answer_tokens = len(answer) // 4
    context_tokens = len(retrieved_context) // 4
    
    ideal_answer_range = (50, 500)
    ideal_context_range = (200, 2000)
    
    if ideal_answer_range[0] <= answer_tokens <= ideal_answer_range[1]:
        answer_score = 1.0
    elif answer_tokens < ideal_answer_range[0]:
        answer_score = 0.5
    else:
        answer_score = max(0.0, 1.0 - (answer_tokens - ideal_answer_range[1]) / 1000)
    
    if ideal_context_range[0] <= context_tokens <= ideal_context_range[1]:
        context_score = 1.0
    elif context_tokens < ideal_context_range[0]:
        context_score = 0.7
    else:
        context_score = max(0.0, 1.0 - (context_tokens - ideal_context_range[1]) / 2000)
    
    total_score = (answer_score * 0.6) + (context_score * 0.4)
    
    return EvaluationResult(
        key="token_usage",
        score=min(1.0, max(0.0, total_score)),
        comment=f"Answer: {answer_tokens} tokens, Context: {context_tokens} tokens",
        metadata={
            "case_id": example.metadata.get("case_id", "unknown"),
            "query_tokens": query_tokens,
            "answer_tokens": answer_tokens,
            "context_tokens": context_tokens
        }
    )


# ── Main Evaluation ────────────────────────────────────────────────────────

def run_unified_evaluation(
    experiment_name: str,
    description: str = None,
    max_concurrency: int = 1  # Lower concurrency to ensure all runs get evaluated
):
    """
    Run evaluation with all 100 cases getting feedback.
    Uses same scoring logic as local evaluation.
    """
    print("=" * 80)
    print(f"CardWise AI — Unified Evaluation")
    print("=" * 80)
    print(f"Experiment: {experiment_name}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Project: {PROJECT_NAME}")
    print(f"Concurrency: {max_concurrency} (lower = more reliable feedback)")
    print("=" * 80)
    
    evaluators = [
        intent_accuracy_evaluator,
        card_filter_accuracy_evaluator,
        context_recall_evaluator,
        answer_relevance_evaluator,
        refusal_rate_evaluator,
        token_usage_evaluator
    ]
    
    print(f"\nEvaluators: {[e.__name__ for e in evaluators]}")
    print("\nStarting evaluation...")
    print("This may take several minutes to ensure all 100 runs get feedback...")
    
    # Run evaluation with settings to maximize feedback coverage
    results = client.evaluate(
        cardwise_agent_target,
        data=DATASET_NAME,
        evaluators=evaluators,
        experiment_prefix=experiment_name,
        metadata={
            "prompt_version": "v2.0",
            "retrieval": "hybrid-bm25-faiss-rrf",
            "description": description or f"CardWise evaluation: {experiment_name}",
            "evaluation_version": "unified-v1"
        },
        max_concurrency=max_concurrency,
        num_repetitions=1,
        blocking=True,
        error_handling="log"  # Log errors but continue
    )
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Experiment: {results.experiment_name}")
    print(f"\nView in LangSmith UI:")
    print(f"https://smith.langchain.com/o/a9c4699a-b822-4097-b8a3-86a04ebb31d3/projects/p/{results.experiment_name}")
    print("=" * 80)
    
    return results


def add_feedback_to_existing_runs(experiment_name: str):
    """
    Manually add feedback to existing runs that don't have it.
    This ensures all 100 runs get scored even if client.evaluate() missed some.
    """
    print("=" * 80)
    print(f"Adding Feedback to Existing Runs: {experiment_name}")
    print("=" * 80)
    
    # Get all runs in the experiment
    runs = list(client.list_runs(project_name=experiment_name, limit=100))
    target_runs = [r for r in runs if r.name == "Target"]
    
    print(f"Total runs: {len(runs)}")
    print(f"Target runs: {len(target_runs)}")
    
    # Load golden dataset for expected values
    dataset_path = Path(__file__).parent / "data" / "golden_dataset.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        golden_cases = {case["id"]: case for case in json.load(f)}
    
    # Add feedback to runs that don't have it
    runs_updated = 0
    for run in target_runs:
        # Check if run already has feedback
        if hasattr(run, 'feedback_stats') and run.feedback_stats:
            continue
        
        # Get case_id from inputs
        query = run.inputs.get("query", "")
        case_id = None
        
        # Find matching case
        for cid, case in golden_cases.items():
            if case["query"] == query:
                case_id = cid
                break
        
        if not case_id:
            print(f"  Warning: Could not find case for query: {query[:50]}...")
            continue
        
        case = golden_cases[case_id]
        
        # Calculate scores
        predicted_intent = run.outputs.get("intent", "")
        expected_intent = case["expected_intent"]
        intent_score = 1.0 if predicted_intent == expected_intent else 0.0
        
        predicted_filter = run.outputs.get("card_filter")
        expected_filter = case["expected_card_filter"]
        norm_pred = predicted_filter if predicted_filter not in (None, "", "null") else None
        norm_exp = expected_filter if expected_filter not in (None, "", "null") else None
        filter_score = 1.0 if norm_pred == norm_exp else 0.0
        
        # Add feedback
        try:
            client.create_feedback(
                run_id=run.id,
                key="intent_accuracy",
                score=intent_score,
                comment=f"Predicted: {predicted_intent}, Expected: {expected_intent}"
            )
            
            client.create_feedback(
                run_id=run.id,
                key="card_filter_accuracy",
                score=filter_score,
                comment=f"Predicted: {predicted_filter}, Expected: {expected_filter}"
            )
            
            runs_updated += 1
            print(f"  ✓ Added feedback to {case_id}")
            
        except Exception as e:
            print(f"  ✗ Error adding feedback to {case_id}: {e}")
    
    print(f"\nUpdated {runs_updated} runs with feedback")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, help="Experiment name")
    parser.add_argument("--description", type=str, default=None, help="Experiment description")
    parser.add_argument("--concurrency", type=int, default=1, help="Max concurrency (lower = more reliable)")
    parser.add_argument("--add-feedback", action="store_true", help="Add feedback to existing runs")
    args = parser.parse_args()
    
    if args.add_feedback:
        add_feedback_to_existing_runs(args.name)
    else:
        run_unified_evaluation(args.name, args.description, args.concurrency)
