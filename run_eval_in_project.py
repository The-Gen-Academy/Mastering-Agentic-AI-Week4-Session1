"""
LangSmith Evaluation with Persistent Project
Runs all evaluations in the CardWise_Eval project for cross-experiment comparison.
"""

import os
import sys
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

# ── LangSmith Configuration ────────────────────────────────────────────────
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "CardWise_Eval"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"

sys.path.insert(0, str(Path(__file__).parent))

from src.graph import CardWiseGraph
from langsmith import Client
from langsmith.evaluation import EvaluationResult
from langsmith.schemas import TracerSession

# Initialize
client = Client()
app = CardWiseGraph()

PROJECT_NAME = "CardWise_Eval"
DATASET_NAME = "CardWise_Golden_Dataset"


def get_or_create_experiment_session(
    experiment_name: str,
    description: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> TracerSession:
    """
    Get or create an experiment session in the CardWise_Eval project.
    This ensures all runs go to the same project for comparison.
    """
    # Check if session exists
    sessions = list(client.list_sessions(project_name=PROJECT_NAME, name=experiment_name))
    
    if sessions:
        print(f"Using existing experiment session: {experiment_name}")
        return sessions[0]
    
    # Create new session
    print(f"Creating new experiment session: {experiment_name}")
    session = client.create_session(
        name=experiment_name,
        project_name=PROJECT_NAME,
        description=description or f"CardWise evaluation run: {experiment_name}",
        metadata=metadata or {}
    )
    return session


# ── Target Function ────────────────────────────────────────────────────────
def cardwise_agent_target(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Target function for LangSmith evaluate()."""
    query = inputs["query"]
    spend_profile = inputs.get("spend_profile", {})
    
    result = app.run(query, spend_profile=spend_profile)
    
    return {
        "intent": result.get("intent", ""),
        "card_filter": result.get("card_filter"),
        "retrieved_topics": [
            doc.metadata.get("topic", "")
            for doc in result.get("retrieved_docs", [])
        ],
        "retrieved_context": result.get("retrieved_context", ""),
        "final_answer": result.get("final_answer", ""),
        "financial_data": result.get("financial_data"),
        "latency_ms": result.get("latency_ms", 0)
    }


# ── Evaluator Functions ────────────────────────────────────────────────────
def intent_accuracy_evaluator(run, example) -> EvaluationResult:
    """Measures intent classification accuracy."""
    predicted = run.outputs.get("intent", "")
    expected = example.outputs.get("expected_intent", "")
    is_correct = predicted == expected
    
    return EvaluationResult(
        key="intent_accuracy",
        score=1.0 if is_correct else 0.0,
        comment=f"Predicted: {predicted}, Expected: {expected}",
        metadata={
            "predicted_intent": predicted,
            "expected_intent": expected,
            "case_id": example.metadata.get("case_id", "unknown")
        }
    )


def card_filter_accuracy_evaluator(run, example) -> EvaluationResult:
    """Measures card filter accuracy."""
    predicted = run.outputs.get("card_filter")
    expected = example.outputs.get("expected_card_filter")
    
    norm_pred = predicted if predicted not in (None, "", "null") else None
    norm_exp = expected if expected not in (None, "", "null") else None
    is_correct = norm_pred == norm_exp
    
    return EvaluationResult(
        key="card_filter_accuracy",
        score=1.0 if is_correct else 0.0,
        comment=f"Predicted: {predicted}, Expected: {expected}",
        metadata={
            "predicted_filter": predicted,
            "expected_filter": expected,
            "case_id": example.metadata.get("case_id", "unknown")
        }
    )


def answer_relevance_evaluator(run, example) -> EvaluationResult:
    """Measures if the answer addresses the user's query with relevant facts."""
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
        "outside my domain", "cannot assist", "guardrail"
    ]
    is_refusal = any(p in answer.lower() for p in refusal_patterns)
    
    # Check entity overlap
    query_entities = set()
    for word in ["amex", "platinum", "chase", "sapphire", "csr", "lounge", "hotel", "credit", "fee", "benefit"]:
        if word in query.lower():
            query_entities.add(word)
    
    answer_entities = set()
    for word in ["amex", "platinum", "chase", "sapphire", "csr", "lounge", "hotel", "credit", "fee", "benefit"]:
        if word in answer.lower():
            answer_entities.add(word)
    
    entity_overlap = len(query_entities & answer_entities) / max(len(query_entities), 1)
    
    # Check fact coverage
    facts_found = 0
    for fact in reference_facts:
        fact_keywords = [w.lower() for w in fact.split() if len(w) > 3]
        if any(kw in answer.lower() for kw in fact_keywords):
            facts_found += 1
    
    fact_coverage = facts_found / max(len(reference_facts), 1)
    
    # Calculate score
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


def context_recall_evaluator(run, example) -> EvaluationResult:
    """Measures retrieval recall."""
    expected_topics = example.outputs.get("expected_retrieved_topics", [])
    retrieved_topics = run.outputs.get("retrieved_topics", [])
    retrieved_context = run.outputs.get("retrieved_context", "")
    
    if not expected_topics:
        return EvaluationResult(
            key="context_recall",
            score=1.0 if not retrieved_topics else 0.5,
            comment="No expected topics",
            metadata={"case_id": example.metadata.get("case_id", "unknown")}
        )
    
    found_topics = []
    for exp_topic in expected_topics:
        exp_lower = exp_topic.lower()
        in_topics = any(exp_lower in t.lower() for t in retrieved_topics)
        in_context = exp_lower in retrieved_context.lower()
        if in_topics or in_context:
            found_topics.append(exp_topic)
    
    recall = len(found_topics) / len(expected_topics)
    
    return EvaluationResult(
        key="context_recall",
        score=recall,
        comment=f"Found {len(found_topics)}/{len(expected_topics)} expected topics",
        metadata={
            "case_id": example.metadata.get("case_id", "unknown"),
            "expected_count": len(expected_topics),
            "found_count": len(found_topics)
        }
    )


def refusal_rate_evaluator(run, example) -> EvaluationResult:
    """Measures appropriate refusal behavior."""
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
    """Measures efficiency of token usage."""
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
def run_evaluation_in_project(
    experiment_name: str,
    description: Optional[str] = None,
    max_concurrency: int = 4
):
    """
    Run evaluation in the CardWise_Eval project using an existing experiment session.
    This allows comparison across multiple experiments in the same project.
    """
    print("=" * 70)
    print(f"CardWise AI — Evaluation in Project: {PROJECT_NAME}")
    print("=" * 70)
    print(f"Experiment: {experiment_name}")
    print(f"Dataset: {DATASET_NAME}")
    print("=" * 70)
    
    # Create or get experiment session in the CardWise_Eval project
    session = get_or_create_experiment_session(
        experiment_name=experiment_name,
        description=description,
        metadata={
            "prompt_version": "v2.0",
            "retrieval": "hybrid-bm25-faiss-rrf",
            "timestamp": datetime.now().isoformat()
        }
    )
    
    print(f"\nSession ID: {session.id}")
    print(f"Project: {PROJECT_NAME}")
    
    # Define evaluators
    evaluators = [
        intent_accuracy_evaluator,
        card_filter_accuracy_evaluator,
        answer_relevance_evaluator,
        context_recall_evaluator,
        refusal_rate_evaluator,
        token_usage_evaluator
    ]
    
    # Run evaluation using the existing session
    print(f"\nStarting evaluation with {len(evaluators)} evaluators...")
    print(f"Evaluators: {[e.__name__ for e in evaluators]}")
    print()
    
    results = client.evaluate(
        cardwise_agent_target,
        data=DATASET_NAME,
        evaluators=evaluators,
        experiment=session.id,  # Use existing session in CardWise_Eval project
        max_concurrency=max_concurrency,
        blocking=True
    )
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Experiment: {experiment_name}")
    print(f"Session ID: {session.id}")
    print(f"Project: {PROJECT_NAME}")
    print(f"\nView all experiments in project at:")
    print(f"https://smith.langchain.com/o/a9c4699a-b822-4097-b8a3-86a04ebb31d3/projects/p/{PROJECT_NAME}")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, help="Experiment name (e.g., 'baseline-v1', 'improved-retrieval-v2')")
    parser.add_argument("--description", type=str, default=None, help="Experiment description")
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    
    run_evaluation_in_project(args.name, args.description, args.concurrency)
