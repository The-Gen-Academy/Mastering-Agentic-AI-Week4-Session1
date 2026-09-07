"""
Batch Evaluation Script for CardWise AI
Runs all 100 cases and manually adds feedback to ensure 100% coverage in LangSmith.
"""

import os
import sys
import json
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional

# ── LangSmith Configuration ────────────────────────────────────────────────
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "CardWise_Eval"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"

sys.path.insert(0, str(Path(__file__).parent))

from src.graph import CardWiseGraph
from langsmith import Client

client = Client()
app = CardWiseGraph()

PROJECT_NAME = "CardWise_Eval"


def calculate_scores(result: Dict[str, Any], case: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate all evaluation scores for a single case.
    Returns scores matching the local evaluation logic.
    """
    # 1. Intent Accuracy
    predicted_intent = result.get("intent", "")
    expected_intent = case["expected_intent"]
    intent_score = 1.0 if predicted_intent == expected_intent else 0.0
    
    # 2. Card Filter Accuracy
    predicted_filter = result.get("card_filter")
    expected_filter = case["expected_card_filter"]
    
    # Normalize: None and "" both mean "no specific card filter"
    norm_pred = predicted_filter if predicted_filter not in (None, "", "null") else None
    norm_exp = expected_filter if expected_filter not in (None, "", "null") else None
    filter_score = 1.0 if norm_pred == norm_exp else 0.0
    
    # 3. Context Recall
    expected_topics = case.get("expected_retrieved_topics", [])
    retrieved_topics = [doc.metadata.get("topic", "") for doc in result.get("retrieved_docs", [])]
    retrieved_context = result.get("retrieved_context", "").lower()
    
    if expected_topics:
        found = 0
        for exp_topic in expected_topics:
            exp_lower = exp_topic.lower()
            # Check in retrieved topics
            in_topics = any(exp_lower in t.lower() for t in retrieved_topics)
            # Check in context text
            in_context = exp_lower in retrieved_context
            if in_topics or in_context:
                found += 1
        recall_score = found / len(expected_topics)
    else:
        recall_score = 1.0 if not retrieved_topics else 0.5
    
    # 4. Answer Relevance
    answer = result.get("final_answer", "")
    reference_facts = case.get("reference_facts", [])
    query = case["query"]
    
    if not answer or not answer.strip():
        relevance_score = 0.0
    else:
        # Check for refusal patterns
        refusal_patterns = [
            "i cannot", "i can't", "i'm unable", "not permitted",
            "not allowed", "against my guidelines", "out of scope",
            "outside my domain", "cannot assist", "guardrail", "adversarial"
        ]
        is_refusal = any(p in answer.lower() for p in refusal_patterns)
        
        if is_refusal:
            # Appropriate refusal for adversarial/out-of-scope
            if expected_intent in ["adversarial", "out_of_scope"]:
                relevance_score = 0.8
            else:
                relevance_score = 0.2
        else:
            # Entity overlap
            card_words = ["amex", "platinum", "chase", "sapphire", "csr", "lounge", "hotel", "credit", "fee", "benefit"]
            q_entities = set(w for w in card_words if w in query.lower())
            a_entities = set(w for w in card_words if w in answer.lower())
            entity_overlap = len(q_entities & a_entities) / max(len(q_entities), 1)
            
            # Fact coverage
            facts_found = 0
            for fact in reference_facts:
                fact_keywords = [w.lower() for w in fact.split() if len(w) > 3]
                if any(kw in answer.lower() for kw in fact_keywords):
                    facts_found += 1
            fact_coverage = facts_found / max(len(reference_facts), 1)
            
            relevance_score = (entity_overlap * 0.4) + (fact_coverage * 0.6)
    
    # 5. Refusal Rate
    refusal_patterns = [
        "i cannot", "i can't", "i'm unable", "not permitted",
        "not allowed", "against my guidelines", "out of scope",
        "outside my domain", "cannot assist", "guardrail", "adversarial"
    ]
    is_refusal = any(p in answer.lower() for p in refusal_patterns)
    should_refuse = expected_intent in ["adversarial", "out_of_scope"]
    
    if should_refuse and is_refusal:
        refusal_score = 1.0
    elif should_refuse and not is_refusal:
        refusal_score = 0.0
    elif not should_refuse and is_refusal:
        refusal_score = 0.0
    else:
        refusal_score = 1.0
    
    # 6. Token Usage
    retrieved_context = result.get("retrieved_context", "")
    query_tokens = len(query) // 4
    answer_tokens = len(answer) // 4
    context_tokens = len(retrieved_context) // 4
    
    ideal_answer = (50, 500)
    ideal_context = (200, 2000)
    
    if ideal_answer[0] <= answer_tokens <= ideal_answer[1]:
        answer_score = 1.0
    elif answer_tokens < ideal_answer[0]:
        answer_score = 0.5
    else:
        answer_score = max(0.0, 1.0 - (answer_tokens - ideal_answer[1]) / 1000)
    
    if ideal_context[0] <= context_tokens <= ideal_context[1]:
        context_score = 1.0
    elif context_tokens < ideal_context[0]:
        context_score = 0.7
    else:
        context_score = max(0.0, 1.0 - (context_tokens - ideal_context[1]) / 2000)
    
    token_score = (answer_score * 0.6) + (context_score * 0.4)
    
    return {
        "intent_accuracy": intent_score,
        "card_filter_accuracy": filter_score,
        "context_recall": recall_score,
        "answer_relevance": relevance_score,
        "refusal_rate": refusal_score,
        "token_usage": token_score
    }


def run_batch_evaluation(
    experiment_name: str,
    description: Optional[str] = None,
    limit: Optional[int] = None
):
    """
    Run batch evaluation on all 100 cases with manual feedback.
    Ensures 100% coverage, not just the 16% sampling from client.evaluate().
    """
    print("=" * 80)
    print(f"CardWise AI — Batch Evaluation with Manual Feedback")
    print("=" * 80)
    print(f"Experiment: {experiment_name}")
    print(f"Project: {PROJECT_NAME}")
    print(f"Description: {description or 'N/A'}")
    print("=" * 80)
    
    # Load golden dataset
    dataset_path = Path(__file__).parent / "data" / "golden_dataset.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        golden_cases = json.load(f)
    
    if limit:
        golden_cases = golden_cases[:limit]
    
    print(f"\nTotal cases: {len(golden_cases)}")
    print("Running evaluation...")
    
    # Track results
    all_scores = []
    run_ids = []
    
    for i, case in enumerate(golden_cases):
        case_id = case["id"]
        query = case["query"]
        spend_profile = case.get("expected_spend_profile", {})
        
        # Run the agent
        start_time = time.time()
        result = app.run(query, spend_profile=spend_profile)
        latency_ms = (time.time() - start_time) * 1000
        
        # Calculate scores
        scores = calculate_scores(result, case)
        all_scores.append(scores)
        
        # Create run in LangSmith
        run_id = uuid.uuid4()
        run_name = f"{experiment_name}_{case_id}"
        
        try:
            client.create_run(
                id=run_id,
                name=run_name,
                run_type="chain",
                project_name=PROJECT_NAME,
                inputs={"query": query, "spend_profile": spend_profile},
                outputs={
                    "intent": result.get("intent", ""),
                    "card_filter": result.get("card_filter"),
                    "retrieved_topics": [doc.metadata.get("topic", "") for doc in result.get("retrieved_docs", [])],
                    "retrieved_context": result.get("retrieved_context", ""),
                    "final_answer": result.get("final_answer", ""),
                    "financial_data": result.get("financial_data")
                },
                extra={
                    "metadata": {
                        "case_id": case_id,
                        "experiment": experiment_name,
                        "prompt_version": "v2.0",
                        "true_category": case["expected_intent"],
                        "predicted_category": result.get("intent", ""),
                        "correctness": {
                            "intent_match": scores["intent_accuracy"] == 1.0,
                            "filter_match": scores["card_filter_accuracy"] == 1.0,
                            "overall": scores["intent_accuracy"] == 1.0 and scores["card_filter_accuracy"] == 1.0
                        },
                        "timing": {
                            "latency_ms": latency_ms,
                            "start_time_unix": start_time
                        }
                    }
                },
                start_time=start_time,
                end_time=time.time()
            )
            
            # Add feedback for each metric
            for key, score in scores.items():
                client.create_feedback(
                    run_id=run_id,
                    key=key,
                    score=score,
                    comment=f"Batch evaluation score for {case_id}"
                )
            
            run_ids.append(str(run_id))
            
            # Log progress
            status = "PASS" if scores["intent_accuracy"] == 1.0 else "FAIL"
            print(f"[{i+1}/{len(golden_cases)}] {case_id}: {status} "
                  f"(intent={scores['intent_accuracy']:.0f}, filter={scores['card_filter_accuracy']:.0f}, "
                  f"recall={scores['context_recall']:.2f})")
            
        except Exception as e:
            print(f"[{i+1}/{len(golden_cases)}] {case_id}: ERROR - {e}")
    
    # Calculate aggregate metrics
    print("\n" + "=" * 80)
    print("AGGREGATE METRICS")
    print("=" * 80)
    
    metric_names = ["intent_accuracy", "card_filter_accuracy", "context_recall", 
                   "answer_relevance", "refusal_rate", "token_usage"]
    
    print(f"{'Metric':<25} {'Avg Score':>12} {'Min':>8} {'Max':>8}")
    print("-" * 80)
    
    for metric in metric_names:
        scores_list = [s[metric] for s in all_scores]
        avg = sum(scores_list) / len(scores_list)
        min_score = min(scores_list)
        max_score = max(scores_list)
        print(f"{metric:<25} {avg:>11.1%} {min_score:>7.1%} {max_score:>7.1%}")
    
    print("-" * 80)
    print(f"Total runs created: {len(run_ids)}")
    print(f"Runs with feedback: {len(run_ids)} (100%)")
    print("=" * 80)
    
    print(f"\nView in LangSmith UI:")
    print(f"https://smith.langchain.com/o/a9c4699a-b822-4097-b8a3-86a04ebb31d3/projects/p/{PROJECT_NAME}")
    print("=" * 80)
    
    return all_scores, run_ids


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, help="Experiment name")
    parser.add_argument("--description", type=str, default=None, help="Experiment description")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of cases")
    args = parser.parse_args()
    
    run_batch_evaluation(args.name, args.description, args.limit)
