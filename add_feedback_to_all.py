"""
Add feedback to ALL runs in an experiment.
This ensures 100% coverage, not just the 16% that client.evaluate() samples.
"""

import os
import sys
import json
from pathlib import Path

from langsmith import Client

client = Client(api_key=os.environ["LANGSMITH_API_KEY"])

# Load golden dataset for expected values
dataset_path = Path(__file__).parent / "data" / "golden_dataset.json"
with open(dataset_path, "r", encoding="utf-8") as f:
    golden_cases = {case["id"]: case for case in json.load(f)}


def calculate_scores(run_outputs, case):
    """Calculate all evaluation scores for a run."""
    # Intent accuracy
    predicted_intent = run_outputs.get("intent", "")
    expected_intent = case["expected_intent"]
    intent_score = 1.0 if predicted_intent == expected_intent else 0.0
    
    # Filter accuracy
    predicted_filter = run_outputs.get("card_filter")
    expected_filter = case["expected_card_filter"]
    norm_pred = predicted_filter if predicted_filter not in (None, "", "null") else None
    norm_exp = expected_filter if expected_filter not in (None, "", "null") else None
    filter_score = 1.0 if norm_pred == norm_exp else 0.0
    
    # Context recall
    expected_topics = case.get("expected_retrieved_topics", [])
    retrieved_topics = run_outputs.get("retrieved_topics", [])
    retrieved_context = run_outputs.get("retrieved_context", "").lower()
    
    if expected_topics:
        found = sum(1 for t in expected_topics 
                   if any(t.lower() in rt.lower() for rt in retrieved_topics) 
                   or t.lower() in retrieved_context)
        recall_score = found / len(expected_topics)
    else:
        recall_score = 1.0 if not retrieved_topics else 0.5
    
    # Answer relevance (simplified)
    answer = run_outputs.get("final_answer", "")
    reference_facts = case.get("reference_facts", [])
    
    if not answer:
        relevance_score = 0.0
    else:
        # Check refusal
        refusal_patterns = ["i cannot", "i can't", "not permitted", "out of scope", "guardrail"]
        is_refusal = any(p in answer.lower() for p in refusal_patterns)
        
        if is_refusal and expected_intent in ["adversarial", "out_of_scope"]:
            relevance_score = 0.8
        elif is_refusal:
            relevance_score = 0.2
        else:
            # Entity overlap
            card_words = ["amex", "platinum", "chase", "sapphire", "csr", "lounge", "hotel", "credit", "fee", "benefit"]
            query = case["query"]
            q_entities = set(w for w in card_words if w in query.lower())
            a_entities = set(w for w in card_words if w in answer.lower())
            entity_overlap = len(q_entities & a_entities) / max(len(q_entities), 1)
            
            # Fact coverage
            facts_found = sum(1 for f in reference_facts 
                            if any(kw in answer.lower() for kw in [w.lower() for w in f.split() if len(w) > 3]))
            fact_coverage = facts_found / max(len(reference_facts), 1)
            
            relevance_score = (entity_overlap * 0.4) + (fact_coverage * 0.6)
    
    # Refusal rate
    refusal_patterns = ["i cannot", "i can't", "not permitted", "out of scope", "guardrail", "adversarial"]
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
    
    # Token usage
    retrieved_context = run_outputs.get("retrieved_context", "")
    query_tokens = len(query) // 4
    answer_tokens = len(answer) // 4
    context_tokens = len(retrieved_context) // 4
    
    ideal_answer = (50, 500)
    ideal_context = (200, 2000)
    
    answer_score = 1.0 if ideal_answer[0] <= answer_tokens <= ideal_answer[1] else 0.5
    context_score = 1.0 if ideal_context[0] <= context_tokens <= ideal_context[1] else 0.7
    
    token_score = (answer_score * 0.6) + (context_score * 0.4)
    
    return {
        "intent_accuracy": intent_score,
        "card_filter_accuracy": filter_score,
        "context_recall": recall_score,
        "answer_relevance": relevance_score,
        "refusal_rate": refusal_score,
        "token_usage": token_score
    }


def add_feedback_to_all_runs(project_name):
    """Add feedback to all Target runs in an experiment."""
    print(f"Processing: {project_name}")
    print("=" * 80)
    
    runs = list(client.list_runs(project_name=project_name, limit=100))
    target_runs = [r for r in runs if r.name == "Target"]
    
    print(f"Total runs: {len(runs)}")
    print(f"Target runs: {len(target_runs)}")
    
    # Check which runs already have feedback
    runs_with_feedback = [r for r in target_runs if hasattr(r, 'feedback_stats') and r.feedback_stats]
    runs_without_feedback = [r for r in target_runs if not (hasattr(r, 'feedback_stats') and r.feedback_stats)]
    
    print(f"Runs with feedback: {len(runs_with_feedback)}")
    print(f"Runs without feedback: {len(runs_without_feedback)}")
    
    # Add feedback to runs that don't have it
    updated = 0
    errors = 0
    
    for run in runs_without_feedback:
        # Find matching golden case
        query = run.inputs.get("query", "")
        case_id = None
        
        for cid, case in golden_cases.items():
            if case["query"] == query:
                case_id = cid
                break
        
        if not case_id:
            print(f"  ⚠ No case found for: {query[:50]}...")
            continue
        
        case = golden_cases[case_id]
        scores = calculate_scores(run.outputs, case)
        
        # Add feedback for each metric
        try:
            for key, score in scores.items():
                client.create_feedback(
                    run_id=run.id,
                    key=key,
                    score=score,
                    comment=f"Auto-generated feedback for {case_id}"
                )
            updated += 1
            print(f"  ✓ {case_id}: intent={scores['intent_accuracy']:.0f}, filter={scores['card_filter_accuracy']:.0f}, recall={scores['context_recall']:.2f}")
        except Exception as e:
            errors += 1
            print(f"  ✗ Error for {case_id}: {e}")
    
    print(f"\n✓ Updated: {updated} runs")
    print(f"✗ Errors: {errors} runs")
    print("=" * 80)
    
    return updated, errors


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("project", help="Project/experiment name")
    args = parser.parse_args()
    
    add_feedback_to_all_runs(args.project)
