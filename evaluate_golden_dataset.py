"""
CardWise AI — Golden Dataset Evaluation Suite
Evaluates Intent Accuracy, Card Filter Accuracy, Topic Retrieval Recall,
Recommendation Accuracy, and Fact Groundedness against the Golden Dataset.
"""

import sys
import json
import csv
import re
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).parent))

from src.graph import CardWiseGraph


def evaluate_cardwise_agent(
    dataset_path: Optional[Path] = None,
    output_report_path: Optional[Path] = None
):
    data_dir = Path(__file__).parent / "data"
    dataset_file = dataset_path or (data_dir / "golden_dataset.json")
    report_file = output_report_path or (data_dir / "evaluation_results.json")

    with open(dataset_file, "r", encoding="utf-8") as f:
        golden_cases = json.load(f)

    print("=" * 80)
    print(f"CardWise AI — Golden Dataset Evaluation Suite")
    print(f"Evaluating {len(golden_cases)} Ground-Truth Test Cases")
    print("=" * 80)

    app = CardWiseGraph()

    total_cases = len(golden_cases)
    intent_correct = 0
    filter_correct = 0
    retrieval_hits = 0
    recommendation_correct = 0
    total_facts_found = 0
    total_facts_expected = 0

    results_detail = []

    for item in golden_cases:
        case_id = item["id"]
        query = item["query"]
        exp_intent = item["expected_intent"]
        exp_filter = item["expected_card_filter"]
        exp_spend = item["expected_spend_profile"]
        exp_topics = item["expected_retrieved_topics"]
        exp_winner = item["expected_winner"]
        ref_facts = item["reference_facts"]

        # Run CardWise LangGraph Agent
        res = app.run(query, spend_profile=exp_spend)

        # 1. Intent Check
        act_intent = res["intent"]
        is_intent_match = (act_intent == exp_intent)
        if is_intent_match:
            intent_correct += 1

        # 2. Card Filter Check
        act_filter = res["card_filter"]
        # Normalize: None and "" both mean "no specific card filter"
        norm_act = act_filter if act_filter not in (None, "", "null") else None
        norm_exp = exp_filter if exp_filter not in (None, "", "null") else None
        is_filter_match = (norm_act == norm_exp)
        if is_filter_match:
            filter_correct += 1

        # 3. Topic Retrieval Recall Check
        retrieved_topics = [
            doc.metadata.get("topic", "")
            for doc in res.get("retrieved_docs", [])
        ]
        retrieved_context_text = res.get("retrieved_context", "").lower()
        
        hit_topics = [
            topic for topic in exp_topics
            if any(topic.lower() in t.lower() for t in retrieved_topics) or topic.lower() in retrieved_context_text
        ]
        topic_recall = len(hit_topics) / len(exp_topics) if exp_topics else 1.0
        if topic_recall >= 0.5:
            retrieval_hits += 1

        # 4. Recommendation Accuracy Check
        act_winner = "N/A"
        if res.get("financial_data"):
            act_winner = res["financial_data"].get("recommendation", "N/A")
        
        is_winner_match = True
        if exp_winner != "N/A":
            is_winner_match = (exp_winner.lower() in act_winner.lower())
            if is_winner_match:
                recommendation_correct += 1

        # 5. Fact Groundedness Check
        answer_text = res.get("final_answer", "").lower()
        facts_found = 0
        for fact in ref_facts:
            # Extract key information from the fact
            fact_lower = fact.lower()
            
            # Strategy 1: Check for key entities (dollar amounts, card names, benefit names)
            key_entities = [
                # Dollar amounts
                *re.findall(r'\$\d+(?:,\d{3})*(?:\.\d{2})?', fact_lower),
                # Card names
                *re.findall(r'\b(?:amex|platinum|chase|sapphire|csr)\b', fact_lower),
                # Benefit names
                *re.findall(r'\b(?:lounge|hotel|credit|fee|benefit|protection|insurance|doordash|resy|lululemon|clear|global entry|tsa precheck)\b', fact_lower),
            ]
            
            # Strategy 2: Check for keyword overlap
            keywords = [w.lower() for w in fact.split() if len(w) > 3]
            keyword_matches = sum(1 for kw in keywords if kw in answer_text)
            keyword_score = keyword_matches / max(len(keywords), 1)
            
            # Strategy 3: Check for entity overlap
            entity_matches = sum(1 for entity in key_entities if entity in answer_text)
            entity_score = entity_matches / max(len(key_entities), 1) if key_entities else 0
            
            # Strategy 4: Check for semantic similarity (simple version)
            # Check if the main concept of the fact is present
            main_concepts = []
            if "reject" in fact_lower or "refuse" in fact_lower or "do not" in fact_lower:
                main_concepts.extend(["not", "cannot", "refuse", "reject"])
            if "correct" in fact_lower or "right" in fact_lower:
                main_concepts.extend(["correct", "right", "accurate"])
            if "dataset" in fact_lower or "data" in fact_lower:
                main_concepts.extend(["dataset", "data", "provided", "available"])
            if "clarify" in fact_lower or "clarification" in fact_lower:
                main_concepts.extend(["clarify", "clarification", "clear"])
            if "ambiguous" in fact_lower or "multiple" in fact_lower:
                main_concepts.extend(["ambiguous", "multiple", "several", "various"])
            
            concept_matches = sum(1 for concept in main_concepts if concept in answer_text)
            concept_score = concept_matches / max(len(main_concepts), 1) if main_concepts else 0
            
            # Combined score — fact is "found" if any strategy matches well
            # Use the best score from keyword, entity, or concept matching
            best_score = max(keyword_score, entity_score, concept_score)
            
            # Lower threshold for adversarial cases (refusals)
            threshold = 0.2 if res.get("intent") == "adversarial" else 0.3
            
            if best_score >= threshold:
                facts_found += 1
        
        total_facts_found += facts_found
        total_facts_expected += len(ref_facts)

        case_summary = {
            "id": case_id,
            "query": query,
            "intent_pass": is_intent_match,
            "actual_intent": act_intent,
            "expected_intent": exp_intent,
            "filter_pass": is_filter_match,
            "actual_filter": act_filter,
            "expected_filter": exp_filter,
            "retrieval_recall": topic_recall,
            "winner_pass": is_winner_match,
            "actual_winner": act_winner,
            "expected_winner": exp_winner,
            "facts_recalled": f"{facts_found}/{len(ref_facts)}"
        }
        results_detail.append(case_summary)

        print(
            f"[{case_id}] Intent: {'PASS' if is_intent_match else 'FAIL'} | "
            f"Filter: {'PASS' if is_filter_match else 'FAIL'} | "
            f"Retrieval Recall: {topic_recall * 100:.0f}% | "
            f"Fact Match: {facts_found}/{len(ref_facts)}"
        )

    # Metrics Summary
    intent_accuracy = (intent_correct / total_cases) * 100
    filter_accuracy = (filter_correct / total_cases) * 100
    retrieval_hit_rate = (retrieval_hits / total_cases) * 100
    rec_cases_count = sum(1 for c in golden_cases if c["expected_winner"] != "N/A")
    rec_accuracy = (recommendation_correct / rec_cases_count * 100) if rec_cases_count > 0 else 100.0
    fact_coverage = (total_facts_found / total_facts_expected * 100) if total_facts_expected > 0 else 0.0

    eval_report = {
        "metrics": {
            "total_test_cases": total_cases,
            "intent_classification_accuracy": round(intent_accuracy, 2),
            "card_filter_accuracy": round(filter_accuracy, 2),
            "retrieval_topic_hit_rate": round(retrieval_hit_rate, 2),
            "recommendation_accuracy": round(rec_accuracy, 2),
            "fact_coverage_score": round(fact_coverage, 2)
        },
        "details": results_detail
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)

    print("\n" + "=" * 80)
    print("CARDWISE AI EVALUATION REPORT SUMMARY")
    print("=" * 80)
    print(f"• Total Golden Test Cases Evaluated : {total_cases}")
    print(f"• Intent Classification Accuracy    : {intent_accuracy:.1f}%")
    print(f"• Card Filter Accuracy              : {filter_accuracy:.1f}%")
    print(f"• Topic Retrieval Hit Rate          : {retrieval_hit_rate:.1f}%")
    print(f"• Recommendation Accuracy           : {rec_accuracy:.1f}%")
    print(f"• Fact Groundedness Coverage        : {fact_coverage:.1f}%")
    print(f"Report saved to: {report_file}")
    print("=" * 80)

    return eval_report


if __name__ == "__main__":
    evaluate_cardwise_agent()
