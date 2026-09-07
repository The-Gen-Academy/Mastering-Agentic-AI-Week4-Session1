"""
CardWise AI - Main Execution Runner
Demonstrates all 3 core agentic workflows using LangGraph and Hybrid Retrieval (BM25 + FAISS).
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from src.graph import CardWiseGraph


def run_cardwise_demo():
    print("=" * 80)
    print("CardWise AI — AI Credit Card Benefits Advisor")
    print("=" * 80)

    # Initialize Graph Architecture
    print("\nInitializing CardWise LangGraph Workflow with Hybrid Retriever (BM25 + FAISS)...")
    app = CardWiseGraph()

    # ------------------------------------------------------------------
    # TEST CASE 1: Direct Policy Q&A
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TEST CASE 1: Direct Policy Q&A")
    print("=" * 80)
    query_1 = "What is the cell phone protection policy for the Amex Platinum card, including deductible and claim limits?"
    print(f"User Query: '{query_1}'\n")

    res_1 = app.run(query_1)
    print(f"Intent Classified: {res_1['intent']}")
    print(f"Card Filter Applied: {res_1['card_filter']}")
    print(f"Retrieved {len(res_1['retrieved_docs'])} relevant policy documents.\n")
    print("--- ADVISOR RESPONSE ---")
    print(res_1["final_answer"])

    # ------------------------------------------------------------------
    # TEST CASE 2: Head-to-Head Comparison
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TEST CASE 2: Head-to-Head Comparison")
    print("=" * 80)
    query_2 = "Compare airport lounge access and guest policies between Chase Sapphire Reserve and Amex Platinum."
    print(f"User Query: '{query_2}'\n")

    res_2 = app.run(query_2)
    print(f"Intent Classified: {res_2['intent']}")
    print(f"Retrieved {len(res_2['retrieved_docs'])} relevant policy documents.\n")
    print("--- ADVISOR RESPONSE ---")
    print(res_2["final_answer"])

    # ------------------------------------------------------------------
    # TEST CASE 3: Scenario-Based Card Recommendation
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TEST CASE 3: Scenario-Based Recommendation with Spend Profile")
    print("=" * 80)
    query_3 = "I spend $12,000/yr on flights, $6,000 on hotels, and $8,000 on dining. Which card offers the highest Net Annual Value for my situation?"
    spend_3 = {
        "flights_direct": 12000.0,
        "hotels_portal": 6000.0,
        "dining": 8000.0,
        "other_spend": 2000.0
    }
    print(f"User Query: '{query_3}'")
    print(f"Spend Profile Input: {spend_3}\n")

    res_3 = app.run(query_3, spend_profile=spend_3)
    print(f"Intent Classified: {res_3['intent']}")
    print(f"Financial Recommendation: {res_3['financial_data']['recommendation'] if res_3['financial_data'] else 'N/A'}\n")
    print("--- ADVISOR RESPONSE ---")
    print(res_3["final_answer"])

    print("\n" + "=" * 80)
    print("CardWise AI Execution Completed Successfully!")
    print("=" * 80)


if __name__ == "__main__":
    run_cardwise_demo()
