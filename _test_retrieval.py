"""Test retrieval for previously failing comparison/recommendation cases."""
from src.graph import CardWiseGraph

app = CardWiseGraph()

test_cases = [
    ("gold_23", "Which card is better for hotel bookings: Chase Sapphire Reserve or Amex Platinum?", "comparison"),
    ("gold_27", "How do travel protection and insurance limits compare between Sapphire Reserve and Amex Platinum?", "comparison"),
    ("gold_42", "I spend $15,000 on dining, $10,000 on lululemon, streaming, and Uber, and $5,000 on flights. Which card is recommended?", "recommendation"),
    ("gold_43", "Which card is best if I travel with my spouse and two kids 4 times a year?", "recommendation"),
    ("gold_46", "I buy expensive electronics and designer goods. Which card offers the best purchase protection and return protection?", "recommendation"),
    ("gold_47", "I spend $3,000/yr on Lyft rideshares. Which card gives me the best value for rideshare?", "recommendation"),
    ("gold_48", "I want a card that pays for my CLEAR+ and Global Entry applications. Which card covers both?", "recommendation"),
]

for cid, query, expected_intent in test_cases:
    state = {
        "query": query, "intent": expected_intent, "card_filter": "",
        "spend_profile": {}, "retrieved_docs": [], "retrieved_context": "",
        "financial_summary": "", "financial_data": None, "final_answer": ""
    }
    
    # Test retrieval node directly
    result = app.retrieval_node(state)
    topics = [doc.metadata.get("topic", "") for doc in result["retrieved_docs"]]
    
    print(f"[{cid}] {query[:60]}...")
    print(f"  Intent: {expected_intent}")
    print(f"  Retrieved {len(result['retrieved_docs'])} docs")
    print(f"  Topics: {topics[:5]}")
    print()
