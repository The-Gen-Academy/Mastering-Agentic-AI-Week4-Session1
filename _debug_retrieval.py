"""Debug the retrieval for the Net Annual Value query."""
from src.graph import CardWiseGraph

app = CardWiseGraph()

query = "I spend $12,000 on flights, $6,000 on hotels, and $8,000 on dining annually. Which card gives me the highest Net Annual Value, and why?"

# Check the retrieval node directly
state = {
    "query": query, "intent": "recommendation", "card_filter": "",
    "spend_profile": {"flights": 12000, "hotels": 6000, "dining": 8000},
    "retrieved_docs": [], "retrieved_context": "",
    "financial_summary": "", "financial_data": None, "final_answer": ""
}

result = app.retrieval_node(state)

print(f"Query: {query}")
print(f"Intent: {result['intent']}")
print(f"Card Filter: {result['card_filter']}")
print(f"Retrieved {len(result['retrieved_docs'])} documents:")
print()

for i, doc in enumerate(result['retrieved_docs'][:10]):
    meta = doc.metadata
    print(f"  {i+1}. {meta.get('card', 'N/A')} — {meta.get('topic', 'N/A')}")
    print(f"     Card Code: {meta.get('card_code', 'N/A')}")
    print()
