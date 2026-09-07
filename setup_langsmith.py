"""
LangSmith Setup for CardWise AI Evaluation
Creates project, configures tracing, uploads golden dataset, registers evaluators.

SECURITY NOTE: Never commit API keys. Set LANGSMITH_API_KEY as an environment variable.
"""

import os
import sys
from pathlib import Path

# ── Environment Configuration ──────────────────────────────────────────────
# Set these BEFORE importing langsmith/langchain to ensure tracing is active
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "CardWise_Eval"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"

# API key should be set via environment variable, not hardcoded
# If not set, prompt the user
if not os.getenv("LANGSMITH_API_KEY"):
    # Try .env file
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(env_path)
    
    if not os.getenv("LANGSMITH_API_KEY"):
        print("WARNING: LANGSMITH_API_KEY not set. Tracing will fail.")
        print("Set it via: $env:LANGSMITH_API_KEY='your-key-here'")
        print("Or create a .env file with LANGSMITH_API_KEY=your-key-here")

from langsmith import Client
from langsmith.utils import LangSmithError

# Initialize client
client = Client()

PROJECT_NAME = "CardWise_Eval"
DATASET_NAME = "CardWise_Golden_Dataset"

def create_project():
    """Create or verify the CardWise_Eval project exists."""
    print(f"Setting up LangSmith project: {PROJECT_NAME}")
    
    # LangSmith projects are created implicitly when first trace is sent
    # But we can verify connectivity and list projects
    try:
        # Test connectivity by listing datasets (lightweight call)
        datasets = list(client.list_datasets(limit=1))
        print(f"  ✓ Connected to LangSmith API")
        print(f"  ✓ Project will be created on first trace: {PROJECT_NAME}")
        return True
    except LangSmithError as e:
        print(f"  ✗ LangSmith connection failed: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")
        return False


def create_dataset():
    """Upload the golden dataset to LangSmith as a Dataset."""
    print(f"\nCreating dataset: {DATASET_NAME}")
    
    # Check if dataset already exists
    existing = list(client.list_datasets(dataset_name=DATASET_NAME))
    if existing:
        print(f"  Dataset '{DATASET_NAME}' already exists. Deleting and recreating...")
        client.delete_dataset(dataset_id=existing[0].id)
    
    # Create dataset
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="CardWise AI Golden Evaluation Dataset — 100 test cases for credit card benefits advisory"
    )
    print(f"  ✓ Created dataset: {dataset.id}")
    
    # Load golden dataset from CSV
    import csv
    import json
    
    csv_path = Path(__file__).parent / "data" / "golden_dataset.csv"
    
    examples = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse JSON fields
            spend_profile = json.loads(row["expected_spend_profile"]) if row["expected_spend_profile"] else {}
            retrieved_topics = json.loads(row["expected_retrieved_topics"]) if row["expected_retrieved_topics"] else []
            reference_facts = json.loads(row["reference_facts"]) if row["reference_facts"] else []
            
            # Build input
            inputs = {
                "query": row["query"],
                "spend_profile": spend_profile
            }
            
            # Build expected output
            expected_outputs = {
                "expected_intent": row["expected_intent"],
                "expected_card_filter": row["expected_card_filter"] if row["expected_card_filter"] not in ("", "null") else None,
                "expected_retrieved_topics": retrieved_topics,
                "expected_winner": row["expected_winner"],
                "reference_facts": reference_facts
            }
            
            # Metadata
            metadata = {
                "case_id": row["id"],
                "scenario_type": row["test_category"],
                "difficulty": row["test_subtype"],
                "prompt_version": "v2.0"  # Updated with adversarial detection
            }
            
            examples.append({
                "inputs": inputs,
                "outputs": expected_outputs,
                "metadata": metadata
            })
    
    # Upload in batches (LangSmith has rate limits)
    batch_size = 10
    for i in range(0, len(examples), batch_size):
        batch = examples[i:i+batch_size]
        client.create_examples(
            inputs=[e["inputs"] for e in batch],
            outputs=[e["outputs"] for e in batch],
            metadata=[e["metadata"] for e in batch],
            dataset_id=dataset.id
        )
        print(f"  Uploaded batch {i//batch_size + 1}/{(len(examples)-1)//batch_size + 1}")
    
    print(f"  ✓ Uploaded {len(examples)} examples to dataset '{DATASET_NAME}'")
    return dataset


def create_evaluators():
    """Register code-based evaluators for the dataset."""
    print(f"\nRegistering evaluators for dataset: {DATASET_NAME}")
    
    # Note: LangSmith evaluators are typically defined in the UI or via the SDK
    # For code-based evaluators, we'll create evaluation functions that can be run
    # via the LangSmith evaluate() function or as part of the test runner
    
    print("  Evaluators defined in: evaluators.py")
    print("  - Answer Relevance")
    print("  - Context Precision/Recall")
    print("  - Refusal Rate")
    print("  - Token Usage")
    print("\n  These will be run via the evaluation pipeline, not registered as UI evaluators.")
    print("  LangSmith UI evaluators require manual configuration in the dashboard.")


def main():
    print("=" * 70)
    print("CardWise AI — LangSmith Setup")
    print("=" * 70)
    
    # Step 1: Verify connectivity and create project
    if not create_project():
        print("\nSetup failed. Please check your LANGSMITH_API_KEY and try again.")
        sys.exit(1)
    
    # Step 2: Create dataset
    dataset = create_dataset()
    
    # Step 3: Note about evaluators
    create_evaluators()
    
    print("\n" + "=" * 70)
    print("SETUP COMPLETE")
    print("=" * 70)
    print(f"Project: {PROJECT_NAME}")
    print(f"Dataset: {DATASET_NAME} ({dataset.id})")
    print(f"\nNext steps:")
    print(f"1. Run a test: python run_traced_eval.py")
    print(f"2. View traces at: https://smith.langchain.com/")
    print(f"3. Configure evaluators in LangSmith UI for dataset '{DATASET_NAME}'")
    print("=" * 70)


if __name__ == "__main__":
    main()
