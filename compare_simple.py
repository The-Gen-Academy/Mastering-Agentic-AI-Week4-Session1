"""Simple experiment comparison using recent runs."""
import os

from langsmith import Client
from collections import defaultdict

client = Client(api_key=os.environ["LANGSMITH_API_KEY"])
PROJECT_NAME = "CardWise_Eval"

print("=" * 80)
print(f"Experiment Comparison — Project: {PROJECT_NAME}")
print("=" * 80)

# Get recent runs (max 100 allowed by API)
print("\nFetching recent runs...")
runs = list(client.list_runs(project_name=PROJECT_NAME, limit=100))
print(f"Fetched {len(runs)} runs")

# Group by experiment
experiments = defaultdict(list)
for r in runs:
    name = r.name
    if "_gold_" in name:
        exp_name = name.split("_gold_")[0]
    else:
        exp_name = "other"
    experiments[exp_name].append(r)

print(f"\nExperiments found:")
for exp_name, exp_runs in sorted(experiments.items()):
    print(f"  - {exp_name}: {len(exp_runs)} runs")

# Calculate metrics
print("\n" + "=" * 80)
print("Metrics Comparison")
print("=" * 80)
print(f"{'Experiment':<25} {'Runs':>6} {'Intent%':>9} {'Filter%':>9} {'Overall%':>9}")
print("-" * 80)

for exp_name, exp_runs in sorted(experiments.items()):
    if exp_name == "other":
        continue
    
    valid_runs = [r for r in exp_runs if r.extra and "metadata" in r.extra]
    
    if not valid_runs:
        continue
    
    intent_correct = sum(1 for r in valid_runs 
                        if r.extra["metadata"].get("correctness", {}).get("intent_match"))
    filter_correct = sum(1 for r in valid_runs 
                        if r.extra["metadata"].get("correctness", {}).get("filter_match"))
    overall_correct = sum(1 for r in valid_runs 
                         if r.extra["metadata"].get("correctness", {}).get("overall"))
    total = len(valid_runs)
    
    if total > 0:
        print(f"{exp_name:<25} {total:>6} {intent_correct/total*100:>8.1f}% "
              f"{filter_correct/total*100:>8.1f}% {overall_correct/total*100:>8.1f}%")

print("-" * 80)

# Show which cases failed in each experiment
print("\n" + "=" * 80)
print("Failing Cases by Experiment")
print("=" * 80)

for exp_name, exp_runs in sorted(experiments.items()):
    if exp_name == "other":
        continue
    
    valid_runs = [r for r in exp_runs if r.extra and "metadata" in r.extra]
    failing = [r for r in valid_runs 
               if not r.extra["metadata"].get("correctness", {}).get("overall")]
    
    if failing:
        print(f"\n{exp_name}: {len(failing)} failing")
        for r in failing[:5]:
            meta = r.extra["metadata"]
            print(f"  {meta.get('case_id')}: {meta.get('true_category')} -> {meta.get('predicted_category')}")

print("\n" + "=" * 80)
print("View in LangSmith:")
print(f"https://smith.langchain.com/o/a9c4699a-b822-4097-b8a3-86a04ebb31d3/projects/p/{PROJECT_NAME}")
print("=" * 80)
