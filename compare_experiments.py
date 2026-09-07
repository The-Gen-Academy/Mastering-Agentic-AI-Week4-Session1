"""Compare experiments with pagination to get all runs."""
import os

from langsmith import Client
from collections import defaultdict

client = Client(api_key=os.environ["LANGSMITH_API_KEY"])
PROJECT_NAME = "CardWise_Eval"

print("=" * 80)
print(f"Experiment Comparison — Project: {PROJECT_NAME}")
print("=" * 80)

# Get all runs with pagination
all_runs = []
offset = 0
limit = 100
max_iterations = 20  # Safety limit

print("Fetching runs...", end=" ")
for i in range(max_iterations):
    runs = list(client.list_runs(project_name=PROJECT_NAME, limit=limit, offset=offset))
    if not runs:
        break
    all_runs.extend(runs)
    offset += limit
    print(f"{len(all_runs)}", end=" ")
    if len(runs) < limit:
        break
print()

print(f"\nTotal runs fetched: {len(all_runs)}")

# Group runs by experiment name
experiments = defaultdict(list)
for r in all_runs:
    name = r.name
    if "_gold_" in name:
        exp_name = name.split("_gold_")[0]
    else:
        exp_name = "other"
    experiments[exp_name].append(r)

print(f"\nExperiments found:")
for exp_name, exp_runs in sorted(experiments.items()):
    print(f"  - {exp_name}: {len(exp_runs)} runs")

# Calculate metrics for each experiment
print("\n" + "=" * 80)
print("Metrics Comparison")
print("=" * 80)
print(f"{'Experiment':<25} {'Runs':>6} {'Intent%':>9} {'Filter%':>9} {'Overall%':>9} {'Avg Latency':>12}")
print("-" * 80)

for exp_name, exp_runs in sorted(experiments.items()):
    if exp_name == "other":
        continue
    
    # Filter to runs with metadata
    valid_runs = [r for r in exp_runs if r.extra and "metadata" in r.extra]
    
    if not valid_runs:
        continue
    
    # Calculate metrics
    intent_correct = 0
    filter_correct = 0
    overall_correct = 0
    total_latency = 0
    total = 0
    
    for r in valid_runs:
        meta = r.extra["metadata"]
        correctness = meta.get("correctness", {})
        timing = meta.get("timing", {})
        
        if correctness.get("intent_match"):
            intent_correct += 1
        if correctness.get("filter_match"):
            filter_correct += 1
        if correctness.get("overall"):
            overall_correct += 1
        
        total_latency += timing.get("latency_ms", 0)
        total += 1
    
    if total > 0:
        intent_acc = intent_correct / total * 100
        filter_acc = filter_correct / total * 100
        overall_acc = overall_correct / total * 100
        avg_latency = total_latency / total
        
        print(f"{exp_name:<25} {total:>6} {intent_acc:>8.1f}% {filter_acc:>8.1f}% {overall_acc:>8.1f}% {avg_latency:>10.0f}ms")

print("-" * 80)

# Show failing cases for each experiment
print("\n" + "=" * 80)
print("Failing Cases by Experiment")
print("=" * 80)

for exp_name, exp_runs in sorted(experiments.items()):
    if exp_name == "unknown":
        continue
    
    valid_runs = [r for r in exp_runs if r.extra and "metadata" in r.extra]
    failing = [r for r in valid_runs if not r.extra["metadata"].get("correctness", {}).get("overall")]
    
    if failing:
        print(f"\n{exp_name} — {len(failing)} failing cases:")
        for r in failing[:10]:  # Show first 10
            meta = r.extra["metadata"]
            case_id = meta.get("case_id", "unknown")
            true_cat = meta.get("true_category", "N/A")
            pred_cat = meta.get("predicted_category", "N/A")
            print(f"  {case_id}: expected={true_cat}, got={pred_cat}")

print("\n" + "=" * 80)
print("View detailed comparison in LangSmith UI:")
print(f"https://smith.langchain.com/o/a9c4699a-b822-4097-b8a3-86a04ebb31d3/projects/p/{PROJECT_NAME}")
print("=" * 80)

# Improvement/Regression Analysis
if len(experiments) >= 2:
    exp_names = [n for n in sorted(experiments.keys()) if n != "other"]
    if len(exp_names) >= 2:
        baseline = exp_names[0]
        current = exp_names[1]
        
        # Get failing case IDs for each experiment
        exp_failures = {}
        for exp_name in exp_names:
            exp_runs = experiments[exp_name]
            valid_runs = [r for r in exp_runs if r.extra and "metadata" in r.extra]
            failing = [r for r in valid_runs if not r.extra["metadata"].get("correctness", {}).get("overall")]
            exp_failures[exp_name] = {r.extra["metadata"].get("case_id", "unknown") for r in failing}
        
        baseline_failures = exp_failures[baseline]
        current_failures = exp_failures[current]
        
        fixed = baseline_failures - current_failures
        broken = current_failures - baseline_failures
        still_failing = baseline_failures & current_failures
        
        print("\n" + "=" * 80)
        print(f"Improvement Analysis: {baseline} → {current}")
        print("=" * 80)
        print(f"✓ Fixed: {len(fixed)} cases")
        if fixed:
            print(f"  {sorted(fixed)}")
        print(f"✗ Broken: {len(broken)} cases")
        if broken:
            print(f"  {sorted(broken)}")
        print(f"≈ Still failing: {len(still_failing)} cases")
        if still_failing:
            print(f"  {sorted(still_failing)}")
        print("=" * 80)
