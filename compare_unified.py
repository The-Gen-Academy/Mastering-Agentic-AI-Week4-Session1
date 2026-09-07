"""Compare baseline-unified and improved-unified experiments."""
import os

from langsmith import Client
from collections import defaultdict

client = Client(api_key=os.environ["LANGSMITH_API_KEY"])

def get_experiment_metrics(project_name):
    """Get metrics for an experiment project."""
    runs = list(client.list_runs(project_name=project_name, limit=100))
    target_runs = [r for r in runs if r.name == "Target"]
    
    metrics = defaultdict(list)
    
    for r in target_runs:
        if hasattr(r, 'feedback_stats') and r.feedback_stats:
            for key, stats in r.feedback_stats.items():
                if isinstance(stats, dict) and 'avg' in stats:
                    metrics[key].append(stats['avg'])
    
    # Calculate averages
    result = {}
    for key, scores in metrics.items():
        result[key] = {
            'count': len(scores),
            'avg': sum(scores) / len(scores) if scores else 0,
            'min': min(scores) if scores else 0,
            'max': max(scores) if scores else 0
        }
    
    return result, len(target_runs)

print("=" * 80)
print("Unified Experiment Comparison")
print("=" * 80)

# Get metrics for both experiments
baseline_name = "baseline-unified-c8afbfd0"
improved_name = "improved-unified-82e62000"

print(f"\nFetching baseline: {baseline_name}...")
baseline_metrics, baseline_total = get_experiment_metrics(baseline_name)

print(f"Fetching improved: {improved_name}...")
improved_metrics, improved_total = get_experiment_metrics(improved_name)

print("\n" + "=" * 80)
print("METRICS COMPARISON")
print("=" * 80)
print(f"{'Metric':<25} {'Baseline':>15} {'Improved':>15} {'Delta':>15}")
print("-" * 80)

all_keys = sorted(set(baseline_metrics.keys()) | set(improved_metrics.keys()))

for key in all_keys:
    b = baseline_metrics.get(key, {'avg': 0, 'count': 0})
    i = improved_metrics.get(key, {'avg': 0, 'count': 0})
    
    b_avg = b['avg']
    i_avg = i['avg']
    delta = i_avg - b_avg
    
    # Format as percentage for readability
    b_str = f"{b_avg:.1%}"
    i_str = f"{i_avg:.1%}"
    delta_str = f"{delta:+.1%}"
    
    print(f"{key:<25} {b_str:>15} {i_str:>15} {delta_str:>15}")

print("-" * 80)
print(f"{'Total Target Runs':<25} {baseline_total:>15} {improved_total:>15}")
print("=" * 80)

# Show which cases improved/regressed
print("\n" + "=" * 80)
print("CASE-LEVEL COMPARISON")
print("=" * 80)

# Get case-level scores
baseline_runs = list(client.list_runs(project_name=baseline_name, limit=100))
improved_runs = list(client.list_runs(project_name=improved_name, limit=100))

baseline_cases = {}
for r in baseline_runs:
    if r.name == "Target" and hasattr(r, 'feedback_stats') and r.feedback_stats:
        case_id = r.inputs.get('query', '')[:30]  # Use query as identifier
        baseline_cases[case_id] = r.feedback_stats

improved_cases = {}
for r in improved_runs:
    if r.name == "Target" and hasattr(r, 'feedback_stats') and r.feedback_stats:
        case_id = r.inputs.get('query', '')[:30]
        improved_cases[case_id] = r.feedback_stats

# Find improvements and regressions
improvements = []
regressions = []

for case_id in baseline_cases:
    if case_id in improved_cases:
        b_score = baseline_cases[case_id].get('intent_accuracy', {}).get('avg', 0)
        i_score = improved_cases[case_id].get('intent_accuracy', {}).get('avg', 0)
        
        if i_score > b_score:
            improvements.append((case_id, b_score, i_score))
        elif i_score < b_score:
            regressions.append((case_id, b_score, i_score))

print(f"\nImprovements: {len(improvements)}")
for case_id, b, i in improvements[:10]:
    print(f"  ✓ {case_id}...: {b:.1%} → {i:.1%}")

print(f"\nRegressions: {len(regressions)}")
for case_id, b, i in regressions[:10]:
    print(f"  ✗ {case_id}...: {b:.1%} → {i:.1%}")

print("\n" + "=" * 80)
print("View in LangSmith UI:")
print(f"Baseline: https://smith.langchain.com/o/a9c4699a-b822-4097-b8a3-86a04ebb31d3/projects/p/{baseline_name}")
print(f"Improved: https://smith.langchain.com/o/a9c4699a-b822-4097-b8a3-86a04ebb31d3/projects/p/{improved_name}")
print("=" * 80)
