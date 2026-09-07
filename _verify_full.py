"""Verify all 100 runs have feedback in the new experiment."""
import os

from langsmith import Client

client = Client(api_key=os.environ["LANGSMITH_API_KEY"])

# Find the new experiment
projects = list(client.list_projects(limit=50))
exp_projects = [p for p in projects if "cardwise-full-eval-v3" in p.name]

if not exp_projects:
    print("Experiment not found!")
    exit(1)

exp_project = exp_projects[0]
print("=" * 80)
print(f"Experiment: {exp_project.name}")
print(f"ID: {exp_project.id}")
print("=" * 80)

# Get all runs
runs = list(client.list_runs(project_name=exp_project.name, limit=100))
print(f"\nTotal runs: {len(runs)}")

# Check feedback
runs_with_feedback = [r for r in runs if hasattr(r, 'feedback_stats') and r.feedback_stats]
print(f"Runs with feedback: {len(runs_with_feedback)}")

# Calculate metrics
from collections import defaultdict
metrics = defaultdict(list)

for r in runs_with_feedback:
    for key, stats in r.feedback_stats.items():
        if isinstance(stats, dict) and 'avg' in stats:
            metrics[key].append(stats['avg'])

print("\n" + "=" * 80)
print("METRICS SUMMARY")
print("=" * 80)
print(f"{'Metric':<25} {'Count':>8} {'Avg Score':>12}")
print("-" * 80)

for key in sorted(metrics.keys()):
    scores = metrics[key]
    avg = sum(scores) / len(scores) if scores else 0
    print(f"{key:<25} {len(scores):>8} {avg:>11.2f}")

print("-" * 80)

# Show which runs are missing feedback
runs_without_feedback = [r for r in runs if not (hasattr(r, 'feedback_stats') and r.feedback_stats)]
if runs_without_feedback:
    print(f"\nRuns WITHOUT feedback: {len(runs_without_feedback)}")
    for r in runs_without_feedback[:10]:
        print(f"  - {r.name}")
else:
    print("\n✓ All runs have feedback!")

print("\n" + "=" * 80)
print("View in LangSmith UI:")
print(f"https://smith.langchain.com/o/a9c4699a-b822-4097-b8a3-86a04ebb31d3/projects/p/{exp_project.name}")
print("=" * 80)
