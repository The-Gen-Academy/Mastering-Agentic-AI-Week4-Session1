"""Check which runs have feedback and their types."""
import os

from langsmith import Client

client = Client(api_key=os.environ["LANGSMITH_API_KEY"])

exp_project = "cardwise-full-eval-v3-b1e1ecc7"
runs = list(client.list_runs(project_name=exp_project, limit=100))

print("=" * 80)
print("Run Type Analysis")
print("=" * 80)

# Group by name
from collections import defaultdict
by_name = defaultdict(list)
for r in runs:
    by_name[r.name].append(r)

for name, name_runs in sorted(by_name.items()):
    with_feedback = sum(1 for r in name_runs if hasattr(r, 'feedback_stats') and r.feedback_stats)
    print(f"{name:<20}: {len(name_runs)} runs, {with_feedback} with feedback")

print("\n" + "=" * 80)
print("Target Runs (should have feedback):")
print("=" * 80)

target_runs = [r for r in runs if r.name == "Target"]
print(f"Total Target runs: {len(target_runs)}")

target_with_feedback = [r for r in target_runs if hasattr(r, 'feedback_stats') and r.feedback_stats]
print(f"Target runs with feedback: {len(target_with_feedback)}")

if target_with_feedback:
    print("\nSample Target run with feedback:")
    r = target_with_feedback[0]
    print(f"  Name: {r.name}")
    print(f"  ID: {r.id}")
    print(f"  Feedback keys: {list(r.feedback_stats.keys())}")
    
    # Show scores
    for key, stats in r.feedback_stats.items():
        if isinstance(stats, dict) and 'avg' in stats:
            print(f"    {key}: {stats['avg']:.2f}")

print("\n" + "=" * 80)
print("Summary:")
print(f"  Total runs: {len(runs)}")
print(f"  Target runs: {len(target_runs)}")
print(f"  Target runs with feedback: {len(target_with_feedback)}")
print(f"  Coverage: {len(target_with_feedback)/len(target_runs)*100:.0f}%")
print("=" * 80)
