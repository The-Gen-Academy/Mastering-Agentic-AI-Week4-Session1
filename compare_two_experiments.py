"""Compare two specific experiments in CardWise_Eval project."""
import os
import sys

from langsmith import Client

client = Client(api_key=os.environ["LANGSMITH_API_KEY"])
PROJECT_NAME = "CardWise_Eval"

def get_experiment_runs(experiment_prefix: str):
    """Get all runs for a specific experiment prefix."""
    all_runs = []
    offset = 0
    limit = 100
    
    while True:
        runs = list(client.list_runs(project_name=PROJECT_NAME, limit=limit, offset=offset))
        if not runs:
            break
        
        # Filter by experiment prefix
        exp_runs = [r for r in runs if r.name.startswith(f"{experiment_prefix}_gold_")]
        all_runs.extend(exp_runs)
        
        offset += limit
        if len(runs) < limit:
            break
        
        # Safety limit
        if offset > 1000:
            break
    
    return all_runs


def calculate_metrics(runs):
    """Calculate metrics for a set of runs."""
    valid_runs = [r for r in runs if r.extra and "metadata" in r.extra]
    
    if not valid_runs:
        return None
    
    intent_correct = sum(1 for r in valid_runs 
                        if r.extra["metadata"].get("correctness", {}).get("intent_match"))
    filter_correct = sum(1 for r in valid_runs 
                        if r.extra["metadata"].get("correctness", {}).get("filter_match"))
    overall_correct = sum(1 for r in valid_runs 
                         if r.extra["metadata"].get("correctness", {}).get("overall"))
    total_latency = sum(r.extra["metadata"].get("timing", {}).get("latency_ms", 0) 
                       for r in valid_runs)
    total = len(valid_runs)
    
    # Get failing case IDs
    failing = [r.extra["metadata"].get("case_id") for r in valid_runs 
               if not r.extra["metadata"].get("correctness", {}).get("overall")]
    
    return {
        "total": total,
        "intent_accuracy": intent_correct / total * 100 if total else 0,
        "filter_accuracy": filter_correct / total * 100 if total else 0,
        "overall_accuracy": overall_correct / total * 100 if total else 0,
        "avg_latency": total_latency / total if total else 0,
        "failing_cases": sorted(failing)
    }


def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_two_experiments.py <baseline> <current>")
        print("Example: python compare_two_experiments.py baseline-v1 improved-retrieval-v2")
        sys.exit(1)
    
    baseline_name = sys.argv[1]
    current_name = sys.argv[2]
    
    print("=" * 80)
    print(f"Experiment Comparison: {baseline_name} vs {current_name}")
    print("=" * 80)
    
    # Fetch runs for both experiments
    print(f"\nFetching {baseline_name} runs...")
    baseline_runs = get_experiment_runs(baseline_name)
    print(f"  Found {len(baseline_runs)} runs")
    
    print(f"Fetching {current_name} runs...")
    current_runs = get_experiment_runs(current_name)
    print(f"  Found {len(current_runs)} runs")
    
    # Calculate metrics
    baseline_metrics = calculate_metrics(baseline_runs)
    current_metrics = calculate_metrics(current_runs)
    
    if not baseline_metrics or not current_metrics:
        print("\nError: Could not calculate metrics for one or both experiments")
        print(f"  {baseline_name}: {len(baseline_runs)} runs")
        print(f"  {current_name}: {len(current_runs)} runs")
        sys.exit(1)
    
    # Display comparison
    print("\n" + "=" * 80)
    print("METRICS COMPARISON")
    print("=" * 80)
    print(f"{'Metric':<25} {baseline_name:>20} {current_name:>20} {'Delta':>12}")
    print("-" * 80)
    
    metrics = [
        ("Total Runs", "total", ""),
        ("Intent Accuracy", "intent_accuracy", "%"),
        ("Filter Accuracy", "filter_accuracy", "%"),
        ("Overall Accuracy", "overall_accuracy", "%"),
        ("Avg Latency", "avg_latency", "ms")
    ]
    
    for label, key, unit in metrics:
        b_val = baseline_metrics[key]
        c_val = current_metrics[key]
        
        if key == "total":
            delta = c_val - b_val
            delta_str = f"{delta:+d}"
        else:
            delta = c_val - b_val
            delta_str = f"{delta:+.1f}{unit}"
        
        if unit == "%":
            b_str = f"{b_val:.1f}%"
            c_str = f"{c_val:.1f}%"
        elif unit == "ms":
            b_str = f"{b_val:.0f}ms"
            c_str = f"{c_val:.0f}ms"
        else:
            b_str = str(b_val)
            c_str = str(c_val)
        
        print(f"{label:<25} {b_str:>20} {c_str:>20} {delta_str:>12}")
    
    print("-" * 80)
    
    # Show improvement/regression
    baseline_failing = set(baseline_metrics["failing_cases"])
    current_failing = set(current_metrics["failing_cases"])
    
    fixed = baseline_failing - current_failing
    broken = current_failing - baseline_failing
    still_failing = baseline_failing & current_failing
    
    print("\n" + "=" * 80)
    print("IMPROVEMENT ANALYSIS")
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
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    overall_delta = current_metrics["overall_accuracy"] - baseline_metrics["overall_accuracy"]
    if overall_delta > 0:
        print(f"✓ Improvement: +{overall_delta:.1f} percentage points")
    elif overall_delta < 0:
        print(f"✗ Regression: {overall_delta:.1f} percentage points")
    else:
        print("≈ No change in overall accuracy")
    
    print(f"\nView in LangSmith:")
    print(f"https://smith.langchain.com/o/a9c4699a-b822-4097-b8a3-86a04ebb31d3/projects/p/{PROJECT_NAME}")
    print("=" * 80)


if __name__ == "__main__":
    main()
