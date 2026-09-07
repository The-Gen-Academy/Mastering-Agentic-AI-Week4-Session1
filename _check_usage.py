"""Check LangSmith usage and trace count."""
import os

from langsmith import Client

client = Client(api_key=os.environ["LANGSMITH_API_KEY"])

print("=" * 80)
print("LangSmith Usage Analysis")
print("=" * 80)

# List all projects
projects = list(client.list_projects(limit=50))
print(f"\nTotal projects: {len(projects)}")

total_runs = 0
for p in projects:
    try:
        runs = list(client.list_runs(project_name=p.name, limit=100))
        run_count = len(runs)
        total_runs += run_count
        print(f"  {p.name}: {run_count} runs")
    except Exception as e:
        print(f"  {p.name}: Error - {e}")

print(f"\n{'='*80}")
print(f"TOTAL RUNS ACROSS ALL PROJECTS: {total_runs}")
print(f"{'='*80}")

# Check if we can get usage info
try:
    info = client.info()
    print(f"\nTenant info: {info}")
except Exception as e:
    print(f"\nCould not get tenant info: {e}")

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)
print("If you've exceeded the 5,000 trace limit, you can:")
print("1. Delete old experiments/projects to free up traces")
print("2. Wait for the monthly reset")
print("3. Upgrade to a paid plan")
print("=" * 80)
