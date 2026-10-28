"""Execute live judge notebooks in separate kernels using API credentials."""
import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jev-only", action="store_true", help="Run the Jev notebook only; no LLM credential needed")
    parser.add_argument("--include-router", action="store_true", help="Also execute the router generation lesson")
    args = parser.parse_args()
    lab = Path(__file__).resolve().parent
    output = lab / ".runs/executed"
    output.mkdir(parents=True, exist_ok=True)
    names = ["jev_as_judge.ipynb"]
    if args.include_router:
        names.insert(0, "routing_agent_evals.ipynb")
    if not args.jev_only:
        names.append("jev_vs_llm_judge.ipynb")
    for name in names:
        notebook = nbformat.read(lab / name, as_version=4)
        nbformat.validate(notebook)
        NotebookClient(notebook, timeout=1800, kernel_name="python3",
                       resources={"metadata": {"path": str(lab)}}).execute()
        nbformat.write(notebook, output / name)
        print(f"Completed {name}: {sum(c.cell_type == 'code' for c in notebook.cells)} code cells", flush=True)


if __name__ == "__main__":
    main()
