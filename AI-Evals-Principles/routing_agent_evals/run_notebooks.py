"""Execute the two lessons in fresh kernels. Defaults to no-charge replay."""
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    lab = Path(__file__).resolve().parent
    output = lab / ".runs/executed"
    output.mkdir(parents=True, exist_ok=True)
    for name in ("routing_agent_evals.ipynb", "llm_as_judge.ipynb"):
        notebook = nbformat.read(lab / name, as_version=4)
        nbformat.validate(notebook)
        NotebookClient(notebook, timeout=1800, kernel_name="python3",
                       resources={"metadata": {"path": str(lab)}}).execute()
        nbformat.write(notebook, output / name)
        print(f"Completed {name}: {sum(c.cell_type == 'code' for c in notebook.cells)} code cells", flush=True)


if __name__ == "__main__":
    main()
