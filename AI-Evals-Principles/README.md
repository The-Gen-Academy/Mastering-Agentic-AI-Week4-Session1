# 🚦 AI Evals: Evaluating a Routing Agent

A hands-on tutorial on **AI Evaluation principles**. You'll build a real routing agent, test it against a golden dataset, and then learn how to use an **LLM-as-a-Judge** to scale evaluation — and how to check whether that judge can actually be trusted.

The running example is a **doctor's scheduling assistant**: it takes natural language (*"I'm busy tomorrow"*) and maps it to an executable action (`BLOCK_SLOTS`).

---

## 📋 Why This Matters

In most agentic systems, **routing** is the critical first step — figuring out *what* the user wants before doing anything. If routing is wrong, everything downstream is wrong.

This tutorial answers two questions every AI team eventually faces:

1. **Is my agent good enough?** → Measure it against ground truth (Notebook 1)
2. **Can I evaluate at scale without a human reading every log?** → Use an LLM judge, then prove it agrees with humans (Notebook 2)

---

## 📁 Repository Layout

```
routing_agent_evals/
├── routing_agent_evals.ipynb     # Notebook 1: Build & evaluate the router
├── llm_as_judge.ipynb            # Notebook 2: LLM-as-a-Judge alignment
├── data/
│   ├── golden_dataset.csv                  # Ground-truth test cases
│   ├── golden_dataset_small_5.csv          # 5-row subset for quick smoke tests
│   ├── golden_dataset_with_predictions.csv # Router predictions (output of NB1)
│   ├── llm_as_judge_dataset.csv            # Human-labeled failures
│   └── judge_alignment_*.csv               # Judge vs. human results
├── prompts/
│   ├── router_prompts.py                   # Router system prompt
│   ├── llm_as_judge_prompt.py              # Judge evaluation prompt
│   └── failure_categorization_prompts.py   # Prompt to mine failure categories
└── images/                                 # Diagrams used in the notebooks
```

---

## 🚀 Getting Started

### 1. Install dependencies

This project uses [**uv**](https://docs.astral.sh/uv/) for dependency management (see `pyproject.toml` + `uv.lock`). From the repo root:

```bash
uv sync
```

This creates a `.venv` with everything pinned in `uv.lock`. Requires **Python ≥ 3.10**. (Prefer pip? Use `pip install -e .` into your own virtual environment instead.)

### 2. Add your API key

The notebooks load secrets from a `.env` file **inside `routing_agent_evals/`** (they call `load_dotenv(dotenv_path=".env")` from that directory). Create it there:

```bash
# routing_agent_evals/.env
OPENAI_API_KEY=your_api_key_here

# Optional — only needed for the LangSmith tracing cells
LANGSMITH_API_KEY=your_langsmith_key_here
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=ai-evals-principles
```

Get an OpenAI key from [OpenAI](https://platform.openai.com/); the LangSmith keys are optional and only used for the tracing/feedback cells.

### 3. Run the notebooks in order

Open `routing_agent_evals/routing_agent_evals.ipynb` first, then `routing_agent_evals/llm_as_judge.ipynb`. Select the `.venv` created by `uv sync` as the notebook kernel, then run all cells top to bottom.

> ⏱️ **Heads up:** Both notebooks make one API call per row. A full run of Notebook 1 takes ~7 min; Notebook 2 takes ~20–35 min depending on the judge model. The repo already ships the saved CSV outputs, so you can also just read along.

---

## 📓 Notebook 1 — Build & Evaluate the Router

**File:** `routing_agent_evals.ipynb`

This notebook builds the router from scratch and grades it.

| Step | What happens | Key idea |
|------|--------------|----------|
| **1. Setup** | Load libraries and API keys | — |
| **2. Define intents** | Declare all 10 actions as a Python `Enum`, each with a description | The description *is* the prompt that teaches the LLM what each intent means |
| **3. Flatten intents** | Turn the Enum into a list to inject into the prompt | Single source of truth for intent definitions |
| **4. Init model** | `gpt-4o-mini`, `temperature=0` | Routing should be **fast** (small model) and **deterministic** (temp 0) |
| **5. Router agent** | Bind a **Pydantic schema** (`intent`, `confidence`, `reasoning`) to the LLM | Forces structured output + chain-of-thought, not rambling text |
| **6. Sanity check** | Run one query to confirm it works | Always test one before testing all |
| **7. Batch eval** | Run the router over the **golden dataset** | Compare each prediction to human ground truth |
| **8–11. Analyze** | Class distribution → classification report | Use **Precision / Recall / F1**, not just accuracy |

### The metrics that matter

- **Precision** — *"When the model says X, is it really X?"* Low precision = false alarms.
- **Recall** — *"Of all the real X queries, how many did we catch?"* Low recall = missed cases.
- **F1** — the harmonic mean; one number to compare intents.

> 🔍 **Watch for confusable intents.** Does the router mix up `BLOCK_SLOTS` (doctor unavailable) with `UNBLOCK_SLOTS` (doctor available)? A low score there usually means the intent *description* needs sharpening — not a bigger model.

---

## 👨‍⚖️ Notebook 2 — LLM-as-a-Judge & Alignment

**File:** `llm_as_judge.ipynb`

Reading thousands of agent logs by hand doesn't scale. So we hire an **LLM judge** to grade them. But a judge is only useful if it agrees with humans — so the real lesson here is **measuring that agreement**.

The judge here answers one focused question: *Did the router fail because the user's utterance was **vague**?*

| Step | What happens | Key idea |
|------|--------------|----------|
| **1. Init judges** | `gpt-4o` and `gpt-5-mini`, `temperature=0` | The judge is usually a *stronger* model than the one being tested |
| **2. Schema** | Pydantic `label` (`TRUE`/`FALSE`) + `reasoning` | Clean booleans we can compute metrics on |
| **3. Sanity check** | Grade one edge case (*"Clear my 3 PM"*) | Confirm the judge logic holds |
| **4. Load data** | `llm_as_judge_dataset.csv` with **human labels** | The human column is our ground truth for the judge |
| **5. Batch judge** | Run the judge over every row | One API call per log |
| **7–10. Align** | Class balance → confusion matrix → Precision/Recall | Measure judge vs. human agreement |
| **11. Verify** | Re-check on the second judge's saved results | Reproducibility check |

### Reading the alignment

The dataset is **heavily imbalanced** (~93% "no failure", ~7% "failure"). That's the whole point:

> A lazy judge that always answers "FALSE" scores **93% accuracy** while catching **zero** real failures. Accuracy lies on imbalanced data — that's why we look at the **confusion matrix** and the **Precision/Recall of the rare `TRUE` class**.

In this tutorial, `gpt-5-mini` reaches ~88% accuracy but only ~0.28 F1 on the `TRUE` class — a concrete reminder that a judge can look good and still miss most of the failures you actually care about.

---

## 🧠 The Big Lessons

1. **Definitions are prompts.** Most routing errors are fixed by sharper intent descriptions, not bigger models.
2. **Accuracy is misleading.** On imbalanced data, lean on Precision, Recall, F1, and the confusion matrix.
3. **Trust, then verify your judge.** An LLM judge only earns the right to scale your evals *after* you've measured its alignment with humans.

---

## 🛠️ Tech Stack

| Tool | Role |
|------|------|
| **LangChain 1.0** | Chains prompts + models together |
| **OpenAI GPT models** | `gpt-4o-mini` (router), `gpt-4o` + `gpt-5-mini` (judges) |
| **Pydantic v2** | Forces valid, structured output |
| **Pandas** | Batch processing and analysis |
| **scikit-learn** | Precision, Recall, F1, confusion matrix |
| **LangSmith** | Optional tracing / feedback / dataset cells |
| **uv** | Dependency management + reproducible env |

---

## 📚 Resources

- [Evals Workbook (The Gen Academy)](https://github.com/The-Gen-Academy/Mastering-Agentic-AI_Week4)
- [LangChain Structured Output](https://python.langchain.com/docs/how_to/structured_output/)
- [Semantic Routing Concepts](https://python.langchain.com/docs/concepts/#routing)
- [Scikit-Learn Classification Report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [LangSmith Evaluation](https://docs.smith.langchain.com/evaluation)
