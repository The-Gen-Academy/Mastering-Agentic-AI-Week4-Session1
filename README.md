# CardWise AI

CardWise AI is a RAG-powered credit card benefits advisor for the Chase Sapphire Reserve and American Express Platinum cards. It combines LangGraph workflow orchestration, hybrid BM25 + FAISS retrieval, deterministic financial calculations, and optional LLM-generated answers.

## Features

- Answers questions about card benefits, policies, credits, and protections.
- Compares benefits between the Chase Sapphire Reserve and Amex Platinum.
- Calculates estimated points, statement credits, annual fees, and net annual value from a spend profile.
- Detects unsupported, adversarial, or out-of-scope questions.
- Provides a FastAPI web interface and JSON API.
- Includes golden-dataset evaluation and optional LangSmith tracing/evaluation workflows.

## Architecture

The main workflow is implemented by `src.graph.CardWiseGraph`:

1. Classify the request and identify a card filter.
2. Retrieve relevant benefit documents using BM25 and FAISS.
3. Run the deterministic financial calculator when a spend profile is provided.
4. Generate the final advisor response using an available LLM, or use the built-in fallback response path.

The source data is stored in `data/card_benefits_data.py`. The calculator is implemented in `src/financial_calculator.py`, and retrieval is implemented in `src/hybrid_retriever.py`.

## Requirements

- Python 3.10 or newer
- A virtual environment is recommended
- API access to an LLM provider is optional for the rule-based and retrieval components, but required for LLM-generated responses

## Installation

From this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

## Environment Variables

Copy `.env.example` to `.env` and replace the placeholders with your own values. Never commit `.env` or API keys.

For the advisor model, configure one of these providers:

```dotenv
OPENAI_API_KEY=your-openai-api-key
# GROQ_API_KEY=your-groq-api-key
```

`src.graph` checks `OPENAI_API_KEY` first and uses `gpt-4o-mini`. If it is unavailable, it checks `GROQ_API_KEY` and uses the configured Groq model. Without either key, the application can still run its retrieval, routing, and deterministic calculation paths, but LLM-generated answers are not available.

For LangSmith tracing and evaluation scripts, set the key in the process environment:

```powershell
$env:LANGSMITH_API_KEY = "your-langsmith-api-key"
$env:LANGCHAIN_TRACING_V2 = "true"
$env:LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
$env:LANGCHAIN_PROJECT = "CardWise-AI-Advisor"
```

The LangSmith helper scripts read `LANGSMITH_API_KEY` from the environment. They do not contain API keys in source code.

## Run the Terminal Demo

```powershell
python main.py
```

The demo runs three examples:

- Direct policy question about cell phone protection.
- Head-to-head lounge access comparison.
- Spend-profile recommendation based on estimated net annual value.

## Run the Web Application

```powershell
python app.py
```

Open http://127.0.0.1:8000 in a browser. The application serves the interactive dashboard and FastAPI documentation is available at http://127.0.0.1:8000/docs.

The server exposes these API endpoints:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/api/cards` | Return summary metadata for both cards |
| GET | `/api/benefits` | Return structured card-benefit documents |
| POST | `/api/query` | Run the routing, retrieval, calculation, and answer workflow |
| POST | `/api/calculate` | Run deterministic financial calculations |

Example query request:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/query `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"query":"Compare airport lounge access for both cards."}'
```

## Evaluation

The repository includes scripts for evaluating the application against the golden dataset and for sending evaluations to LangSmith. Common entry points include:

```powershell
python evaluate_golden_dataset.py
python run_batch_eval.py
python run_langsmith_eval.py
python run_unified_eval.py
```

Review each script before running it because some workflows make many model or LangSmith API calls. LangSmith scripts require `LANGSMITH_API_KEY` to be set in the environment.

## Project Layout

```text
CardWise AI/
├── app.py                    # FastAPI server and web dashboard
├── main.py                   # Terminal demonstration runner
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variable template
├── data/                     # Card benefit data and evaluation datasets
├── src/
│   ├── graph.py              # LangGraph workflow
│   ├── hybrid_retriever.py   # BM25 + FAISS retrieval
│   ├── financial_calculator.py  # Deterministic value calculations
│   └── prompts.py            # Router and advisor prompts
├── PRD/                      # Product requirements and supporting documents
└── *_eval.py                 # Evaluation and LangSmith utilities
```

## Security Notes

- Keep `.env` local and do not commit it.
- Set `LANGSMITH_API_KEY` through the environment when running LangSmith tools.
- If an API key has ever been committed or shared, revoke it and create a replacement.
- Card benefits, fees, credits, and reward rules can change. Verify current terms with the card issuer before making financial decisions.
