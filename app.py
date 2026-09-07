"""
CardWise AI Web Server (FastAPI + Uvicorn)
Serves the CardWise AI Web Interface and API endpoints for RAG lookup & financial math.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel

# Add project directory to path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from src.graph import CardWiseGraph
from data.card_benefits_data import get_card_summary, get_all_benefits

from contextlib import asynccontextmanager

cardwise_app: Optional[CardWiseGraph] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global cardwise_app
    print("Initializing CardWise AI LangGraph Engine...")
    cardwise_app = CardWiseGraph()
    print("CardWise AI Ready!")
    yield

app = FastAPI(
    title="CardWise AI — AI Credit Card Advisor",
    description="RAG-powered advisor for Chase Sapphire Reserve & Amex Platinum",
    version="1.0.0",
    lifespan=lifespan
)


class QueryRequest(BaseModel):
    query: str
    spend_profile: Optional[Dict[str, float]] = None


class CalculateRequest(BaseModel):
    flights_direct: float = 0.0
    flights_portal: float = 0.0
    hotels_direct: float = 0.0
    hotels_portal: float = 0.0
    dining: float = 0.0
    lyft: float = 0.0
    other_spend: float = 0.0


def get_cardwise_app() -> CardWiseGraph:
    global cardwise_app
    if cardwise_app is None:
        cardwise_app = CardWiseGraph()
    return cardwise_app


@app.get("/api/cards", response_class=JSONResponse)
def get_cards():
    """Returns summary metadata for both premium cards."""
    return get_card_summary()


@app.get("/api/benefits", response_class=JSONResponse)
def get_benefits():
    """Returns all structured benefit documents."""
    return get_all_benefits()


@app.post("/api/query", response_class=JSONResponse)
def process_query(req: QueryRequest):
    """Executes RAG pipeline, router, retrieval, math, and advisor synthesis."""
    graph_app = get_cardwise_app()
    res = graph_app.run(query=req.query, spend_profile=req.spend_profile or {})
    return {
        "query": res["query"],
        "intent": res["intent"],
        "card_filter": res["card_filter"],
        "retrieved_docs_count": len(res["retrieved_docs"]),
        "retrieved_context": res["retrieved_context"],
        "financial_summary": res["financial_summary"],
        "financial_data": res["financial_data"],
        "final_answer": res["final_answer"]
    }


@app.post("/api/calculate", response_class=JSONResponse)
def calculate_financials(req: CalculateRequest):
    """Executes deterministic financial math for spend profile."""
    graph_app = get_cardwise_app()
    spend_dict = req.dict()
    calc_res = graph_app.calculator.calculate_card_value(spend_dict)
    summary_md = graph_app.calculator.format_summary_markdown(calc_res)
    return {
        "calculation": calc_res,
        "summary_markdown": summary_md
    }


@app.get("/", response_class=HTMLResponse)
def get_ui():
    """Serves the rich CardWise AI interactive web dashboard."""
    html_content = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CardWise AI — Premium Credit Card Advisor</title>
    <!-- Tailwind CSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    colors: {
                        brand: {
                            gold: '#D4AF37',
                            platinum: '#E5E4E2',
                            sapphire: '#0F52BA',
                            dark: '#0B0F19',
                            cardDark: '#111827',
                            cardBorder: '#1F2937'
                        }
                    }
                }
            }
        }
    </script>
    <!-- Chart.js CDN -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <!-- Lucide Icons -->
    <script src="https://unpkg.com/lucide@latest"></script>
    <!-- Marked JS for Markdown Rendering -->
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        body {
            background-color: #0B0F19;
            color: #F3F4F6;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        .glass-panel {
            background: rgba(17, 24, 39, 0.75);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
        }
        .card-amex {
            background: linear-gradient(135deg, #2c3540 0%, #1e242d 50%, #12161c 100%);
            border: 1px solid rgba(229, 228, 226, 0.3);
            box-shadow: 0 10px 30px -10px rgba(229, 228, 226, 0.15);
        }
        .card-csr {
            background: linear-gradient(135deg, #0f2b48 0%, #0a1b30 50%, #050d1a 100%);
            border: 1px solid rgba(15, 82, 186, 0.4);
            box-shadow: 0 10px 30px -10px rgba(15, 82, 186, 0.25);
        }
        .glow-gold {
            box-shadow: 0 0 20px rgba(212, 175, 55, 0.15);
        }
        .glow-sapphire {
            box-shadow: 0 0 20px rgba(15, 82, 186, 0.25);
        }
        /* Markdown Prose Spacing & Typography Overrides */
        .prose h3 {
            margin-top: 0.5rem !important;
            margin-bottom: 1.25rem !important;
            color: #F3F4F6 !important;
            font-size: 1.05rem !important;
            font-weight: 700 !important;
        }
        .prose h4 {
            margin-top: 1.5rem !important;
            margin-bottom: 0.75rem !important;
            color: #F59E0B !important;
            font-size: 0.925rem !important;
            font-weight: 700 !important;
        }
        .prose p {
            margin-top: 0.75rem !important;
            margin-bottom: 1.25rem !important;
            line-height: 1.65 !important;
        }
    </style>
</head>
<body class="min-h-screen flex flex-col">

    <!-- Header / Navbar -->
    <header class="glass-panel sticky top-0 z-50 border-b border-gray-800 px-6 py-4 flex items-center justify-between">
        <div class="flex items-center space-x-3">
            <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-amber-500 via-yellow-400 to-blue-600 flex items-center justify-center font-extrabold text-xl text-black shadow-lg">
                CW
            </div>
            <div>
                <h1 class="text-xl font-bold tracking-tight text-white flex items-center gap-2">
                    CardWise <span class="text-xs px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-400 border border-amber-500/30 font-medium">AI Advisor</span>
                </h1>
                <p class="text-xs text-gray-400">RAG + Hybrid Retrieval (BM25 + FAISS) + Financial Optimizer</p>
            </div>
        </div>
        <div class="flex items-center space-x-4 text-xs text-gray-300">
            <span class="flex items-center gap-1 bg-gray-800/80 px-3 py-1.5 rounded-lg border border-gray-700">
                <i data-lucide="shield-check" class="w-4 h-4 text-emerald-400"></i> LangGraph Powered
            </span>
            <span class="flex items-center gap-1 bg-gray-800/80 px-3 py-1.5 rounded-lg border border-gray-700">
                <i data-lucide="activity" class="w-4 h-4 text-amber-400"></i> LangSmith Traced
            </span>
        </div>
    </header>

    <!-- Main Content Area -->
    <main class="flex-1 max-w-full px-8 w-full mx-auto p-6 grid grid-cols-1 lg:grid-cols-12 gap-6">

        <!-- Left Column: Visual Card Showcase & Spend Simulator -->
        <div class="lg:col-span-3 space-y-6">

            <!-- Card Visual Comparison Cards -->
            <div class="grid grid-cols-2 gap-4">
                <!-- Amex Platinum Visual Card -->
                <div class="card-amex rounded-2xl p-4 flex flex-col justify-between h-44 relative overflow-hidden transition transform hover:-translate-y-1">
                    <div class="flex justify-between items-start">
                        <div>
                            <p class="text-[10px] tracking-widest text-gray-300 font-semibold uppercase">American Express</p>
                            <h3 class="text-sm font-bold text-white">Platinum Card®</h3>
                        </div>
                        <span class="text-xs font-bold text-amber-300 bg-amber-400/10 px-2 py-0.5 rounded border border-amber-400/20">$895/yr</span>
                    </div>
                    <div>
                        <p class="text-[10px] text-gray-400">Top Perks</p>
                        <p class="text-xs text-gray-200 font-medium">$600 Hotel | $200 Flight | $400 Resy | 5X Points</p>
                    </div>
                </div>

                <!-- Chase Sapphire Reserve Visual Card -->
                <div class="card-csr rounded-2xl p-4 flex flex-col justify-between h-44 relative overflow-hidden transition transform hover:-translate-y-1">
                    <div class="flex justify-between items-start">
                        <div>
                            <p class="text-[10px] tracking-widest text-blue-300 font-semibold uppercase">Chase Sapphire</p>
                            <h3 class="text-sm font-bold text-white">Reserve®</h3>
                        </div>
                        <span class="text-xs font-bold text-blue-300 bg-blue-500/10 px-2 py-0.5 rounded border border-blue-500/20">$795/yr</span>
                    </div>
                    <div>
                        <p class="text-[10px] text-blue-300">Top Perks</p>
                        <p class="text-xs text-gray-200 font-medium">$300 Travel | 8X Portal | 4X Direct | Points Boost</p>
                    </div>
                </div>
            </div>

            <!-- Interactive Spend Simulator & Calculator -->
            <div class="glass-panel rounded-2xl p-5 border border-gray-800">
                <div class="flex justify-between items-center mb-4">
                    <h2 class="text-base font-bold text-white flex items-center gap-2">
                        <i data-lucide="sliders" class="w-4 h-4 text-amber-400"></i> Annual Spend Simulator
                    </h2>
                    <button onclick="resetSliders()" class="text-xs text-gray-400 hover:text-white transition">Reset</button>
                </div>

                <div class="space-y-3">
                    <div>
                        <div class="flex justify-between text-xs mb-1">
                            <span class="text-gray-300">Flights (Direct Airlines)</span>
                            <span class="font-bold text-amber-400" id="val-flights_direct">$12,000</span>
                        </div>
                        <input type="range" id="input-flights_direct" min="0" max="50000" step="500" value="12000" class="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-amber-400" oninput="updateCalculations()">
                    </div>

                    <div>
                        <div class="flex justify-between text-xs mb-1">
                            <span class="text-gray-300">Hotels (Travel Portal)</span>
                            <span class="font-bold text-blue-400" id="val-hotels_portal">$6,000</span>
                        </div>
                        <input type="range" id="input-hotels_portal" min="0" max="30000" step="500" value="6000" class="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-400" oninput="updateCalculations()">
                    </div>

                    <div>
                        <div class="flex justify-between text-xs mb-1">
                            <span class="text-gray-300">Dining & Restaurants</span>
                            <span class="font-bold text-emerald-400" id="val-dining">$8,000</span>
                        </div>
                        <input type="range" id="input-dining" min="0" max="30000" step="500" value="8000" class="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-emerald-400" oninput="updateCalculations()">
                    </div>

                    <div>
                        <div class="flex justify-between text-xs mb-1">
                            <span class="text-gray-300">Rideshare / Lyft</span>
                            <span class="font-bold text-purple-400" id="val-lyft">$1,200</span>
                        </div>
                        <input type="range" id="input-lyft" min="0" max="10000" step="100" value="1200" class="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-400" oninput="updateCalculations()">
                    </div>
                </div>

                <!-- Financial Chart -->
                <div class="mt-5 pt-4 border-t border-gray-800">
                    <h3 class="text-xs font-semibold text-gray-400 mb-2 uppercase tracking-wider">Net Annual Value Comparison</h3>
                    <div class="h-44">
                        <canvas id="financialChart"></canvas>
                    </div>
                    <div id="calc-recommendation-box" class="mt-3 p-3 rounded-xl bg-amber-500/10 border border-amber-500/20 text-xs text-amber-200">
                        Calculating recommendation...
                    </div>
                </div>
            </div>

        </div>

        <!-- Right Column: Conversational AI Advisor & RAG Inspector -->
        <div class="lg:col-span-9 flex flex-col space-y-6">

            <!-- Chat & Query Box -->
            <div class="glass-panel rounded-2xl p-6 border border-gray-800 flex-1 flex flex-col">
                <div class="flex justify-between items-center mb-4 pb-3 border-b border-gray-800">
                    <h2 class="text-base font-bold text-white flex items-center gap-2">
                        <i data-lucide="bot" class="w-5 h-5 text-amber-400"></i> CardWise AI Benefits Advisor
                    </h2>
                </div>

                <!-- Full-Width AI Advisor Answer Window -->
                <div class="mb-4 min-h-[380px]">
                    <div class="flex flex-col bg-gray-900/60 rounded-xl p-5 border border-gray-800">
                        <div class="flex items-center justify-between mb-3 pb-2 border-b border-gray-800/80">
                            <span class="text-xs font-bold text-amber-400 flex items-center gap-1.5">
                                <i data-lucide="sparkles" class="w-4 h-4"></i> Advisor Synthesis
                            </span>
                            <span class="text-[10px] text-gray-500 font-medium">LLM Answer</span>
                        </div>
                        <div id="chat-messages" class="flex-1 overflow-y-auto space-y-3 pr-2 max-h-[480px]">
                            <div class="bg-gray-800/80 p-3.5 rounded-xl border border-gray-700 text-xs text-gray-200 leading-relaxed">
                                Hello! I am <strong>CardWise AI</strong>. Ask me any question about Chase Sapphire Reserve and Amex Platinum benefits, hotel credits, or insurance policies!
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Input Box -->
                <form id="chat-form" onsubmit="handleChatSubmit(event)" class="relative">
                    <input type="text" id="user-input" placeholder="Ask about benefits, hotel credits, insurance, or compare cards..." class="w-full bg-gray-900/90 text-sm text-white pl-4 pr-12 py-3 rounded-xl border border-gray-700 focus:outline-none focus:border-amber-400 transition placeholder-gray-500">
                    <button type="submit" class="absolute right-2 top-2 w-8 h-8 bg-amber-500 hover:bg-amber-400 text-black font-bold rounded-lg flex items-center justify-center transition">
                        <i data-lucide="send" class="w-4 h-4"></i>
                    </button>
                </form>
            </div>

            <!-- Hybrid RAG Trace Inspector -->
            <div class="glass-panel rounded-2xl p-5 border border-gray-800">
                <div class="flex justify-between items-center mb-3">
                    <h3 class="text-xs font-bold text-gray-300 uppercase tracking-wider flex items-center gap-1.5">
                        <i data-lucide="layers" class="w-4 h-4 text-emerald-400"></i> Hybrid Retrieval Context Inspector (BM25 + FAISS) - Retrieved Policies
                    </h3>
                    <span id="rag-doc-count-bottom" class="text-[10px] text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">0 Chunks Retrieved</span>
                </div>
                <div id="rag-context-box" class="bg-gray-950 p-4 rounded-xl border border-gray-800 text-[11px] font-mono text-gray-300 min-h-[280px] max-h-[550px] overflow-y-auto whitespace-pre-wrap leading-relaxed">
// RAG Hybrid Retrieval context traces will appear here when you submit queries...
                </div>
            </div>

        </div>

    </main>

    <script>
        lucide.createIcons();

        let myChart = null;

        function initChart() {
            const ctx = document.getElementById('financialChart').getContext('2d');
            myChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: ['Chase Sapphire Reserve', 'Amex Platinum Card'],
                    datasets: [
                        {
                            label: 'Points Value ($)',
                            data: [0, 0],
                            backgroundColor: '#3B82F6'
                        },
                        {
                            label: 'Statement Credits ($)',
                            data: [0, 0],
                            backgroundColor: '#10B981'
                        },
                        {
                            label: 'Annual Fee (-$)',
                            data: [-795, -895],
                            backgroundColor: '#EF4444'
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { stacked: true, grid: { display: false }, ticks: { color: '#9CA3AF', font: { size: 10 } } },
                        y: { stacked: true, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#9CA3AF', font: { size: 10 } } }
                    },
                    plugins: {
                        legend: { labels: { color: '#D1D5DB', font: { size: 10 } } }
                    }
                }
            });
        }

        async function updateCalculations() {
            const inputs = {
                flights_direct: parseFloat(document.getElementById('input-flights_direct').value),
                hotels_portal: parseFloat(document.getElementById('input-hotels_portal').value),
                dining: parseFloat(document.getElementById('input-dining').value),
                lyft: parseFloat(document.getElementById('input-lyft').value)
            };

            document.getElementById('val-flights_direct').innerText = '$' + inputs.flights_direct.toLocaleString();
            document.getElementById('val-hotels_portal').innerText = '$' + inputs.hotels_portal.toLocaleString();
            document.getElementById('val-dining').innerText = '$' + inputs.dining.toLocaleString();
            document.getElementById('val-lyft').innerText = '$' + inputs.lyft.toLocaleString();

            try {
                const res = await fetch('/api/calculate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(inputs)
                });
                const data = await res.json();
                const calc = data.calculation;

                // Update Chart
                const csr = calc.chase_sapphire_reserve;
                const amex = calc.amex_platinum;

                myChart.data.datasets[0].data = [csr.points_dollar_value, amex.points_dollar_value];
                myChart.data.datasets[1].data = [csr.total_credits_value, amex.total_credits_value];
                myChart.update();

                // Update Recommendation Box
                const box = document.getElementById('calc-recommendation-box');
                box.innerHTML = `<strong>Winner: ${calc.recommendation}</strong><br><span class="text-[11px] opacity-90">${calc.recommendation_reason}</span><br><span class="font-mono text-[10px] text-gray-300">CSR Net Value: $${csr.net_annual_value.toFixed(2)} | Amex Net Value: $${amex.net_annual_value.toFixed(2)}</span>`;
            } catch (err) {
                console.error(err);
            }
        }

        function resetSliders() {
            document.getElementById('input-flights_direct').value = 12000;
            document.getElementById('input-hotels_portal').value = 6000;
            document.getElementById('input-dining').value = 8000;
            document.getElementById('input-lyft').value = 1200;
            updateCalculations();
        }

        async function handleChatSubmit(e) {
            e.preventDefault();
            const input = document.getElementById('user-input');
            const query = input.value.trim();
            if (!query) return;

            input.value = '';
            appendMessage('user', query);

            const spendProfile = {
                flights_direct: parseFloat(document.getElementById('input-flights_direct').value),
                hotels_portal: parseFloat(document.getElementById('input-hotels_portal').value),
                dining: parseFloat(document.getElementById('input-dining').value),
                lyft: parseFloat(document.getElementById('input-lyft').value)
            };

            try {
                const res = await fetch('/api/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query, spend_profile: spendProfile })
                });
                const data = await res.json();

                appendMessage('bot', data.final_answer);

                // Update Bottom RAG Inspector
                if (document.getElementById('rag-doc-count-bottom')) {
                    document.getElementById('rag-doc-count-bottom').innerText = `${data.retrieved_docs_count} Chunks Retrieved (${data.intent.toUpperCase()})`;
                }
                document.getElementById('rag-context-box').innerText = data.retrieved_context || 'No documents retrieved.';

            } catch (err) {
                appendMessage('bot', 'Error connecting to CardWise AI server.');
            }
        }

        function askQuickQuery(q) {
            document.getElementById('user-input').value = q;
            document.getElementById('chat-form').dispatchEvent(new Event('submit'));
        }

        function appendMessage(sender, text) {
            const container = document.getElementById('chat-messages');
            const wrapper = document.createElement('div');
            wrapper.className = 'flex items-start gap-3';

            const parsedText = marked.parse(text);

            if (sender === 'user') {
                wrapper.innerHTML = `
                    <div class="w-8 h-8 rounded-lg bg-amber-500 text-black font-bold flex items-center justify-center shrink-0">
                        <i data-lucide="user" class="w-4 h-4"></i>
                    </div>
                    <div class="bg-amber-500/10 text-amber-200 border border-amber-500/20 font-medium p-3.5 rounded-2xl rounded-tl-none text-xs leading-relaxed w-3/4 shadow-md">
                        ${queryEscape(text)}
                    </div>
                `;
            } else {
                wrapper.innerHTML = `
                    <div class="w-8 h-8 rounded-lg bg-amber-500/20 text-amber-400 border border-amber-500/30 flex items-center justify-center shrink-0">
                        <i data-lucide="sparkles" class="w-4 h-4"></i>
                    </div>
                    <div class="bg-gray-800/90 p-4 rounded-2xl rounded-tl-none border border-gray-700 text-xs text-gray-200 leading-relaxed w-3/4 max-w-none prose prose-invert prose-xs">
                        ${parsedText}
                    </div>
                `;
            }

            container.appendChild(wrapper);
            container.scrollTop = container.scrollHeight;
            lucide.createIcons();
        }

        function queryEscape(str) {
            return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
        }

        window.onload = () => {
            initChart();
            updateCalculations();
        };
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
