# Agent Demonstration Interface

An explainable evaluation framework for **LLM-powered GIS agents**.  
This system evaluates large language models not just by final answers, but by **intent understanding, reasoning behavior, execution efficiency, and policy-driven performance trade-offs**.

The framework supports multi-model comparison (Gemma-3, Mistral, LLaMA-3), reasoning trace replay, dynamic scoring, and an interactive Streamlit interface.

---

## 🔍 Key Features

- LLM-based **intent classification**
- ReAct-style **agent reasoning pipeline**
- **Multi-model benchmarking** (Gemma-3, Mistral, LLaMA-3)
- Policy-driven **dynamic scoring**
- **Step-by-step reasoning trace replay**
- Query-level and model-level comparison
- Interactive **3D Streamlit UI**
- Explainable, research-ready design

---

## 📁 Project Structure

```
AgentDemonstrationInterface/
│
├── run.py                     # Streamlit interface (main entry point)
│
├── enhanced_react_agent.py    # Intent-aware GIS agent with traceable reasoning
│
├── config/
│   ├── models.yaml            # Model definitions (Gemma, Mistral, LLaMA)
│   └── evaluation.yaml        # Scoring thresholds and defaults
│
├── data/
│   ├── dummy6_queries.json    # Benchmark GIS queries
│   └── results/               # Generated experiment results (gitignored)
│
├── evaluation/
│   ├── scorer.py              # Policy-based scoring logic
│   ├── metrics.py             # Performance metrics
│   └── reasoning_metrics.py   # Reasoning quality metrics
│
├── runners/
│   ├── model_runner.py        # Runs one model on one query
│   └── batch_runner.py        # Runs all models on all queries
│
├── utils/
│   ├── intent_classifier.py   # LLM-based intent classification
│   ├── timers.py              # Timing utilities
│   └── validators.py          # Safety and schema checks
│
└── .gitignore
```

---

## 🧠 System Workflow

1. User queries are loaded from a benchmark dataset.
2. Each query is classified by the LLM to infer **intent**.
3. An enhanced ReAct agent performs reasoning and generates observations.
4. Execution traces and metrics are logged.
5. Models are scored using a **policy-weighted evaluation function**.
6. Results are visualized in an interactive Streamlit interface.

---

## ⚙️ Prerequisites

- **Python 3.10+**
- **Git**
- **Ollama** (for running local LLMs)

---

## 🧩 Install Ollama & Models

### 1️⃣ Install Ollama

Download from: https://ollama.com

Ensure it is running:

```bash
ollama serve
```

### 2️⃣ Pull required models

```bash
ollama pull gemma3:latest
ollama pull mistral:latest
ollama pull llama3:latest
```

Verify:

```bash
ollama list
```

---

## 🐍 Python Setup

### 1️⃣ Create virtual environment (recommended)

```bash
python -m venv venv
```

Activate:

```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not present:

```bash
pip install streamlit pandas langchain langchain-ollama pyyaml
```

---

## 🚀 Run Experiments (Generate Results)

From the project root:

```bash
python -m runners.batch_runner
```

This will:

- Run all models on all benchmark queries
- Generate a timestamped JSON file in: `data/results/`

---

## 🖥️ Launch the Interface

```bash
python -m streamlit run run.py
```

Open your browser at: `http://localhost:8501`

---

## 🎛️ Using the Interface

- Select an experiment run from the sidebar
- Adjust scoring policy weights dynamically
- Compare models across intent, latency, and efficiency
- Drill down to query-level behavior
- Replay agent reasoning step by step
- Observe how model preference changes with policy

---

## 🧪 Notes on Observations

The current system generates **model-based geospatial observations** (textual reasoning outputs). The architecture is designed to plug in real GIS APIs or spatial databases without changing the evaluation pipeline.

---

## 🎓 Academic Use & Evaluation

This project is suitable for:

- Research demonstrations
- Model comparison studies
- Explainable AI experiments
- Agentic LLM system evaluation

---

## 📌 License

This project is intended for **academic and research use**.
