import json
import os
import time
import streamlit as st
import pandas as pd

from evaluation.scorer import score_experiment

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Agent Demonstration Interface",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "data", "results")

# --------------------------------------------------
# Styling (3D feel)
# --------------------------------------------------
st.markdown("""
<style>
body { background-color: #0b1220; }

.card {
    background: linear-gradient(145deg, #0f172a, #020617);
    border-radius: 18px;
    padding: 1.2rem;
    box-shadow: 0 20px 40px rgba(0,0,0,0.45);
    margin-bottom: 1rem;
}

.metric {
    font-size: 2rem;
    font-weight: 700;
}

.fade-in {
    animation: fadeIn 0.8s ease-in;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown("""
<div class="card fade-in">
<h1>🧠 Agent Demonstration Interface</h1>
<p>
Explainable evaluation of LLM-driven GIS agents with
intent inference, performance analysis, and reasoning replay.
</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("⚙️ Controls")

result_files = sorted(
    [f for f in os.listdir(RESULTS_DIR) if f.endswith(".json")],
    reverse=True
)

selected_file = st.sidebar.selectbox("Experiment run", result_files)

st.sidebar.markdown("---")
st.sidebar.subheader("🎚️ Policy Weights")

w_success = st.sidebar.slider("Success", 0.0, 1.0, 0.4, 0.05)
w_intent = st.sidebar.slider("Intent", 0.0, 1.0, 0.3, 0.05)
w_latency = st.sidebar.slider("Latency", 0.0, 1.0, 0.2, 0.05)
w_eff = st.sidebar.slider("Efficiency", 0.0, 1.0, 0.1, 0.05)

total = w_success + w_intent + w_latency + w_eff
if total > 0:
    w_success /= total
    w_intent /= total
    w_latency /= total
    w_eff /= total

# --------------------------------------------------
# Load results
# --------------------------------------------------
with open(os.path.join(RESULTS_DIR, selected_file), "r") as f:
    results = json.load(f)

df_all = pd.DataFrame(results)
scores = score_experiment(results)

# --------------------------------------------------
# Build query map (ID → TEXT)
# --------------------------------------------------
query_map = (
    df_all[["query_id", "query_text"]]
    .drop_duplicates()
    .set_index("query_id")["query_text"]
    .to_dict()
)

# --------------------------------------------------
# Dynamic scoring
# --------------------------------------------------
dynamic_scores = {}
for model, s in scores.items():
    runs = s["runs"]
    score = (
        (s["success"] / runs) * w_success +
        (s["intent_known"] / runs) * w_intent +
        max(0, 1 - s["avg_response_time"] / 120) * w_latency +
        max(0, 1 - s["avg_iterations"] / 10) * w_eff
    )
    dynamic_scores[model] = round(score, 3)

winner = max(dynamic_scores, key=dynamic_scores.get)

# --------------------------------------------------
# Score cards
# --------------------------------------------------
st.markdown("## 🏆 Model Scores")

cols = st.columns(len(dynamic_scores))
for col, (m, s) in zip(cols, dynamic_scores.items()):
    col.markdown(
        f"""
        <div class="card fade-in">
            <div class="metric">{s}</div>
            <div>{m}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.success(f"Best model under current policy → **{winner}**")

# --------------------------------------------------
# Intent distribution
# --------------------------------------------------
st.markdown("## 🧭 Intent Distribution by Model")
intent_df = (
    df_all.groupby(["model_name", "classified_intent"])
    .size()
    .unstack(fill_value=0)
)

st.bar_chart(intent_df)

# --------------------------------------------------
# Query-level comparison (TEXT, not IDs)
# --------------------------------------------------
st.markdown("## 🔍 Query-Level Comparison")

selected_query_id = st.selectbox(
    "Select query",
    options=list(query_map.keys()),
    format_func=lambda qid: query_map[qid]
)

qdf = df_all[df_all["query_id"] == selected_query_id]

cols = st.columns(len(qdf))
for col, (_, row) in zip(cols, qdf.iterrows()):
    col.markdown(
        f"""
        <div class="card fade-in">
            <h3>{row['model_name']}</h3>
            <p><b>Intent:</b> {row['classified_intent']}</p>
            <p><b>Latency:</b> {row['response_time_sec']} s</p>
            <p><b>Iterations:</b> {row['iterations_used']}</p>
            <p><b>Success:</b> {row['success']}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# --------------------------------------------------
# Trace replay (REAL observations)
# --------------------------------------------------
st.markdown("## 🧩 Agent Trace Replay")

model_choice = st.selectbox("Model", df_all["model_name"].unique())
query_choice = st.selectbox(
    "Query",
    options=list(query_map.keys()),
    format_func=lambda qid: query_map[qid],
    key="trace_query"
)

trace_row = df_all[
    (df_all["model_name"] == model_choice) &
    (df_all["query_id"] == query_choice)
].iloc[0]

trace = trace_row["iteration_details"]

if trace:
    if st.button("▶ Replay Reasoning"):
        placeholder = st.empty()
        for step in trace:
            with placeholder.container():
                st.markdown(
                    f"""
                    <div class="card fade-in">
                        <p><b>Intent:</b> {step['intent']}</p>
                        <p><b>Action:</b> {step['action']}</p>
                        <p><b>Observation:</b><br>{step['observation']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            time.sleep(1.2)
else:
    st.warning("No trace available.")

# --------------------------------------------------
# Explanation
# --------------------------------------------------
st.markdown("## 🧠 Decision Explanation")

st.info(
    f"""
    **{winner}** is preferred under the current policy due to its balance of
    success ({w_success:.2f}), intent understanding ({w_intent:.2f}),
    latency efficiency ({w_latency:.2f}), and reasoning efficiency ({w_eff:.2f}).
    Changing these priorities dynamically may alter the optimal model choice.
    """
)
