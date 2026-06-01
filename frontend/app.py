"""
Streamlit frontend for the AI-Powered Product Strategy Assistant.
Connects to the FastAPI backend at http://localhost:8000.
"""

import time
from pathlib import Path

import requests
import streamlit as st

API = "http://localhost:8000"
SAMPLE_CSV = Path(__file__).parent.parent / "data" / "sample" / "sample_sales_data.csv"

# ---------------------------------------------------------------------------
# Page config & global style
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Product Strategy Assistant",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
/* ---- Global ---- */
[data-testid="stAppViewContainer"] { background: #f0f4f8; }
[data-testid="stSidebar"] { background: #1a3a5c; }
[data-testid="stSidebar"] * { color: #ecf0f1 !important; }
[data-testid="stSidebar"] .stButton button {
    background: #2980b9; color: white; border: none;
    border-radius: 8px; font-weight: 600; width: 100%;
}
[data-testid="stSidebar"] .stButton button:hover { background: #1a6fa0; }

/* ---- Agent badge ---- */
.agent-badge {
    display: inline-block;
    background: #2980b9; color: white;
    padding: 4px 12px; border-radius: 20px;
    font-size: 0.8rem; margin: 3px 2px;
}
.agent-done { background: #27ae60; }

/* ---- Metric cards ---- */
.metric-card {
    background: white; border-radius: 10px;
    padding: 16px; text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
.metric-value { font-size: 2rem; font-weight: 700; color: #1a3a5c; }
.metric-label { font-size: 0.85rem; color: #7f8c8d; margin-top: 4px; }

/* ---- Section cards ---- */
.section-card {
    background: white; border-radius: 10px;
    padding: 20px; margin-bottom: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    border-left: 4px solid #2980b9;
}

/* ---- Header ---- */
.main-header {
    background: linear-gradient(135deg, #1a3a5c 0%, #2980b9 100%);
    color: white; padding: 24px 32px; border-radius: 12px;
    margin-bottom: 24px;
}
.main-header h1 { margin: 0; font-size: 2rem; }
.main-header p { margin: 6px 0 0; opacity: 0.85; font-size: 1rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
for k, v in {
    "session_id": None,
    "analysis": None,
    "chat_history": [],
    "files_uploaded": False,
    "analysis_done": False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ---------------------------------------------------------------------------
# Helper: call backend
# ---------------------------------------------------------------------------
def _post(endpoint: str, **kwargs):
    try:
        r = requests.post(f"{API}{endpoint}", timeout=600, **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the backend. Make sure FastAPI is running on port 8000.")
        return None
    except requests.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", e.response.text)
        except Exception:
            detail = e.response.text if e.response else str(e)
        st.error(f"API error: {e}\n\n{detail}")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def _get(endpoint: str, **kwargs):
    try:
        r = requests.get(f"{API}{endpoint}", timeout=30, **kwargs)
        r.raise_for_status()
        return r
    except Exception as e:
        st.error(f"API error: {e}")
        return None


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
<div class="main-header">
  <h1>🎯 AI Product Strategy Assistant</h1>
  <p>Multi-agent AI system that transforms business data into actionable strategic insights</p>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📂 Data Upload")

    # Sample data loader
    if SAMPLE_CSV.exists():
        if st.button("⚡ Load Sample Sales Data"):
            with st.spinner("Uploading sample data…"):
                with open(SAMPLE_CSV, "rb") as f:
                    result = _post(
                        "/api/upload",
                        files=[("files", ("sample_sales_data.csv", f, "text/csv"))],
                    )
                if result:
                    st.session_state.session_id = result["session_id"]
                    st.session_state.files_uploaded = True
                    st.session_state.analysis = None
                    st.session_state.analysis_done = False
                    st.session_state.chat_history = []
                    st.success(f"Loaded! {result['total_chunks']} chunks indexed.")

    st.markdown("**— or upload your own files —**")
    uploaded = st.file_uploader(
        "Upload CSV / TXT / PDF / JSON",
        type=["csv", "txt", "pdf", "json"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded and st.button("📤 Upload Files"):
        with st.spinner("Processing & indexing…"):
            files_payload = [
                ("files", (f.name, f.getvalue(), f.type or "application/octet-stream"))
                for f in uploaded
            ]
            result = _post("/api/upload", files=files_payload)
            if result:
                st.session_state.session_id = result["session_id"]
                st.session_state.files_uploaded = True
                st.session_state.analysis = None
                st.session_state.analysis_done = False
                st.session_state.chat_history = []
                st.success(
                    f"✅ {result['files_processed']} file(s) uploaded\n"
                    f"{result['total_chunks']} chunks indexed"
                )

    # Upload status
    if st.session_state.files_uploaded and st.session_state.session_id:
        st.info(f"Session: `{st.session_state.session_id[:12]}…`")

    st.markdown("---")
    st.markdown("## 🤖 Multi-Agent Pipeline")

    agents = [
        "Customer Feedback Agent",
        "Market Research Agent",
        "Competitor Analysis Agent",
        "SWOT Analysis Agent",
        "Opportunity Assessment Agent",
        "Feature Prioritization Agent",
        "Product Roadmap Agent",
        "Strategy Recommendation Agent",
        "Executive Report Agent",
    ]
    done_count = 0
    if st.session_state.analysis_done:
        done_count = len(agents)
    for i, agent in enumerate(agents):
        icon = "✅" if i < done_count else "⏳"
        st.markdown(f"{icon} {agent}")

    st.markdown("---")

    # Run analysis button
    run_disabled = not st.session_state.files_uploaded
    if st.button(
        "🚀 Run Full Analysis",
        disabled=run_disabled,
        help="Upload files first, then click to start the 7-agent pipeline.",
    ):
        st.session_state.analysis = None
        st.session_state.analysis_done = False
        with st.spinner(
            "Running 7-agent analysis… this takes ~2-3 minutes. Please wait."
        ):
            result = _post(
                "/api/analyze",
                json={"session_id": st.session_state.session_id},
            )
        if result and result.get("status") == "complete":
            st.session_state.analysis = result.get("analysis", {})
            st.session_state.analysis_done = True
            st.success("Analysis complete!")
            st.rerun()
        elif result:
            st.error("Analysis returned an unexpected status.")

    # PDF download
    if st.session_state.analysis_done:
        st.markdown("---")
        if st.button("📄 Download PDF Report"):
            resp = _get(f"/api/download/{st.session_state.session_id}")
            if resp:
                st.download_button(
                    label="💾 Save PDF",
                    data=resp.content,
                    file_name=f"strategy_report_{st.session_state.session_id[:8]}.pdf",
                    mime="application/pdf",
                )


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
if not st.session_state.files_uploaded:
    # Landing / instructions
    c1, c2, c3 = st.columns(3)
    for col, icon, title, desc in [
        (c1, "📤", "1. Upload Data", "Upload CSV sales reports, customer reviews, market research docs, or competitor data."),
        (c2, "🤖", "2. Run Analysis", "Seven AI agents collaborate — customer insights, market research, competitor analysis, SWOT, feature prioritization, strategy, and executive summary."),
        (c3, "📊", "3. Get Insights", "Explore tabbed results, chat with the AI assistant, and download a boardroom-ready PDF report."),
    ]:
        with col:
            st.markdown(
                f"""<div class="section-card" style="text-align:center;">
                <div style="font-size:2.5rem;">{icon}</div>
                <h4 style="color:#1a3a5c;">{title}</h4>
                <p style="color:#555;">{desc}</p>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("### 📋 Supported Data Types")
    d1, d2, d3, d4 = st.columns(4)
    for col, label, example in [
        (d1, "📈 Sales Reports", "CSV with revenue, units, regions"),
        (d2, "💬 Customer Reviews", "CSV/TXT with feedback text"),
        (d3, "🔬 Market Research", "PDF or TXT documents"),
        (d4, "🏢 Competitor Info", "Any structured/unstructured text"),
    ]:
        with col:
            st.info(f"**{label}**\n{example}")

elif not st.session_state.analysis_done:
    st.info(
        "✅ Files uploaded and indexed. Click **Run Full Analysis** in the sidebar to start the 7-agent pipeline."
    )
    st.markdown("#### Session Summary")
    resp = _get(f"/api/status/{st.session_state.session_id}")
    if resp:
        data = resp.json()
        m1, m2 = st.columns(2)
        m1.metric("Files Uploaded", len(data.get("files", [])))
        m2.metric("Text Chunks Indexed", data.get("total_chunks", 0))
        if data.get("files"):
            st.markdown("**Uploaded files:**")
            for f in data["files"]:
                st.markdown(f"- `{f['filename']}` — {f['chunks']} chunks")

else:
    # -------------------------------------------------------------------------
    # Analysis results
    # -------------------------------------------------------------------------
    analysis = st.session_state.analysis or {}

    # Quick-glance metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown('<div class="metric-card"><div class="metric-value">9</div><div class="metric-label">AI Agents Run</div></div>', unsafe_allow_html=True)
    m2.markdown('<div class="metric-card"><div class="metric-value">✅</div><div class="metric-label">Analysis Complete</div></div>', unsafe_allow_html=True)
    m3.markdown('<div class="metric-card"><div class="metric-value">8</div><div class="metric-label">Reports Generated</div></div>', unsafe_allow_html=True)
    m4.markdown('<div class="metric-card"><div class="metric-value">PDF</div><div class="metric-label">Report Ready</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tabs = st.tabs([
        "📋 Executive Summary",
        "👥 Customer Insights",
        "📈 Market Research",
        "🏢 Competitor Analysis",
        "⚖️ SWOT Analysis",
        "🔍 Opportunity Assessment",
        "🎯 Feature Priorities",
        "🗺️ Product Roadmap",
        "🧭 Strategic Plan",
        "💬 Chat",
    ])

    def _render(tab, key, placeholder="Analysis not available."):
        with tab:
            content = analysis.get(key, "")
            if content:
                st.markdown(content)
            else:
                st.warning(placeholder)

    _render(tabs[0], "executive_summary")
    _render(tabs[1], "customer_insights")
    _render(tabs[2], "market_research")
    _render(tabs[3], "competitor_analysis")
    _render(tabs[4], "swot_analysis")
    _render(tabs[5], "opportunity_assessment")
    _render(tabs[6], "feature_priorities")
    _render(tabs[7], "product_roadmap")
    _render(tabs[8], "strategy_recommendations")

    # -------------------------------------------------------------------------
    # Chat tab
    # -------------------------------------------------------------------------
    with tabs[9]:
        st.markdown("### 💬 Ask the AI Strategy Assistant")
        st.caption("Ask any question about your data, the analysis results, or product strategy.")

        chat_container = st.container(height=420)
        with chat_container:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        if prompt := st.chat_input("e.g. Which product has the best profit margin?"):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

            with st.spinner("Thinking…"):
                result = _post(
                    "/api/chat",
                    json={
                        "session_id": st.session_state.session_id,
                        "message": prompt,
                        "chat_history": st.session_state.chat_history[:-1],
                    },
                )
            if result:
                reply = result.get("response", "Sorry, I could not generate a response.")
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
                with chat_container:
                    with st.chat_message("assistant"):
                        st.markdown(reply)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#95a5a6;font-size:0.8rem;'>"
    "AI Product Strategy Assistant • Powered by GPT-4o Mini + LangGraph + ChromaDB"
    "</p>",
    unsafe_allow_html=True,
)
