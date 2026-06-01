"""
Multi-agent LangGraph workflow for Product Strategy Analysis.

7-agent sequential pipeline:
  1. Customer Feedback Agent
  2. Market Research Agent
  3. Competitor Analysis Agent
  4. SWOT Analysis Agent
  5. Feature Prioritization Agent
  6. Strategy Recommendation Agent
  7. Executive Report Agent
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any

import httpx
import openai
from langgraph.graph import StateGraph, END

from config import MODEL_NAME, OPENAI_API_KEY, OPENAI_BASE_URL
from graph.state import AgentState
from utils.vector_store import VectorStore

_client = openai.OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    http_client=httpx.Client(verify=False),
)
_executor = ThreadPoolExecutor(max_workers=2)

_CALL_DELAY = 1
_MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _call_llm(messages: list, max_tokens: int = 1800) -> str:
    for attempt in range(_MAX_RETRIES):
        try:
            resp = _client.chat.completions.create(
                model=MODEL_NAME,
                max_tokens=max_tokens,
                messages=messages,
                temperature=0.4,
            )
            return resp.choices[0].message.content
        except openai.RateLimitError:
            wait = 20 * (attempt + 1)
            time.sleep(wait)
        except Exception as e:
            if attempt == _MAX_RETRIES - 1:
                raise
            time.sleep(5)
    return "Analysis could not be completed due to API limits. Please retry."


def _retrieve(session_id: str, vs: VectorStore, query: str, n: int = 6) -> str:
    docs = vs.query(session_id, query, n_results=n)
    if not docs:
        docs = vs.get_all_documents(session_id, limit=20)
    return "\n\n---\n\n".join(docs[:5]) if docs else "No data provided."


# ---------------------------------------------------------------------------
# Agent node factories
# ---------------------------------------------------------------------------

def _make_customer_feedback_node(vs: VectorStore):
    def node(state: AgentState) -> dict:
        ctx = _retrieve(
            state["session_id"], vs,
            "customer reviews ratings feedback satisfaction complaints returns"
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a senior Customer Insights Analyst. "
                    "Generate structured, data-driven reports in markdown. "
                    "Always cite specific numbers from the data."
                ),
            },
            {
                "role": "user",
                "content": f"""Analyse the business data below and produce a Customer Insights Report.

BUSINESS DATA:
{ctx}

Include these sections:
## Executive Summary
## Sentiment Analysis (positive/neutral/negative % with evidence)
## Top 6 Customer Pain Points (ranked, with data)
## Top 5 Customer Praises (ranked, with data)
## Feature Requests & Suggestions
## Satisfaction Metrics (ratings by product, return rates, new customer trends)
## Customer Segment Insights (regional or product-level patterns)
## 5 Actionable Recommendations (specific and prioritised)

Be data-driven and cite exact numbers wherever available.""",
            },
        ]
        result = _call_llm(messages, max_tokens=500)
        time.sleep(_CALL_DELAY)
        return {"customer_insights": result, "current_agent": "Customer Feedback Agent"}
    return node


def _make_market_research_node(vs: VectorStore):
    def node(state: AgentState) -> dict:
        ctx = _retrieve(
            state["session_id"], vs,
            "market sales revenue growth trends regions categories profit"
        )
        ci = state.get("customer_insights", "")[:600]
        messages = [
            {
                "role": "system",
                "content": "You are a Market Research Analyst. Produce concise, numbered markdown reports backed by data.",
            },
            {
                "role": "user",
                "content": f"""Produce a Market Research Summary from the data below.

BUSINESS DATA:
{ctx}

CUSTOMER INSIGHTS (summary):
{ci}

Include:
## Market Overview (scale, position)
## Revenue & Growth Analysis (totals, trends, margins)
## Product Performance Ranking (best to worst with numbers)
## Regional Analysis (performance and opportunities by region)
## Category Performance (Electronics, Wearables, Audio, Accessories, Smart Home)
## Marketing Efficiency (spend vs. revenue/profit ROI signals)
## 5 Market Opportunities (specific, data-backed)
## Key Market Risks
## 5 Strategic Market Recommendations""",
            },
        ]
        result = _call_llm(messages, max_tokens=500)
        time.sleep(_CALL_DELAY)
        return {"market_research": result, "current_agent": "Market Research Agent"}
    return node


def _make_competitor_analysis_node(vs: VectorStore):
    def node(state: AgentState) -> dict:
        ctx = _retrieve(
            state["session_id"], vs,
            "competitor pricing market share benchmark product comparison differentiation"
        )
        mr = state.get("market_research", "")[:600]
        messages = [
            {
                "role": "system",
                "content": "You are a Competitive Intelligence Analyst. Derive competitive insights from available data when explicit competitor data is absent.",
            },
            {
                "role": "user",
                "content": f"""Generate a Competitor Analysis Report from the data below.

BUSINESS DATA:
{ctx}

MARKET RESEARCH SUMMARY:
{mr}

Include:
## Competitive Landscape Overview
## Product Positioning (where each product sits in the market)
## Price-Value Analysis (revenue per unit, value perception via ratings)
## Category-Level Competitive Assessment
## 5 Competitive Advantages to defend
## 4 Competitive Vulnerabilities
## 4 Differentiation Opportunities
## 5 Competitive Strategies

If no competitor data is present, derive competitive positioning from
sales performance, ratings, return rates, and market indicators.""",
            },
        ]
        result = _call_llm(messages, max_tokens=500)
        time.sleep(_CALL_DELAY)
        return {"competitor_analysis": result, "current_agent": "Competitor Analysis Agent"}
    return node


def _make_swot_node(vs: VectorStore):
    def node(state: AgentState) -> dict:
        ci = state.get("customer_insights", "")[:600]
        mr = state.get("market_research", "")[:600]
        ca = state.get("competitor_analysis", "")[:600]
        messages = [
            {
                "role": "system",
                "content": "You are a Strategic Business Analyst. Build rigorous, evidence-backed SWOT analyses.",
            },
            {
                "role": "user",
                "content": f"""Synthesise the analyses below into a comprehensive SWOT Analysis.

CUSTOMER INSIGHTS:
{ci}

MARKET RESEARCH:
{mr}

COMPETITOR ANALYSIS:
{ca}

Format:
## STRENGTHS (7-8 points with evidence)
## WEAKNESSES (6-7 points with evidence)
## OPPORTUNITIES (6-7 external factors)
## THREATS (5-6 external risks)
## SWOT Strategy Matrix
  - SO Strategies (3): use strengths to capture opportunities
  - ST Strategies (3): use strengths to counter threats
  - WO Strategies (3): overcome weaknesses using opportunities
  - WT Strategies (2-3): minimise weaknesses to avoid threats

Every point must tie back to specific data from the analyses above.""",
            },
        ]
        result = _call_llm(messages, max_tokens=500)
        time.sleep(_CALL_DELAY)
        return {"swot_analysis": result, "current_agent": "SWOT Analysis Agent"}
    return node


def _make_opportunity_assessment_node(vs: VectorStore):
    def node(state: AgentState) -> dict:
        ctx = _retrieve(
            state["session_id"], vs,
            "market opportunity gap unmet need growth segment demand trend"
        )
        ci = state.get("customer_insights", "")[:500]
        mr = state.get("market_research", "")[:500]
        ca = state.get("competitor_analysis", "")[:500]
        sw = state.get("swot_analysis", "")[:400]
        messages = [
            {
                "role": "system",
                "content": "You are a Product Opportunity Specialist. Identify and score specific, data-backed product and market opportunities.",
            },
            {
                "role": "user",
                "content": f"""Generate a Product Opportunity Assessment Report from the analyses below.

CUSTOMER INSIGHTS: {ci}
MARKET RESEARCH: {mr}
COMPETITOR ANALYSIS: {ca}
SWOT ANALYSIS: {sw}
ADDITIONAL DATA: {ctx[:300]}

Format:
## Opportunity Landscape Overview (2-3 sentences on the overall opportunity space)
## Top 5 High-Priority Opportunities (ranked by potential impact)
   For each: Opportunity Name | Evidence from data | Market size indicator | Competitive gap | Confidence (High/Med/Low)
## Market Gap Analysis (unmet needs and underserved customer segments with evidence)
## Blue Ocean Opportunities (areas with minimal competition where we can lead)
## Quick-Win Opportunities (executable in <30 days with high visibility)
## Opportunity Risk Assessment (top 3 risks per major opportunity)
## Opportunity Prioritization Matrix (Impact vs Effort 2×2 placement for each opportunity)
## Recommended Focus Areas (top 3 opportunities to pursue immediately with rationale)

Cite specific numbers and data points throughout.""",
            },
        ]
        result = _call_llm(messages, max_tokens=500)
        time.sleep(_CALL_DELAY)
        return {"opportunity_assessment": result, "current_agent": "Opportunity Assessment Agent"}
    return node


def _make_feature_prioritization_node(vs: VectorStore):
    def node(state: AgentState) -> dict:
        ctx = _retrieve(
            state["session_id"], vs,
            "feature requests product improvements enhancements new capabilities"
        )
        ci = state.get("customer_insights", "")[:500]
        mr = state.get("market_research", "")[:500]
        messages = [
            {
                "role": "system",
                "content": "You are a Senior Product Manager. Use the RICE framework (Reach × Impact × Confidence ÷ Effort, scored 1-10) for all prioritisation.",
            },
            {
                "role": "user",
                "content": f"""Generate Feature Prioritization Recommendations.

CUSTOMER INSIGHTS:
{ci}

MARKET RESEARCH:
{mr}

ADDITIONAL DATA:
{ctx[:400]}

Format:
## Priority 1 — Must-Have (0-3 months): 5-6 features with RICE scores
## Priority 2 — Should-Have (3-6 months): 4-5 features with RICE scores
## Priority 3 — Future Roadmap (6-12 months): 4-5 features with RICE scores
## Quick Wins (< 2-week effort, high visibility): 3-4 items
## Deprioritization Candidates: 2-3 items to reduce/drop with rationale
## Prioritization Philosophy: brief paragraph explaining the overall approach

For each feature: name | customer/market rationale | RICE estimate | expected impact metric.""",
            },
        ]
        result = _call_llm(messages, max_tokens=500)
        time.sleep(_CALL_DELAY)
        return {"feature_priorities": result, "current_agent": "Feature Prioritization Agent"}
    return node


def _make_product_roadmap_node(vs: VectorStore):
    def node(state: AgentState) -> dict:
        oa = state.get("opportunity_assessment", "")[:500]
        fp = state.get("feature_priorities", "")[:500]
        mr = state.get("market_research", "")[:400]
        sw = state.get("swot_analysis", "")[:300]
        messages = [
            {
                "role": "system",
                "content": "You are a Product Roadmap Strategist. Create clear, timeline-driven roadmaps that balance customer needs, business goals, and technical feasibility.",
            },
            {
                "role": "user",
                "content": f"""Create a detailed Product Roadmap based on the analyses below.

OPPORTUNITY ASSESSMENT: {oa}
FEATURE PRIORITIES: {fp}
MARKET RESEARCH: {mr}
SWOT: {sw}

Format:
## Roadmap Vision Statement (1-2 sentences on what success looks like in 12 months)
## Q1 Now — Month 3 (Theme + 4-5 deliverables with owner role and success metric)
## Q2 Month 3 — Month 6 (Theme + 4-5 deliverables with owner role and success metric)
## Q3-Q4 Month 6 — Month 12 (Theme + 3-4 strategic bets)
## 12-18 Month Horizon (2-3 longer-term directions and the bets behind them)
## Dependency Map (which items must ship before others unlock)
## Release Milestones & Go/No-Go Gates (3-4 checkpoints with measurable criteria)
## Resource Allocation by Quarter (team focus areas: engineering / design / marketing / data)
## Risks to the Roadmap (top 4 execution risks and contingency actions)

Be specific: name features, owners, and metrics — not generic themes.""",
            },
        ]
        result = _call_llm(messages, max_tokens=500)
        time.sleep(_CALL_DELAY)
        return {"product_roadmap": result, "current_agent": "Product Roadmap Agent"}
    return node


def _make_strategy_node(vs: VectorStore):
    def node(state: AgentState) -> dict:
        ci = state.get("customer_insights", "")[:400]
        mr = state.get("market_research", "")[:400]
        ca = state.get("competitor_analysis", "")[:300]
        sw = state.get("swot_analysis", "")[:400]
        fp = state.get("feature_priorities", "")[:300]
        oa = state.get("opportunity_assessment", "")[:300]
        rm = state.get("product_roadmap", "")[:300]
        messages = [
            {
                "role": "system",
                "content": "You are a Chief Product Officer. Build actionable, specific strategic plans — no vague guidance.",
            },
            {
                "role": "user",
                "content": f"""Create a Strategic Recommendations & Action Plan from the analyses below.

CUSTOMER INSIGHTS: {ci}
MARKET RESEARCH: {mr}
COMPETITIVE POSITION: {ca}
SWOT: {sw}
OPPORTUNITY ASSESSMENT: {oa}
FEATURE PRIORITIES: {fp}
PRODUCT ROADMAP: {rm}

Format:
## Strategic Vision (12-18 months): 3-4 sentences
## Three Strategic Pillars: name, description, why it matters
## 90-Day Sprint Plan: 6-8 actions (action | owner role | KPI | expected outcome)
## 6-Month Initiative Roadmap: 5-6 initiatives (name | resources | success metrics)
## 12-Month Strategic Bets: 3-4 major themes
## Key Performance Indicators: 8-10 specific, measurable KPIs
## Resource Requirements: indicative team / budget / tooling
## Risk Register: top 5 risks with mitigation strategies
## Go-to-Market Considerations: messaging, channels, targeting""",
            },
        ]
        result = _call_llm(messages, max_tokens=500)
        time.sleep(_CALL_DELAY)
        return {"strategy_recommendations": result, "current_agent": "Strategy Recommendation Agent"}
    return node


def _make_report_node(vs: VectorStore):
    def node(state: AgentState) -> dict:
        ci = state.get("customer_insights", "")[:400]
        mr = state.get("market_research", "")[:400]
        ca = state.get("competitor_analysis", "")[:300]
        sw = state.get("swot_analysis", "")[:400]
        fp = state.get("feature_priorities", "")[:300]
        sr = state.get("strategy_recommendations", "")[:400]
        messages = [
            {
                "role": "system",
                "content": "You are an Executive Communication Specialist. Write crisp, decision-ready board summaries.",
            },
            {
                "role": "user",
                "content": f"""Create a board-ready Executive Summary from the analyses below.

CUSTOMER INSIGHTS: {ci}
MARKET RESEARCH: {mr}
COMPETITOR ANALYSIS: {ca}
SWOT: {sw}
FEATURE PRIORITIES: {fp}
STRATEGIC PLAN: {sr}

Format:
## Business Situation (3 sentences)
## 5 Key Findings (data-backed bullet points)
## Top 3 Opportunities (act-on-now)
## Strategic Priorities — Next Quarter (5 must-dos)
## Investment Focus (where to concentrate budget and attention)
## Success Metrics (3-4 headline KPIs)
## Expected Outcomes in 6-12 Months

No jargon. Every sentence should be decision-ready for a C-suite audience.""",
            },
        ]
        result = _call_llm(messages, max_tokens=500)

        report_data = {
            "executive_summary": result,
            "customer_insights": state.get("customer_insights", ""),
            "market_research": state.get("market_research", ""),
            "competitor_analysis": state.get("competitor_analysis", ""),
            "swot_analysis": state.get("swot_analysis", ""),
            "opportunity_assessment": state.get("opportunity_assessment", ""),
            "feature_priorities": state.get("feature_priorities", ""),
            "product_roadmap": state.get("product_roadmap", ""),
            "strategy_recommendations": state.get("strategy_recommendations", ""),
        }
        return {
            "executive_summary": result,
            "report_data": report_data,
            "current_agent": "Executive Report Agent",
            "analysis_complete": True,
        }
    return node


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def _build_graph(vs: VectorStore):
    g = StateGraph(AgentState)

    g.add_node("customer_feedback",       _make_customer_feedback_node(vs))
    g.add_node("market_research",         _make_market_research_node(vs))
    g.add_node("competitor_analysis",     _make_competitor_analysis_node(vs))
    g.add_node("swot_analysis",           _make_swot_node(vs))
    g.add_node("opportunity_assessment",  _make_opportunity_assessment_node(vs))
    g.add_node("feature_prioritization",  _make_feature_prioritization_node(vs))
    g.add_node("product_roadmap",         _make_product_roadmap_node(vs))
    g.add_node("strategy_recommendation", _make_strategy_node(vs))
    g.add_node("executive_report",        _make_report_node(vs))

    g.set_entry_point("customer_feedback")
    g.add_edge("customer_feedback",       "market_research")
    g.add_edge("market_research",         "competitor_analysis")
    g.add_edge("competitor_analysis",     "swot_analysis")
    g.add_edge("swot_analysis",           "opportunity_assessment")
    g.add_edge("opportunity_assessment",  "feature_prioritization")
    g.add_edge("feature_prioritization",  "product_roadmap")
    g.add_edge("product_roadmap",         "strategy_recommendation")
    g.add_edge("strategy_recommendation", "executive_report")
    g.add_edge("executive_report",        END)

    return g.compile()


# ---------------------------------------------------------------------------
# Public async entry point (called from FastAPI)
# ---------------------------------------------------------------------------

async def run_analysis(session_id: str, vs: VectorStore) -> Dict[str, Any]:
    app_graph = _build_graph(vs)

    initial_state: AgentState = {
        "session_id": session_id,
        "customer_insights": "",
        "market_research": "",
        "competitor_analysis": "",
        "swot_analysis": "",
        "opportunity_assessment": "",
        "feature_priorities": "",
        "product_roadmap": "",
        "strategy_recommendations": "",
        "executive_summary": "",
        "report_data": {},
        "current_agent": "Initialising",
        "error": None,
        "analysis_complete": False,
    }

    loop = asyncio.get_event_loop()
    final_state = await loop.run_in_executor(
        _executor, lambda: app_graph.invoke(initial_state)
    )
    return final_state.get("report_data", {})
