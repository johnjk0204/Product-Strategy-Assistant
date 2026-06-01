from typing import TypedDict, Dict, Any, Optional


class AgentState(TypedDict):
    # Session
    session_id: str

    # Intermediate agent outputs
    customer_insights: str
    market_research: str
    competitor_analysis: str
    swot_analysis: str
    opportunity_assessment: str
    feature_priorities: str
    product_roadmap: str
    strategy_recommendations: str
    executive_summary: str

    # Final compiled report payload
    report_data: Dict[str, Any]

    # Progress tracking
    current_agent: str
    error: Optional[str]
    analysis_complete: bool
