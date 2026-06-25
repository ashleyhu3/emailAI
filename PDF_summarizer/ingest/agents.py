"""
Re-export shim — kept for backward compatibility.

Agents 1–3 (extraction) live in extraction_agents.py
Agents 4–6 (query/RAG)  live in query_agents.py
"""

from ingest.extraction_agents import (
    EpsPeData,
    FinancialReportMetadata,
    run_agent1,
    run_agent2,
    run_agent3,
    extract_metadata,
)

from ingest.query_agents import (
    GatewayIntentParams,
    SQLFilterParams,
    QueryRoutingParams,
    run_agent4_gateway,
    run_agent4,
    run_agent5_query_composer,
    run_agent5a_sql_filter,
    run_agent5b_semantic_refiner,
    run_agent6_synthesis,
)

__all__ = [
    "EpsPeData", "FinancialReportMetadata",
    "run_agent1", "run_agent2", "run_agent3", "extract_metadata",
    "GatewayIntentParams", "SQLFilterParams", "QueryRoutingParams",
    "run_agent4_gateway", "run_agent4",
    "run_agent5_query_composer", "run_agent5a_sql_filter", "run_agent5b_semantic_refiner",
    "run_agent6_synthesis",
]
