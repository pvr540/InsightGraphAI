from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from shared_tools import (
    admanager_from_question,
    calculate_tool,
    current_time_tool,
    describe_table,
    github_tool as shared_github_tool,
    list_tables,
    news_tool as shared_news_tool,
    run_sql,
    sample_rows,
    tail_logs as shared_tail_logs,
    weather_tool as shared_weather_tool,
    web_search_tool as shared_web_search_tool,
    wiki_tool as shared_wiki_tool,
    jobs_tool as shared_jobs_tool,
)

mcp = FastMCP("sales-mcp")


@mcp.tool()
def current_time() -> str:
    return current_time_tool()


@mcp.tool()
def calculate(expression: str) -> str:
    return calculate_tool(expression)


@mcp.tool()
def weather(city: str) -> dict[str, Any]:
    return shared_weather_tool(city)


@mcp.tool()
def news_tool(topic: str, max_results: int = 5) -> dict[str, Any]:
    return shared_news_tool(topic, max_results=max_results)


@mcp.tool()
def github_tool(query: str, max_results: int = 5) -> dict[str, Any]:
    return shared_github_tool(query, max_results=max_results)


@mcp.tool()
def wiki_tool(query: str) -> dict[str, Any]:
    return shared_wiki_tool(query)


@mcp.tool()
def jobs_tool(query: str = "AI Data Engineer", location: str = "Singapore", country: str = "sg", results_per_page: int = 10, page: int = 1, max_days_old: int = 14) -> dict[str, Any]:
    return shared_jobs_tool(query=query, location=location, country=country, results_per_page=results_per_page, page=page, max_days_old=max_days_old)


@mcp.tool()
def web_search_tool(query: str, max_results: int = 5) -> dict[str, Any]:
    return shared_web_search_tool(query, max_results=max_results)


@mcp.tool()
def tail_logs(limit: int = 20) -> list[dict[str, Any]]:
    return shared_tail_logs(limit)


@mcp.tool()
def list_db_tables() -> list[str]:
    return list_tables()


@mcp.tool()
def describe_db_table(table: str) -> list[dict[str, Any]]:
    return describe_table(table)


@mcp.tool()
def sample_db_rows(table: str, limit: int = 5) -> list[dict[str, Any]]:
    return sample_rows(table, limit)


@mcp.tool()
def execute_sql(query: str) -> dict[str, Any]:
    return run_sql(query)


@mcp.tool()
def admanager_orders(question: str) -> list[dict[str, Any]]:
    return admanager_from_question(question)


if __name__ == "__main__":
    mcp.run()