from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional, TypedDict

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langgraph.graph import END, START, StateGraph
from langchain_mcp_adapters.client import MultiServerMCPClient

from shared_tools import (
    admanager_from_question,
    calculate_tool,
    current_time_tool,
    guardrail_check,
    github_tool,
    list_tables,
    news_tool,
    run_sql,
    sample_rows,
    describe_table,
    schema_summary,
    tail_logs,
    weather_tool,
    web_search_tool,
    wiki_tool,
    jobs_tool,
    write_log,
)

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.getenv("DB_PATH", str(BASE_DIR / "sales.db")))
LOG_FILE = Path(os.getenv("LOG_FILE", str(BASE_DIR / "langgraph_app.log")))
MCP_SERVER = Path(os.getenv("MCP_SERVER", str(BASE_DIR / "mcp_server.py")))
DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

UTILITY_INTENTS = {"weather", "time", "calc", "logs", "schema", "news", "github", "wiki", "web", "admanager", "jobs"}


class AgentResult(BaseModel):
    answer: str = ""
    sql: str = ""
    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    tools_used: list[str] = Field(default_factory=list)
    summary: str = ""


class GraphState(TypedDict, total=False):
    question: str
    mode: str
    intent: str
    blocked: bool
    block_reason: str
    result: dict[str, Any]
    comparison: dict[str, Any]
    sql_result: dict[str, Any]
    tool_result: dict[str, Any]


def utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def load_model():
    provider = os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER).strip().lower()
    temperature = float(os.getenv("LLM_TEMPERATURE", "0"))
    if provider == "openai" or os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL), temperature=temperature)
    from langchain_ollama import ChatOllama
    return ChatOllama(model=os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL), temperature=temperature)


def build_llm_registry() -> list[dict[str, Any]]:
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    ollama_base = os.getenv("OLLAMA_BASE_URL", "").strip().rstrip("/")
    registry: list[dict[str, Any]] = []
    if openai_key:
        registry += [
            {"id": "gpt4o", "label": "GPT-4o", "provider": "openai", "model": "gpt-4o", "enabled": True},
            {"id": "gpt4o_mini", "label": "GPT-4o-mini", "provider": "openai", "model": "gpt-4o-mini", "enabled": True},
        ]
    if anthropic_key:
        registry += [
            {"id": "claude_sonnet", "label": "Claude Sonnet", "provider": "anthropic", "model": "claude-sonnet-4-5", "enabled": True},
            {"id": "claude_haiku", "label": "Claude Haiku", "provider": "anthropic", "model": "claude-haiku-4-5-20251001", "enabled": True},
        ]
    if ollama_base:
        ok = False
        try:
            r = requests.get(f"{ollama_base}/api/tags", timeout=3)
            ok = r.status_code == 200
        except Exception:
            ok = False
        registry.append({"id": "ollama_llama3", "label": f"Ollama ({os.getenv('OLLAMA_MODEL', DEFAULT_OLLAMA_MODEL)})", "provider": "ollama", "model": os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL), "enabled": ok})
    return registry


def intent_of(question: str) -> str:
    q = question.lower()
    if re.search(r"\border\s+\d+\b", q) or "ad manager" in q or "admanager" in q:
        return "admanager"
    if any(k in q for k in ["weather", "temperature", "forecast"]):
        return "weather"
    if any(k in q for k in ["time", "current time", "utc"]):
        return "time"
    if any(k in q for k in ["calculate", "what is ", "*", "/", "percent", "%"]):
        return "calc"
    if any(k in q for k in ["log", "logs"]):
        return "logs"
    if any(k in q for k in ["schema", "table", "column", "columns", "describe"]):
        return "schema"
    if any(k in q for k in ["news", "latest news", "headlines"]):
        return "news"
    if any(k in q for k in ["github", "repo", "repository", "issue", "pull request"]):
        return "github"
    if any(k in q for k in ["job", "jobs", "career", "careers", "opening", "openings", "vacancy", "vacancies"]):
        return "jobs"
    if any(k in q for k in ["wikipedia", "wiki"]):
        return "wiki"
    if any(k in q for k in ["search web", "web search", "duckduckgo", "search the web"]):
        return "web"
    return "sql"


def direct_system_prompt(schema: str) -> str:
    return (
        "You are a SQLite SQL expert. Return ONLY one valid SQLite SELECT statement. "
        "No markdown, no explanation, no code fences, no semicolons.\n\n"
        f"Schema:\n{schema}\n\n"
        "Rules:\n"
        "- Use only tables and columns present in the schema.\n"
        "- Prefer clear joins when needed.\n"
        "- Do not invent columns.\n"
        "- Use read-only queries only."
    )


def validate_sql(sql: str) -> str:
    cleaned = re.sub(r"```(?:sql)?", "", sql, flags=re.IGNORECASE).replace("```", "").strip().rstrip(";").strip()
    if not cleaned:
        raise ValueError("LLM returned an empty SQL response.")
    if not re.match(r"^(SELECT|WITH)\b", cleaned, flags=re.IGNORECASE):
        raise ValueError(f"Query must start with SELECT or WITH. Got: {cleaned[:120]!r}")
    if re.search(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|DETACH|TRUNCATE)\b", cleaned, re.I):
        raise ValueError("Only read-only SELECT queries are allowed.")
    if ";" in cleaned:
        raise ValueError("Only one SQL statement is allowed.")
    return cleaned


def generate_sql(question: str, schema: str) -> tuple[str, int, float]:
    model = load_model()
    t0 = time.perf_counter()
    resp = model.invoke([("system", direct_system_prompt(schema)), ("user", question)])
    latency = round((time.perf_counter() - t0) * 1000, 1)
    text = getattr(resp, "content", str(resp))
    tokens = int(getattr(getattr(resp, "usage_metadata", None), "total_tokens", 0) or 0)
    return str(text).strip(), tokens, latency


def compare_models(question: str, schema: str) -> dict[str, Any]:
    enabled = [r for r in build_llm_registry() if r.get("enabled")]
    if not enabled:
        return {"status": "error", "message": "No LLMs are enabled.", "candidates": []}
    t0 = time.perf_counter()
    candidates = [score_candidate(call_provider_sql(llm, schema, question)) for llm in enabled]
    wall_ms = round((time.perf_counter() - t0) * 1000, 1)
    valid = [c for c in candidates if not c["disqualified"]]
    if not valid:
        return {"status": "error", "message": "All models failed to produce valid SQL.", "candidates": candidates, "candidate_errors": {c["llm_label"]: c["validation_error"] for c in candidates}, "wall_ms": wall_ms}
    winner = max(valid, key=lambda c: c["scores"]["total"])
    return {"status": "success", "winner": winner, "candidates": candidates, "wall_ms": wall_ms, "metadata": {"validation_status": "passed", "candidates_run": len(candidates), "candidates_valid": len(valid)}}


def score_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    if candidate["disqualified"]:
        candidate["scores"] = {"validity": 0, "speed": 0, "rows": 0, "total": 0}
        return candidate
    validity = 50
    speed = max(0, 30 - int(candidate.get("latency_ms", 9999) / 100))
    rows_score = min(20, candidate.get("row_count", 0) * 2)
    candidate["scores"] = {"validity": validity, "speed": speed, "rows": rows_score, "total": validity + speed + rows_score}
    return candidate


def call_provider_sql(llm: dict[str, Any], schema: str, question: str) -> dict[str, Any]:
    provider = llm["provider"]
    model = llm["model"]
    label = llm["label"]
    raw_sql = ""
    clean_sql = ""
    tokens = 0
    llm_ms = 0.0
    try:
        if provider == "openai":
            import openai
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
            t0 = time.perf_counter()
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": direct_system_prompt(schema)},
                    {"role": "user", "content": question},
                ],
                temperature=0,
            )
            llm_ms = round((time.perf_counter() - t0) * 1000, 1)
            raw_sql = resp.choices[0].message.content or ""
            tokens = resp.usage.total_tokens if resp.usage else 0
        elif provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
            t0 = time.perf_counter()
            msg = client.messages.create(
                model=model,
                max_tokens=512,
                system=direct_system_prompt(schema),
                messages=[{"role": "user", "content": question}],
            )
            llm_ms = round((time.perf_counter() - t0) * 1000, 1)
            raw_sql = msg.content[0].text if msg.content else ""
            tokens = (msg.usage.input_tokens or 0) + (msg.usage.output_tokens or 0)
        elif provider == "ollama":
            base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
            t0 = time.perf_counter()
            prompt = f"{direct_system_prompt(schema)}\n\nQuestion: {question}"
            r = requests.post(f"{base}/api/generate", json={"model": model, "prompt": prompt, "stream": False}, timeout=120)
            r.raise_for_status()
            llm_ms = round((time.perf_counter() - t0) * 1000, 1)
            raw_sql = r.json().get("response", "")
        else:
            raise ValueError(f"Unknown provider: {provider}")
        clean_sql = validate_sql(raw_sql)
        exec_result = run_sql(clean_sql)
        return {"id": llm.get("id", ""), "llm_label": label, "model": model, "provider": provider, "sql": clean_sql, "tokens": tokens, "latency_ms": llm_ms, "exec_ms": exec_result["exec_ms"], "rows": exec_result["rows"], "columns": exec_result["columns"], "row_count": exec_result["row_count"], "valid": True, "disqualified": False, "validation_error": None, "scores": {}}
    except Exception as e:
        return {"id": llm.get("id", ""), "llm_label": label, "model": model, "provider": provider, "sql": clean_sql or raw_sql, "tokens": tokens, "latency_ms": llm_ms, "exec_ms": 0, "rows": [], "columns": [], "row_count": 0, "valid": False, "disqualified": True, "validation_error": str(e), "scores": {}}


def fallback_single_model_answer(question: str) -> dict[str, Any]:
    model = load_model()
    t0 = time.perf_counter()
    try:
        resp = model.invoke([("system", "You are a concise assistant. Answer the user's question directly and briefly."), ("user", question)])
        answer = str(getattr(resp, "content", resp))
        tokens = int(getattr(getattr(resp, "usage_metadata", None), "total_tokens", 0) or 0)
        return {"status": "success", "answer": answer, "summary": "Single-model fallback response.", "sql": "", "columns": [], "rows": [], "row_count": 0, "tools_used": ["single_model_fallback"], "tokens": tokens, "latency_ms": round((time.perf_counter() - t0) * 1000, 1), "comparison": {}, "model_mode": "single"}
    except Exception as e:
        return {"status": "error", "message": str(e), "sql": "", "columns": [], "rows": [], "row_count": 0, "tools_used": ["single_model_fallback"], "comparison": {}, "model_mode": "single"}


async def load_mcp_tools():
    client = MultiServerMCPClient({"local": {"transport": "stdio", "command": sys.executable, "args": [str(MCP_SERVER)]}})
    return await client.get_tools()

def _normalize_rows(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        if isinstance(data.get("results"), list):
            return [r for r in data["results"] if isinstance(r, dict)]
        if isinstance(data.get("rows"), list):
            return [r for r in data["rows"] if isinstance(r, dict)]
        return [data]
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    return []

def local_utility(intent: str, question: str) -> dict[str, Any]:
    if intent == "time":
        value = current_time_tool()
        return {"kind": "time", "value": value, "answer": value, "tools_used": ["current_time"]}
    if intent == "calc":
        value = calculate_tool(question)
        return {"kind": "calc", "value": value, "answer": value, "tools_used": ["calculate"]}
    if intent == "weather":
        city = re.sub(r"(?i).*weather(?: in| for)?\s*", "", question).strip().rstrip("?")
        data = weather_tool(city or "Singapore")
        return {"kind": "weather", "value": data, "answer": "", "rows": _normalize_rows(data), "row_count": len(_normalize_rows(data)), "tools_used": ["weather"]}
    if intent == "schema":
        tables = list_tables()
        return {"kind": "schema", "value": {"tables": tables, "schema": schema_summary()}, "answer": "\n".join(tables), "tools_used": ["list_tables", "schema_summary"]}
    if intent == "logs":
        logs = tail_logs(20)
        return {"kind": "logs", "value": logs, "answer": f"{len(logs)} log entries", "tools_used": ["tail_logs"]}
    if intent == "news":
        topic = re.sub(r"(?i).*news(?: about| on| for)?\s*", "", question).strip().rstrip("?")
        data = news_tool(topic or question)
        return {"kind": "news", "value": data, "answer": "", "rows": _normalize_rows(data), "row_count": len(_normalize_rows(data)), "tools_used": ["news_tool"]}
    if intent == "github":
        query = re.sub(r"(?i).*(github|repo|repository|issue|pull request)\s*", "", question).strip().rstrip("?")
        data = github_tool(query or question)
        return {"kind": "github", "value": data, "answer": "", "rows": _normalize_rows(data), "row_count": len(_normalize_rows(data)), "tools_used": ["github_tool"]}
    if intent == "jobs":
        q = re.sub(r"(?i).*(job|jobs|career|careers|opening|openings|vacancy|vacancies|in singapore|singapore)\s*", "", question).strip().rstrip("?")
        data = jobs_tool(query=q or "AI Data Engineer", location="Singapore", country="sg")
        rows = _normalize_rows(data)
        return {"kind": "jobs", "value": data, "answer": "", "rows": rows, "row_count": len(rows), "tools_used": ["jobs_tool"]}
    if intent == "wiki":
        query = re.sub(r"(?i).*wiki(?:pedia)?(?: on| about)?\s*", "", question).strip().rstrip("?")
        data = wiki_tool(query or question)
        return {"kind": "wiki", "value": data, "answer": "", "rows": _normalize_rows(data), "row_count": len(_normalize_rows(data)), "tools_used": ["wiki_tool"]}
    if intent == "web":
        query = re.sub(r"(?i).*search(?: the)? web(?: for)?\s*", "", question).strip().rstrip("?")
        data = web_search_tool(query or question)
        return {"kind": "web", "value": data, "answer": "", "rows": _normalize_rows(data), "row_count": len(_normalize_rows(data)), "tools_used": ["web_search_tool"]}
    if intent == "admanager":
        data = admanager_from_question(question)
        return {"kind": "admanager", "value": data, "answer": "", "rows": _normalize_rows(data), "row_count": len(_normalize_rows(data)), "tools_used": ["admanager_from_question"]}
    raise ValueError("Unsupported tool intent.")


class GraphNodeResult(BaseModel):
    status: str = "success"
    answer: str = ""
    sql: str = ""
    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    tools_used: list[str] = Field(default_factory=list)
    summary: str = ""
    model_mode: str = "single"
    sql_status: str = ""
    models_executed: int = 0
    models_valid: int = 0
    winner: dict[str, Any] | None = None
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    comparison: dict[str, Any] = Field(default_factory=dict)
    message: str = ""
    kind: str = ""


def guardrail_node(state: GraphState) -> GraphState:
    reason = guardrail_check(state["question"])
    if reason:
        write_log("WARN", "GUARDRAIL", reason, {"question": state["question"]})
        return {"blocked": True, "block_reason": reason, "result": GraphNodeResult(status="error", message=reason).model_dump()}
    write_log("INFO", "GUARDRAIL", "passed", {"question": state["question"]})
    return {"blocked": False, "block_reason": ""}


def detect_intent_node(state: GraphState) -> GraphState:
    intent = intent_of(state["question"])
    write_log("INFO", "INTENT", "detected", {"intent": intent})
    return {"intent": intent}


def route_node(state: GraphState) -> str:
    if state.get("blocked"):
        return "blocked"
    intent = state.get("intent", "sql")
    if intent in UTILITY_INTENTS:
        return "utility"
    if state.get("mode") == "MCP Agent":
        return "mcp_agent"
    return "sql"


def utility_node(state: GraphState) -> GraphState:
    result = local_utility(state["intent"], state["question"])
    write_log("INFO", "UTILITY", "completed", {"intent": state["intent"]})
    return {"tool_result": result, "result": GraphNodeResult(**result, status="success").model_dump()}


def mcp_agent_node(state: GraphState) -> GraphState:
    async def _run() -> dict[str, Any]:
        model = load_model()
        tools = await load_mcp_tools()
        system_prompt = (
            "You are a careful analytics assistant. Use MCP tools for schema, SQL, weather, time, math, logs, news, GitHub, Wikipedia, web search, and Ad Manager order metrics. "
            "For database questions, inspect schema first, then write a read-only SQL query, then run it. Never invent tables or columns."
        )
        agent = create_agent(model, tools=tools, system_prompt=system_prompt, response_format=AgentResult)
        result = await agent.ainvoke({"messages": [{"role": "user", "content": state["question"]}]})
        result = result or {}
        structured = result.get("structured_response")
        if isinstance(structured, BaseModel):
            payload = structured.model_dump()
        elif isinstance(structured, dict):
            payload = structured
        else:
            msgs = result.get("messages", [])
            answer = ""
            if msgs:
                last = msgs[-1]
                answer = last.get("content", "") if isinstance(last, dict) else str(last)
            payload = {"answer": answer or "No structured response returned.", "sql": "", "columns": [], "rows": [], "row_count": 0, "tools_used": [], "summary": ""}
        payload["raw_messages"] = result.get("messages", [])
        payload["loaded_tools"] = [t.name for t in tools]
        return payload

    try:
        payload = asyncio.run(_run())
        payload["model_mode"] = "single"
        write_log("INFO", "MCP_AGENT", "completed", {"question": state["question"]})
        return {"result": {**payload, "status": "success"}}
    except Exception as e:
        write_log("ERROR", "MCP_AGENT", "failed", {"error": str(e)})
        return {"result": GraphNodeResult(status="error", message=str(e), model_mode="single").model_dump()}


def sql_node(state: GraphState) -> GraphState:
    schema = schema_summary()
    write_log("INFO", "SQL", "start", {"question": state["question"]})
    try:
        raw_sql, tokens, sql_ms = generate_sql(state["question"], schema)
        sql = validate_sql(raw_sql)
        sql_result = run_sql(sql)
        result = GraphNodeResult(
            status="success",
            sql=sql_result["sql"],
            columns=sql_result["columns"],
            rows=sql_result["rows"],
            row_count=sql_result["row_count"],
            tools_used=["schema_reader", "sql_generator", "sql_validator", "sql_executor"],
            summary="Direct SQL pipeline completed successfully.",
            model_mode="single",
            sql_status="passed",
        ).model_dump()
        result["tokens"] = tokens
        result["sql_ms"] = sql_ms
        write_log("INFO", "SQL", "completed", {"row_count": sql_result["row_count"], "sql_ms": sql_ms})
        return {"sql_result": sql_result, "result": result}
    except Exception as e:
        write_log("ERROR", "SQL", "failed", {"error": str(e)})
        fb = fallback_single_model_answer(state["question"])
        fb["sql_status"] = "failed"
        fb["message"] = str(e)
        return {"result": fb}


def sql_post_route(state: GraphState) -> str:
    if state.get("result", {}).get("sql_status") == "failed":
        return "done"
    if state.get("mode") == "Auto Compare":
        return "compare"
    return "done"


def compare_node(state: GraphState) -> GraphState:
    comp = compare_models(state["question"], schema_summary())
    if comp.get("status") == "success":
        result = state["result"]
        result.update({
            "comparison": comp,
            "model_mode": "multi",
            "models_executed": len(comp.get("candidates", [])),
            "models_valid": len([c for c in comp.get("candidates", []) if not c.get("disqualified")]),
            "winner": comp.get("winner"),
            "candidates": comp.get("candidates", []),
        })
        write_log("INFO", "COMPARE", "winner_selected", {"winner": comp.get("winner", {}).get("llm_label"), "candidates": len(comp.get("candidates", []))})
        return {"comparison": comp, "result": result}
    result = state["result"]
    result.update({"comparison": comp, "model_mode": "single"})
    write_log("WARN", "COMPARE", "failed", {"message": comp.get("message")})
    return {"comparison": comp, "result": result}


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("guardrail", guardrail_node)
    graph.add_node("detect_intent", detect_intent_node)
    graph.add_node("utility", utility_node)
    graph.add_node("mcp_agent", mcp_agent_node)
    graph.add_node("sql", sql_node)
    graph.add_node("compare", compare_node)

    graph.add_edge(START, "guardrail")
    graph.add_edge("guardrail", "detect_intent")
    graph.add_conditional_edges("detect_intent", route_node, {"blocked": END, "utility": "utility", "mcp_agent": "mcp_agent", "sql": "sql"})
    graph.add_conditional_edges("sql", sql_post_route, {"compare": "compare", "done": END})
    graph.add_edge("utility", END)
    graph.add_edge("mcp_agent", END)
    graph.add_edge("compare", END)
    return graph.compile()


@st.cache_resource(show_spinner=False)
def get_graph():
    return build_graph()


@st.cache_data(ttl=300, show_spinner=False)
def cached_schema():
    return schema_summary()


@st.cache_data(ttl=180, show_spinner=False)
def cached_compare(question: str):
    return compare_models(question, cached_schema())


def render_result(payload: dict[str, Any]) -> None:
    if payload.get("status") == "error":
        st.error(payload.get("message", "Unknown error"))
        return

    if payload.get("answer"):
        st.subheader("Answer")
        st.write(payload.get("answer"))

    if payload.get("summary"):
        st.subheader("Summary")
        st.write(payload.get("summary"))

    if payload.get("sql"):
        st.subheader("SQL")
        st.code(payload["sql"], language="sql")

    rows = payload.get("rows") or []
    if not rows:
        st.info("No rows returned.")
        return

    df = pd.DataFrame(rows)

    st.subheader("Results")

    # ---------- Filters ----------
    with st.expander("Filters", expanded=True):
        filter_df = df.copy()

        # Salary filter
        salary_cols = [c for c in ["salary_min", "salary_max"] if c in filter_df.columns]
        if salary_cols:
            salary_values = pd.to_numeric(
                pd.concat([filter_df[c] for c in salary_cols], axis=0),
                errors="coerce"
            ).dropna()

            if not salary_values.empty:
                min_salary = int(salary_values.min())
                max_salary = int(salary_values.max())

                salary_range = st.slider(
                    "Salary range",
                    min_value=min_salary,
                    max_value=max_salary,
                    value=(min_salary, max_salary),
                )

                if "salary_min" in filter_df.columns:
                    filter_df["salary_min"] = pd.to_numeric(filter_df["salary_min"], errors="coerce")
                if "salary_max" in filter_df.columns:
                    filter_df["salary_max"] = pd.to_numeric(filter_df["salary_max"], errors="coerce")

                if "salary_min" in filter_df.columns:
                    filter_df = filter_df[
                        filter_df["salary_min"].fillna(0).between(salary_range[0], salary_range[1])
                    ]
                elif "salary_max" in filter_df.columns:
                    filter_df = filter_df[
                        filter_df["salary_max"].fillna(0).between(salary_range[0], salary_range[1])
                    ]

        # Location filter
        if "location" in filter_df.columns:
            location_options = sorted(
                [x for x in filter_df["location"].dropna().astype(str).unique().tolist() if x.strip()]
            )
            if location_options:
                selected_locations = st.multiselect(
                    "Location",
                    options=location_options,
                    default=location_options,
                )
                if selected_locations:
                    filter_df = filter_df[filter_df["location"].astype(str).isin(selected_locations)]

    # ---------- Sorting ----------
    sort_cols = filter_df.columns.tolist()
    if sort_cols:
        c1, c2 = st.columns([2, 1])
        sort_column = c1.selectbox("Sort by", sort_cols, index=0)
        sort_order = c2.selectbox("Order", ["Descending", "Ascending"], index=0)

        sorted_df = filter_df.sort_values(
            by=sort_column,
            ascending=(sort_order == "Ascending"),
            kind="mergesort",
        )
    else:
        sorted_df = filter_df

    # ---------- Pagination ----------
    page_size = st.selectbox("Rows per page", [5, 10, 20, 50, 100], index=1)
    total_rows = len(sorted_df)
    total_pages = max(1, (total_rows + page_size - 1) // page_size)

    if "table_page" not in st.session_state:
        st.session_state.table_page = 1

    page_col1, page_col2, page_col3 = st.columns([1, 1, 2])
    with page_col1:
        if st.button("Prev", use_container_width=True, disabled=st.session_state.table_page <= 1):
            st.session_state.table_page -= 1
            st.rerun()

    with page_col2:
        if st.button("Next", use_container_width=True, disabled=st.session_state.table_page >= total_pages):
            st.session_state.table_page += 1
            st.rerun()

    with page_col3:
        st.caption(f"Page {st.session_state.table_page} of {total_pages}  |  {total_rows} rows")

    st.session_state.table_page = max(1, min(st.session_state.table_page, total_pages))

    start_idx = (st.session_state.table_page - 1) * page_size
    end_idx = start_idx + page_size
    page_df = sorted_df.iloc[start_idx:end_idx].copy()

    # ---------- CSV Export ----------
    csv_data = page_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Export current page to CSV",
        data=csv_data,
        file_name="results_page.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # ---------- Display ----------
    st.dataframe(page_df, use_container_width=True, hide_index=True)

def render_result_tmp(payload: dict[str, Any]) -> None:
    if payload.get("status") == "error":
        st.error(payload.get("message", "Unknown error"))
        return

    if payload.get("answer"):
        st.subheader("Answer")
        st.write(payload.get("answer"))
    if payload.get("summary"):
        st.subheader("Summary")
        st.write(payload.get("summary"))
    if payload.get("sql"):
        st.subheader("SQL")
        st.code(payload["sql"], language="sql")
    if payload.get("row_count") is not None:
        st.metric("Rows returned", payload.get("row_count", 0))
    rows = payload.get("rows") or []
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No rows returned.")


def render_comparison(comp: dict[str, Any]) -> None:
    if not comp:
        st.info("No comparison yet.")
        return
    if comp.get("status") == "error":
        st.error(comp.get("message", "Comparison failed"))
        if comp.get("candidate_errors"):
            st.json(comp["candidate_errors"])
        return
    winner = comp.get("winner")
    if winner:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Winner", winner.get("llm_label", "—"))
        c2.metric("Score", winner.get("scores", {}).get("total", 0))
        c3.metric("Latency (ms)", winner.get("latency_ms", 0))
        c4.metric("Rows", winner.get("row_count", 0))
        st.code(winner.get("sql", ""), language="sql")
    st.write(f"Executed: {comp.get('metadata', {}).get('candidates_run', len(comp.get('candidates', [])))}")
    st.write(f"Valid: {comp.get('metadata', {}).get('candidates_valid', len([c for c in comp.get('candidates', []) if not c.get('disqualified')]))}")
    cand = comp.get("candidates", [])
    if cand:
        df = pd.DataFrame(cand)
        cols = [c for c in ["llm_label", "provider", "model", "valid", "disqualified", "latency_ms", "exec_ms", "row_count", "validation_error", "scores"] if c in df.columns]
        st.dataframe(df[cols], use_container_width=True, hide_index=True)


def render_graph_flow():
    mermaid = """
graph TD
    START --> GUARDRAIL[guardrail]
    GUARDRAIL --> INTENT[detect_intent]
    INTENT -->|utility| UTILITY[utility]
    INTENT -->|MCP Agent| MCP[MCP Agent]
    INTENT -->|SQL| SQL[sql]
    SQL -->|Auto Compare| COMPARE[compare]
    SQL -->|done| END
    UTILITY --> END
    MCP --> END
    COMPARE --> END
    """
    st.code(mermaid.strip(), language="text")


st.set_page_config(page_title="LangGraph SQL + Tools Assistant", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")
st.title("LangGraph SQL + Tools Assistant")
st.caption("Auto Compare, Direct SQL, MCP Agent, Ad Manager order-id support, and job search.")

if "history" not in st.session_state:
    st.session_state.history = []
if "latest" not in st.session_state:
    st.session_state.latest = {}
if "mode" not in st.session_state:
    st.session_state.mode = "Auto Compare"
if "question_input" not in st.session_state:
    st.session_state.question_input = ""
if "show_graph" not in st.session_state:
    st.session_state.show_graph = False

with st.sidebar:
    st.header("Mode")
    st.session_state.mode = st.radio("Run mode", ["Auto Compare", "Direct SQL", "MCP Agent"], index=["Auto Compare", "Direct SQL", "MCP Agent"].index(st.session_state.mode))
    st.divider()
    st.subheader("Sample prompts")
    samples = [
        "Show all customers",
        "What is the total revenue?",
        "Show customer names with their orders",
        "What is the weather in Singapore?",
        "What is 25 * 48?",
        "What time is it now?",
        "Show the latest news about AI",
        "Search GitHub for langgraph",
        "Search Wikipedia for OpenAI",
        "Search the web for SQLite window functions",
        "Show me AI and Data Engineer jobs in Singapore",
        "Show metrics for order 3850440466",
        
    ]
    selected_prompt = st.selectbox(
        "Choose a sample prompt",
        options=samples,
        index=0
    )

    if st.button("Use prompt", use_container_width=True):
        st.session_state.question_input = selected_prompt
    st.divider()
    if st.button("Show Graph Flow", use_container_width=True):
        st.session_state.show_graph = not st.session_state.show_graph
    if st.button("Refresh logs", use_container_width=True):
        st.rerun()
    if st.button("Clear history", use_container_width=True):
        st.session_state.history = []
        st.session_state.latest = {}
        st.success("History cleared.")

question = st.text_area("Question", value=st.session_state.question_input, height=90, key="question_input")
col1, col2 = st.columns([1, 1])
run_pressed = col1.button("Run query", type="primary", use_container_width=True)
reset_pressed = col2.button("Reset view", use_container_width=True)

if reset_pressed:
    st.session_state.latest = {}
    st.rerun()

if st.session_state.show_graph:
    st.subheader("Graph Flow")
    render_graph_flow()

if run_pressed and question.strip():
    start = time.perf_counter()
    graph = get_graph()
    progress = st.progress(0, text="Starting...")
    status_box = st.status("Queued", expanded=True)
    try:
        with status_box:
            st.write(f"**Mode:** {st.session_state.mode}")
            st.write(f"**Question:** {question}")
        progress.progress(20, text="Guardrails and routing...")
        state: GraphState = {"question": question.strip(), "mode": st.session_state.mode}
        result_state = graph.invoke(state)
        progress.progress(100, text="Done")
        status_box.update(label="Completed", state="complete", expanded=False)
        latest = result_state.get("result", {})
        latest["wall_ms"] = round((time.perf_counter() - start) * 1000, 1)
        st.session_state.latest = latest
        st.session_state.history.insert(0, {"ts": utc_now(), "mode": st.session_state.mode, "question": question.strip(), "status": latest.get("status", "unknown"), "wall_ms": latest.get("wall_ms", 0), "model_mode": latest.get("model_mode", "single"), "models_executed": latest.get("models_executed", 0), "models_valid": latest.get("models_valid", 0), "row_count": latest.get("row_count", 0)})
        if latest.get("status") == "error":
            st.error(latest.get("message", "Failed"))
        else:
            st.success(f"Completed in {latest.get('wall_ms', 0)} ms")
    except Exception as e:
        progress.progress(100, text="Failed")
        status_box.update(label="Failed", state="error", expanded=True)
        st.error(str(e))
        st.session_state.latest = {"status": "error", "message": str(e), "comparison": {}, "model_mode": "single"}

history = st.session_state.history
suc = sum(1 for x in history if x.get("status") == "success")
avg_latency = round(sum((x.get("wall_ms") or 0) for x in history) / len(history), 1) if history else 0
avg_rows = round(sum((x.get("row_count") or 0) for x in history) / len(history), 1) if history else 0
m1, m2, m3, m4 = st.columns(4)
m1.metric("Runs", len(history))
m2.metric("Success rate", f"{(suc / len(history) * 100):.0f}%" if history else "0%")
m3.metric("Avg latency (ms)", avg_latency)
m4.metric("Avg rows", avg_rows)

tabs = st.tabs(["Result", "Comparison", "Logs", "History", "Schema"])
with tabs[0]:
    render_result(st.session_state.latest)
with tabs[1]:
    render_comparison(st.session_state.latest.get("comparison") or {})
with tabs[2]:
    logs = tail_logs(150)
    if logs:
        for item in reversed(logs):
            with st.expander(f"{item.get('level', 'LOG')} - {item.get('step', '')} - {item.get('message', '')}"):
                st.code(json.dumps(item, indent=2, ensure_ascii=False), language="json")
    else:
        st.info("No logs yet.")
with tabs[3]:
    if history:
        df = pd.DataFrame(history)
        cols = [c for c in ["ts", "mode", "question", "status", "model_mode", "models_executed", "models_valid", "row_count", "wall_ms"] if c in df.columns]
        st.dataframe(df[cols], use_container_width=True, hide_index=True)
    else:
        st.info("No history yet.")
with tabs[4]:
    st.code(cached_schema(), language="text")