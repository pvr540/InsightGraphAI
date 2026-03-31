from __future__ import annotations

import csv
import gzip
import json
import os
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.getenv("DB_PATH", str(BASE_DIR / "sales.db")))
LOG_FILE = Path(os.getenv("LOG_FILE", str(BASE_DIR / "langgraph_app.log")))
MAX_ROWS = int(os.getenv("MAX_ROWS", "100"))

ADM_NETWORK_CODE = os.getenv("ADM_NETWORK_CODE", "4654").strip()
ADM_START_DATE = os.getenv("ADM_START_DATE", "2025-10-01").strip()
ADM_END_DATE = os.getenv("ADM_END_DATE", "2025-11-11").strip()
ADM_API_VERSION = os.getenv("ADM_API_VERSION", "v202511").strip()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_log(level: str, step: str, message: str, data: dict | None = None) -> None:
    entry: dict[str, Any] = {"ts": utc_now(), "level": level, "step": step, "message": message}
    if data is not None:
        entry["data"] = data
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def connect() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def list_tables() -> list[str]:
    conn = connect()
    try:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
        tables = [r[0] for r in rows]
        write_log("INFO", "DB", "list_tables", {"tables": tables})
        return tables
    finally:
        conn.close()


def describe_table(table: str) -> list[dict[str, Any]]:
    conn = connect()
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        data = [dict(r) for r in rows]
        write_log("INFO", "DB", "describe_table", {"table": table, "columns": len(data)})
        return data
    finally:
        conn.close()


def sample_rows(table: str, limit: int = 5) -> list[dict[str, Any]]:
    conn = connect()
    try:
        cur = conn.execute(f"SELECT * FROM {table} LIMIT ?", (limit,))
        data = [dict(r) for r in cur.fetchall()]
        write_log("INFO", "DB", "sample_rows", {"table": table, "limit": limit, "rows": len(data)})
        return data
    finally:
        conn.close()


def schema_summary() -> str:
    conn = connect()
    try:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()]
        parts: list[str] = []
        for table in tables:
            cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
            col_defs = ", ".join(f"{c[1]} {c[2]}" for c in cols)
            parts.append(f"Table {table}({col_defs})")
        return "\n".join(parts)
    finally:
        conn.close()


def validate_sql(sql: str) -> str:
    cleaned = re.sub(r"```(?:sql)?", "", sql, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "").strip().rstrip(";").strip()
    if not cleaned:
        raise ValueError("LLM returned an empty SQL response.")
    if not re.match(r"^(SELECT|WITH)\b", cleaned, flags=re.IGNORECASE):
        raise ValueError(f"Query must start with SELECT or WITH. Got: {cleaned[:120]!r}")
    if re.search(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|DETACH|TRUNCATE)\b", cleaned, re.I):
        raise ValueError("Only read-only SELECT queries are allowed.")
    if ";" in cleaned:
        raise ValueError("Only one SQL statement is allowed.")
    return cleaned


def run_sql(query: str) -> dict[str, Any]:
    sql = validate_sql(query)
    conn = connect()
    t0 = time.perf_counter()
    try:
        cur = conn.execute(sql)
        rows = cur.fetchmany(MAX_ROWS)
        cols = [d[0] for d in cur.description] if cur.description else []
        data = [dict(r) for r in rows]
        elapsed = round((time.perf_counter() - t0) * 1000, 1)
        write_log("INFO", "DB", "run_sql", {"sql": sql, "row_count": len(data), "exec_ms": elapsed})
        return {"sql": sql, "columns": cols, "rows": data, "row_count": len(data), "exec_ms": elapsed}
    except sqlite3.Error as e:
        write_log("ERROR", "DB", "run_sql_failed", {"sql": sql, "error": str(e)})
        raise ValueError(f"SQL execution error: {e}") from e
    finally:
        conn.close()


def tail_logs(limit: int = 20) -> list[dict[str, Any]]:
    if not LOG_FILE.exists():
        return []
    lines = LOG_FILE.read_text(encoding="utf-8").splitlines()[-max(1, min(limit, 200)):]
    out: list[dict[str, Any]] = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except Exception:
            out.append({"raw": line})
    return out


def current_time_tool() -> str:
    value = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_log("INFO", "TIME", "current_time", {"value": value})
    return value


def calculate_tool(expression: str) -> str:
    expr = expression.lower()
    expr = re.sub(r"(\d+)%\s*of\s*(\d+)", r"(\1/100)*\2", expr)
    cleaned = re.sub(r"[^0-9+\-*/().]", "", expr)
    if not cleaned:
        raise ValueError("No valid mathematical expression found.")
    value = str(eval(cleaned, {"__builtins__": {}}, {}))  # noqa: S307
    write_log("INFO", "CALC", "calculate", {"expression": expression, "cleaned": cleaned, "value": value})
    return value


def weather_tool(city: str) -> dict[str, Any]:
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=20,
        )
        geo.raise_for_status()
        g = geo.json()
        if not g.get("results"):
            return {"city": city, "error": "City not found"}
        loc = g["results"][0]
        wx = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": loc["latitude"],
                "longitude": loc["longitude"],
                "current": "temperature_2m,weather_code,wind_speed_10m",
                "timezone": "auto",
            },
            timeout=20,
        )
        wx.raise_for_status()
        data = wx.json()
        payload = {
            "city": city,
            "name": loc.get("name"),
            "country": loc.get("country"),
            "latitude": loc.get("latitude"),
            "longitude": loc.get("longitude"),
            "current": data.get("current", {}),
        }
        write_log("INFO", "WEATHER", "weather", {"city": city})
        return payload
    except Exception as e:
        write_log("ERROR", "WEATHER", "weather_failed", {"city": city, "error": str(e)})
        return {"city": city, "error": str(e)}


def news_tool(topic: str, max_results: int = 5) -> dict[str, Any]:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return {"topic": topic, "error": "duckduckgo_search package is not installed."}
    try:
        with DDGS() as ddgs:
            results = list(ddgs.news(topic, max_results=max_results))
        write_log("INFO", "NEWS", "news_tool", {"topic": topic, "results": len(results)})
        return {"topic": topic, "results": results}
    except Exception as e:
        write_log("ERROR", "NEWS", "news_tool_failed", {"topic": topic, "error": str(e)})
        return {"topic": topic, "error": str(e)}


def github_tool(query: str, max_results: int = 5) -> dict[str, Any]:
    try:
        resp = requests.get(
            "https://api.github.com/search/repositories",
            params={"q": query, "sort": "stars", "order": "desc", "per_page": max_results},
            timeout=20,
            headers={"Accept": "application/vnd.github+json"},
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])[:max_results]
        results = [
            {
                "name": item.get("name"),
                "full_name": item.get("full_name"),
                "html_url": item.get("html_url"),
                "description": item.get("description"),
                "stars": item.get("stargazers_count", 0),
                "language": item.get("language"),
            }
            for item in items
        ]
        write_log("INFO", "GITHUB", "github_tool", {"query": query, "results": len(results)})
        return {"query": query, "results": results}
    except Exception as e:
        write_log("ERROR", "GITHUB", "github_tool_failed", {"query": query, "error": str(e)})
        return {"query": query, "error": str(e)}


def wiki_tool(query: str) -> dict[str, Any]:
    try:
        import wikipedia
    except ImportError:
        return {"query": query, "error": "wikipedia package is not installed."}
    try:
        matches = wikipedia.search(query, results=3)
        if not matches:
            return {"query": query, "error": "No Wikipedia results found."}
        page = wikipedia.page(matches[0], auto_suggest=False)
        payload = {"query": query, "title": page.title, "url": page.url, "summary": page.summary[:1500]}
        write_log("INFO", "WIKI", "wiki_tool", {"query": query, "title": page.title})
        return payload
    except Exception as e:
        write_log("ERROR", "WIKI", "wiki_tool_failed", {"query": query, "error": str(e)})
        return {"query": query, "error": str(e)}


def web_search_tool(query: str, max_results: int = 5) -> dict[str, Any]:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return {"query": query, "error": "duckduckgo_search package is not installed."}
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        write_log("INFO", "WEB", "web_search_tool", {"query": query, "results": len(results)})
        return {"query": query, "results": results}
    except Exception as e:
        write_log("ERROR", "WEB", "web_search_tool_failed", {"query": query, "error": str(e)})
        return {"query": query, "error": str(e)}


def jobs_tool(query: str = "AI Data Engineer", location: str = "Singapore", country: str = "sg", results_per_page: int = 10, page: int = 1, max_days_old: int = 14) -> dict[str, Any]:
    """Search current job openings using Adzuna.

    Required env vars:
      - ADZUNA_APP_ID
      - ADZUNA_APP_KEY
    """
    app_id = os.getenv("ADZUNA_APP_ID", "").strip()
    app_key = os.getenv("ADZUNA_APP_KEY", "").strip()
    if not app_id or not app_key:
        return {
            "query": query,
            "location": location,
            "country": country,
            "error": "Missing ADZUNA_APP_ID or ADZUNA_APP_KEY. Set both in your .env file.",
        }

    url = f"https://api.adzuna.com/v1/api/jobs/{country}/search/{page}"
    params = {
        "app_id": app_id,
        "app_key": app_key,
        "what": query,
        "where": location,
        "results_per_page": results_per_page,
        "max_days_old": max_days_old,
        "sort_by": "date",
        "salary_include_unknown": 1,
        "content-type": "application/json",
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        results = []
        for item in payload.get("results", []):
            results.append({
                "title": item.get("title"),
                "company": (item.get("company") or {}).get("display_name"),
                "location": (item.get("location") or {}).get("display_name"),
                "created": item.get("created"),
                "salary_min": item.get("salary_min"),
                "salary_max": item.get("salary_max"),
                "contract_type": item.get("contract_type"),
                "contract_time": item.get("contract_time"),
                "category": (item.get("category") or {}).get("label"),
                "description": item.get("description"),
                "apply_url": item.get("redirect_url"),
            })
        write_log("INFO", "JOBS", "jobs_tool", {"query": query, "location": location, "results": len(results)})
        return {"query": query, "location": location, "country": country, "count": payload.get("count", len(results)), "results": results}
    except Exception as e:
        write_log("ERROR", "JOBS", "jobs_tool_failed", {"query": query, "location": location, "error": str(e)})
        return {"query": query, "location": location, "country": country, "error": str(e)}


def guardrail_check(question: str) -> Optional[str]:
    if len(question) > 1000:
        return "Prompt too long."
    if re.search(r"\b(DELETE|DROP|UPDATE|INSERT|ALTER|CREATE|TRUNCATE)\b", question, re.I):
        return "Unsafe operation detected."
    return None


def extract_order_id(text: str) -> Optional[int]:
    match = re.search(r"\b\d{5,}\b", text)
    return int(match.group()) if match else None


def _load_adm_client():
    try:
        from googleads import ad_manager
    except ImportError as e:
        raise RuntimeError("Missing googleads package. Install requirements first.") from e
    return ad_manager.AdManagerClient.LoadFromStorage()


def get_admanager_order_metrics_json(
    question_or_order_id: str | int,
    start_date: dict[str, int] | None = None,
    end_date: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """
    Returns JSON rows for orders with impressions/clicks.
    User can pass a question containing an order id or the numeric order id directly.

    Requires ADM_NETWORK_CODE to be set in the environment.
    """
    if not ADM_NETWORK_CODE:
        raise ValueError("ADM_NETWORK_CODE is missing. Set it in your environment or .env file.")

    order_id = extract_order_id(str(question_or_order_id)) if not isinstance(question_or_order_id, int) else int(question_or_order_id)
    if not order_id:
        raise ValueError("No order id found in the input.")
    
    def _parse_date(s: str) -> dict:
        y, m, d = s.split("-")
        return {"year": int(y), "month": int(m), "day": int(d)}

    if start_date is None:
        start_date = _parse_date(ADM_START_DATE)   # uses env var, default "2024-01-01"
    if end_date is None:
        end_date = _parse_date(ADM_END_DATE) 

    # start_date = start_date or {"year": 2025, "month": 1, "day": 1}
    # end_date = end_date or {"year": 2026, "month": 12, "day": 31}

    client = _load_adm_client()
    report_service = client.GetService("ReportService", version=ADM_API_VERSION)

    report_query = {
        "dimensions": ["ORDER_ID", "ORDER_NAME"],
        "columns": [
            "AD_SERVER_IMPRESSIONS",
            "AD_SERVER_CLICKS",
            "AD_SERVER_CTR",
            "AD_SERVER_ALL_REVENUE",
        ],
        "dateRangeType": "CUSTOM_DATE",
        "startDate": start_date,
        "endDate": end_date,
        "statement": {"query": f"WHERE ORDER_ID = {order_id}"},
    }

    write_log("INFO", "AD_MANAGER", "run_report", {"order_id": order_id, "version": ADM_API_VERSION,"start_date": start_date, "end_date": end_date})

    report_job = {"reportQuery": report_query}
    job = report_service.runReportJob(report_job)
    job_id = job["id"] if isinstance(job, dict) else job.id

    # poll until done
    max_wait = int(os.getenv("ADM_REPORT_MAX_WAIT_SEC", "120"))
    poll_interval = float(os.getenv("ADM_REPORT_POLL_SEC", "2"))
    elapsed = 0.0

    while True:
        status = report_service.getReportJobStatus(job_id)
        status_str = str(status)
        if status_str == "COMPLETED":
            break
        if status_str == "FAILED":
            raise RuntimeError("Ad Manager report failed.")
        if elapsed >= max_wait:
            raise TimeoutError("Timed out waiting for Ad Manager report.")
        time.sleep(poll_interval)
        elapsed += poll_interval

    downloader = client.GetDataDownloader(version=ADM_API_VERSION)
    file_path = "/tmp/admanager_report.csv.gz"

    with open(file_path, "wb") as f:
        downloader.DownloadReportToFile(job_id, "CSV_DUMP", f)


    results: list[dict[str, Any]] = []
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        
        # Log actual headers on first row so we can see exact names
        headers = reader.fieldnames or []
        write_log("INFO", "AD_MANAGER", "csv_headers", {"headers": headers})
        
        for row in reader:
            # Try both prefixed and unprefixed column names
            def col(primary, fallback=""):
                return row.get(primary) or row.get(fallback) or None

            results.append({
                "order_id":   col("Dimension.ORDER_ID",               "ORDER_ID"),
                "order_name": col("Dimension.ORDER_NAME",             "ORDER_NAME"),
                "impressions": int(col("Column.AD_SERVER_IMPRESSIONS","AD_SERVER_IMPRESSIONS") or 0),
                "clicks":      int(col("Column.AD_SERVER_CLICKS",     "AD_SERVER_CLICKS") or 0),
                "ctr":         col("Column.AD_SERVER_CTR",            "AD_SERVER_CTR"),
                "revenue":     col("Column.AD_SERVER_ALL_REVENUE",    "AD_SERVER_ALL_REVENUE"),
            })    

    write_log("INFO", "AD_MANAGER", "report_ready", {"order_id": order_id, "rows": len(results)})
    return results


def admanager_from_question(question: str) -> list[dict[str, Any]]:
    return get_admanager_order_metrics_json(question)