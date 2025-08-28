

import os
import re
import json
from typing import Any, Dict, List, Optional, Tuple
from decimal import Decimal
from datetime import date, datetime

import pymysql
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

# LLM + Graph
from groq import Groq
from langgraph.graph import StateGraph, END



# ============ Load environment ============
load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME", "defaultdb")
DB_CA_CERT = os.getenv("DB_CA_CERT")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY missing in environment variables.")

# ============ FastAPI ============
app = FastAPI(title="Northwind NL→SQL API", version="1.3")

# Allow browser apps to call this API (adjust origins for prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


from decimal import Decimal
from datetime import date, datetime

def _json_sanitize(val):
    if isinstance(val, Decimal):
        return float(val)
    if isinstance(val, (date, datetime)):
        return val.isoformat()
    return val
# ============ DB utils ============
def get_connection():
    if not all([DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME, DB_CA_CERT]):
        raise RuntimeError("Database environment variables are not fully configured.")
    return pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        ssl={"ca": DB_CA_CERT},
        autocommit=True,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )

ALLOWED_TABLES = {"orders", "order_details", "products", "categories", "customers", "sales_facts"}

READ_ONLY_PATTERNS = [
    r"^\s*select\b",
    r"^\s*with\b",
    r"^\s*show\b",
    r"^\s*describe\b",
    r"^\s*explain\b",
]

def is_read_only_sql(sql: str) -> bool:
    """
    Allow only read-oriented statements. Block INSERT/UPDATE/DELETE/DDL/etc.
    """
    s = sql.strip().lower()
    if any(re.match(p, s) for p in READ_ONLY_PATTERNS):
        forbidden = ["insert", "update", "delete", "drop", "alter", "truncate", "create", "grant", "revoke"]
        return not any(f in s for f in forbidden)
    return False

def only_allowed_tables(sql: str) -> bool:
    """
    Naive allowlist: FROM/JOIN tables must be in ALLOWED_TABLES.
    For production, swap to a SQL parser.
    """
    s = re.sub(r"`|\"", "", sql.lower())
    # if it uses FROM/JOIN, ensure at least one allowed table is present
    if re.search(r"\b(from|join)\b", s) and not any(re.search(rf"\b{t}\b", s) for t in ALLOWED_TABLES):
        return False
    # extract table names after FROM/JOIN and ensure all are allowed
    suspects = re.findall(r"\bfrom\s+([a-zA-Z0-9_\.]+)", s) + re.findall(r"\bjoin\s+([a-zA-Z0-9_\.]+)", s)
    for name in suspects:
        short = name.split(".")[-1]  # strip schema prefix if exists
        if short not in ALLOWED_TABLES:
            return False
    return True

# ---------- JSON sanitization ----------
def _json_sanitize(val):
    if isinstance(val, Decimal):
        return float(val)
    if isinstance(val, (date, datetime)):
        return val.isoformat()
    return val

def _df_records_json_safe(df: pd.DataFrame) -> List[Dict[str, Any]]:
    records = df.to_dict(orient="records")
    return [{k: _json_sanitize(v) for k, v in row.items()} for row in records]

def run_sql(sql: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Safety-checked SQL runner (no pandas). Uses DictCursor to avoid any header/value mixups.
    Returns (columns, rows) with JSON-safe values.
    """
    if not is_read_only_sql(sql):
        raise HTTPException(status_code=400, detail="Only read-only (SELECT/SHOW/DESCRIBE/EXPLAIN) queries are allowed.")
    if not only_allowed_tables(sql):
        raise HTTPException(status_code=400, detail="Query references tables outside the allowed list.")

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            # Column names
            columns = [d[0] for d in (cur.description or [])]
            # Row dicts
            raw_rows = cur.fetchall()  # list[dict]
            # JSON-sanitize values
            rows = [
                {k: _json_sanitize(v) for k, v in row.items()}
                for row in raw_rows
            ]

    return columns, rows

# ============ LLM / LangGraph ============
client = Groq(api_key=GROQ_API_KEY)
# Use whatever Groq model you’ve enabled:
MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")  # or "llama-3.1-70b-versatile"
MAX_SQL_RETRIES = 2  # auto-repair attempts

SCHEMA_HINT = """
Tables:
- customers(customer_id, company_name, contact_name, country, region)
- categories(category_id, category_name)
- products(product_id, product_name, category_id, unit_price)
- orders(order_id, customer_id, order_date, ship_country, ship_region)
- order_details(order_id, product_id, unit_price, quantity, discount)
- sales_facts(view): order_id, order_date (DATE), order_ym (YYYY-MM), customer_region, customer_country,
  product_id, product_name, category_id, category_name, unit_price, quantity, discount, line_total

Key joins:
- orders.customer_id = customers.customer_id
- order_details.order_id = orders.order_id
- order_details.product_id = products.product_id
- products.category_id = categories.category_id
"""

FEW_SHOTS = [
    {
        "q": "Show me total sales for last month.",
        "sql": """
SELECT ROUND(SUM(line_total), 2) AS total_sales
FROM sales_facts
WHERE order_date >= DATE_FORMAT(CURDATE() - INTERVAL 1 MONTH, '%Y-%m-01')
  AND order_date <  DATE_FORMAT(CURDATE(), '%Y-%m-01');
""".strip()
    },
    {
        "q": "Which product category had the highest sales in Q2?",
        "sql": """
SELECT category_name, ROUND(SUM(line_total), 2) AS sales
FROM sales_facts
WHERE QUARTER(order_date) = 2 AND YEAR(order_date) = YEAR(CURDATE())
GROUP BY category_name
ORDER BY sales DESC
LIMIT 1;
""".strip()
    },
    {
        "q": "Show me a pie chart of sales by region.",
        "sql": """
SELECT customer_region AS label, ROUND(SUM(line_total), 2) AS value
FROM sales_facts
GROUP BY customer_region
ORDER BY value DESC;
""".strip()
    },
    {
        "q": "Monthly sales trend this year",
        "sql": """
SELECT order_ym AS month, ROUND(SUM(line_total), 2) AS sales
FROM sales_facts
WHERE YEAR(order_date) = YEAR(CURDATE())
GROUP BY order_ym
ORDER BY order_ym;
""".strip()
    },
    {
        "q": "What is the unit price of Chai?",
        "sql": """
SELECT CAST(unit_price AS DOUBLE) AS unit_price
FROM products
WHERE product_name = 'Chai';
""".strip()
    },
]

SYSTEM_SQL_WRITER = f"""
You are a senior analytics engineer generating SAFE MySQL SELECT queries for Northwind.
Return ONLY valid MySQL SQL (single statement, no markdown).
Use only these tables: {', '.join(sorted(ALLOWED_TABLES))}
Prefer the 'sales_facts' view when possible.
Never use INSERT/UPDATE/DELETE/DDL.
Use CURDATE(), YEAR(), QUARTER(), DATE_FORMAT(...).
Schema hints:
{SCHEMA_HINT}

Few-shot examples:
{json.dumps(FEW_SHOTS, indent=2)}
"""

REPAIR_SYSTEM = f"""
You are a senior analytics engineer fixing MySQL SELECT queries for the Northwind schema.
Rules:
- Return ONLY a valid MySQL read-only statement (SELECT/EXPLAIN/SHOW/DESCRIBE), single statement, no markdown.
- Never use INSERT/UPDATE/DELETE/DDL.
- Use only these tables: {', '.join(sorted(ALLOWED_TABLES))}.
- Prefer the 'sales_facts' view when possible.
- Keep the original question's intent identical; only fix errors.
Schema hints:
{SCHEMA_HINT}
"""

def llm_complete(system: str, user: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.1,
    )
    return resp.choices[0].message.content.strip()

def strip_sql_fences(text: str) -> str:
    return re.sub(r"^```sql\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE)

def repair_sql_with_error(question: str, bad_sql: str, error_msg: str) -> str:
    user = f"""
The previous SQL failed. Fix it.

Question:
{question}

Previous SQL:
{bad_sql}

Database error:
{error_msg}

Return ONLY the corrected MySQL SQL (single read-only statement).
"""
    fixed = llm_complete(REPAIR_SYSTEM, user)
    return strip_sql_fences(fixed)

# ============ Chart suggestion + insight ============
def decide_chart_and_insight(columns: List[str], rows: List[Dict[str, Any]], original_question: str) -> Dict[str, Any]:
    """
    Heuristic chart recommender + quick insight text.
    Supports: table, pie, bar, line, kpi (single cell).
    Returns: { chart, insight, suggested_type }
    """
    # KPI: single cell (1x1)
    if len(columns) == 1 and len(rows) == 1 and columns[0] in rows[0]:
        label = columns[0]
        value = rows[0][label]
        return {
            "chart": {"type": "kpi", "label": label, "value": value},
            "insight": f"{label} = {value}",
            "suggested_type": "kpi",
        }

    chart = {"type": "table", "data": rows}
    insight = "Here are the results."
    suggested_type = "table"

    if not rows:
        return {"chart": chart, "insight": "No data returned.", "suggested_type": suggested_type}

    cols_lower = [c.lower() for c in columns]
    data = rows

    # Pie: label/value
    if any(c in cols_lower for c in ["label", "category", "region", "customer_region"]) and \
       any(c in cols_lower for c in ["value","sales","total_sales","sum","count"]):
        label_key = next((c for c in columns if c.lower() in ["label","category","region","customer_region"]), columns[0])
        value_key = next((c for c in columns if c.lower() in ["value","sales","total_sales","sum","count"]), columns[-1])
        chart = {"type": "pie", "label": label_key, "value": value_key, "data": data}
        suggested_type = "pie"
        top = max(data, key=lambda r: (r.get(value_key) or 0))
        insight = f"{top.get(label_key)} contributes the most with {top.get(value_key)}."
        return {"chart": chart, "insight": insight, "suggested_type": suggested_type}

    # Line: time series
    if any("month" in c or "date" in c or "ym" in c for c in cols_lower) and \
       any(c in cols_lower for c in ["sales","total_sales","value","count"]):
        x_key = next((c for c in columns if c.lower() in ["month","order_ym","date"]), columns[0])
        y_key = next((c for c in columns if c.lower() in ["sales","total_sales","value","count"]), columns[-1])
        chart = {"type": "line", "x": x_key, "y": y_key, "data": data}
        suggested_type = "line"
        total = sum(float(r.get(y_key) or 0) for r in data)
        insight = f"Total over the period is {round(total,2)}; latest point: {data[-1].get(x_key)} = {data[-1].get(y_key)}."
        return {"chart": chart, "insight": insight, "suggested_type": suggested_type}

    # Bar: first col categorical, second numeric (common case)
    if len(columns) >= 2:
        x_key, y_key = columns[0], columns[1]
        try:
            _ = float(rows[0].get(y_key)) if rows and rows[0].get(y_key) is not None else None
            chart = {"type": "bar", "x": x_key, "y": y_key, "data": data}
            suggested_type = "bar"
            top = max(data, key=lambda r: (r.get(y_key) or 0))
            insight = f"Top {x_key} is {top.get(x_key)} with {top.get(y_key)}."
            return {"chart": chart, "insight": insight, "suggested_type": suggested_type}
        except Exception:
            pass

    # Fallback
    return {"chart": chart, "insight": insight, "suggested_type": suggested_type}

# ============ LangGraph state & nodes (with repair loop) ============
class NLQState(dict):
    """
    question: user text
    sql: generated (or repaired) SQL
    columns, rows: results
    error: last DB error (if any)
    attempt: retry counter
    chart, insight, suggested_type: presentation info
    """
    question: str
    sql: Optional[str]
    columns: Optional[List[str]]
    rows: Optional[List[Dict[str, Any]]]
    error: Optional[str]
    attempt: int
    chart: Optional[Dict[str, Any]]
    insight: Optional[str]
    suggested_type: Optional[str]

def node_generate_sql(state: NLQState) -> NLQState:
    q = state["question"].strip()
    prompt = f"User question:\n{q}\n\nReturn ONLY a MySQL SELECT query."
    sql = strip_sql_fences(llm_complete(SYSTEM_SQL_WRITER, prompt))
    state["sql"] = sql
    state["error"] = None
    state["attempt"] = 0
    return state

def node_execute_sql(state: NLQState) -> NLQState:
    try:
        cols, rows = run_sql(state.get("sql") or "")
        state["columns"], state["rows"] = cols, rows
        state["error"] = None
    except HTTPException as he:
        # Safety violations — don't try to repair
        raise he
    except Exception as e:
        state["error"] = str(e)
    return state

def node_repair_sql(state: NLQState) -> NLQState:
    if state.get("attempt", 0) >= MAX_SQL_RETRIES:
        return state
    fixed = repair_sql_with_error(state["question"], state.get("sql") or "", state.get("error") or "")
    state["sql"] = fixed
    state["attempt"] = state.get("attempt", 0) + 1
    return state

def node_analyze(state: NLQState) -> NLQState:
    result = decide_chart_and_insight(state.get("columns") or [], state.get("rows") or [], state["question"])
    state["chart"] = result["chart"]
    state["insight"] = result["insight"]
    state["suggested_type"] = result["suggested_type"]
    return state

# Graph with conditional edges
graph = StateGraph(NLQState)
graph.add_node("generate_sql", node_generate_sql)
graph.add_node("execute_sql", node_execute_sql)
graph.add_node("repair_sql", node_repair_sql)
graph.add_node("analyze", node_analyze)

graph.set_entry_point("generate_sql")
graph.add_edge("generate_sql", "execute_sql")

def needs_repair(state: NLQState) -> str:
    if state.get("error"):
        if state.get("attempt", 0) < MAX_SQL_RETRIES:
            return "repair_sql"
        else:
            return "end"
    return "analyze"

graph.add_conditional_edges("execute_sql", needs_repair, {"repair_sql": "repair_sql", "analyze": "analyze", "end": END})
graph.add_edge("repair_sql", "execute_sql")
graph.add_edge("analyze", END)

compiled_graph = graph.compile()

# ============ API models ============
class SQLRequest(BaseModel):
    sql: str = Field(..., description="Read-only SQL (SELECT/SHOW/DESCRIBE/EXPLAIN)")

class SQLResponse(BaseModel):
    columns: List[str]
    rows: List[Dict[str, Any]]

class NLQRequest(BaseModel):
    question: str

class NLQResponse(BaseModel):
    question: str
    sql: str
    columns: List[str]
    rows: List[Dict[str, Any]]
    chart: Dict[str, Any]
    suggested_chart_type: str
    insight: str

# ============ Routes ============
@app.get("/health")
def health():
    try:
        with get_connection() as _:
            pass
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/sql", response_model=SQLResponse)
def run_sql_endpoint(body: SQLRequest):
    try:
        cols, rows = run_sql(body.sql)
        return {"columns": cols, "rows": rows}
    except HTTPException as he:
        raise he
    except Exception as e:
        # Friendlier messages
        msg = str(e)
        if "Unknown column" in msg:
            msg = "One of the columns in your query does not exist."
        elif "Unknown table" in msg:
            msg = "The table name you used is invalid."
        elif "You have an error in your SQL syntax" in msg or "syntax" in msg.lower():
            msg = "There is a syntax error in your SQL query."
        raise HTTPException(status_code=400, detail=msg)

@app.post("/nlq", response_model=NLQResponse)
def nlq_endpoint(body: NLQRequest):
    """
    Full LangGraph pipeline with auto-repair attempts.
    """
    try:
        state: NLQState = {"question": body.question}
        out = compiled_graph.invoke(state)  # synchronous run
        if out.get("error"):
            # Exhausted retries
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Query failed after auto-repair attempts.",
                    "last_error": out.get("error"),
                    "last_sql": out.get("sql"),
                },
            )
        return {
            "question": body.question,
            "sql": out.get("sql") or "",
            "columns": out.get("columns") or [],
            "rows": out.get("rows") or [],
            "chart": out.get("chart") or {"type": "table", "data": []},
            "suggested_chart_type": out.get("suggested_type") or "table",
            "insight": out.get("insight") or "",
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


