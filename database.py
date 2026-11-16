"""
Stores every assessment in a local SQLite file (loan_history.db) so the app can keep
a history and search it. Uses only Python's built-in sqlite3 module.
"""
import json
import os
import sqlite3
from contextlib import closing

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loan_history.db")


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with closing(_connect()) as conn:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS applications (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    reference     TEXT UNIQUE NOT NULL,
                    applicant_name TEXT NOT NULL,
                    age           INTEGER,
                    loan_purpose  TEXT,
                    loan_amount   REAL,
                    decision      TEXT NOT NULL,
                    probability   REAL NOT NULL,
                    model_name    TEXT,
                    created_at    TEXT NOT NULL,
                    inputs_json   TEXT NOT NULL,
                    derived_json  TEXT NOT NULL
                )
            """)


def _row_to_dict(row):
    record = dict(row)
    record["inputs"] = json.loads(record.pop("inputs_json"))
    record["derived"] = json.loads(record.pop("derived_json"))
    return record


def save_application(record):
    with closing(_connect()) as conn:
        with conn:
            conn.execute(
                """INSERT INTO applications
                   (reference, applicant_name, age, loan_purpose, loan_amount, decision,
                    probability, model_name, created_at, inputs_json, derived_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record["reference"], record["applicant_name"], record["age"],
                    record["loan_purpose"], record["loan_amount"], record["decision"],
                    record["probability"], record["model_name"], record["created_at"],
                    json.dumps(record["inputs"]), json.dumps(record["derived"]),
                ),
            )


def get_application(reference):
    with closing(_connect()) as conn:
        row = conn.execute("SELECT * FROM applications WHERE reference = ?",
                           (reference,)).fetchone()
    return _row_to_dict(row) if row else None


def search_applications(query="", decision="All"):
    """Search by applicant name or reference (partial, case-insensitive) and filter by decision."""
    sql = "SELECT * FROM applications WHERE 1 = 1"
    params = []
    if query:
        sql += " AND (applicant_name LIKE ? OR reference LIKE ?)"
        params += [f"%{query}%", f"%{query}%"]
    if decision in ("Approved", "Not approved"):
        sql += " AND decision = ?"
        params.append(decision)
    sql += " ORDER BY id DESC"
    with closing(_connect()) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_stats():
    with closing(_connect()) as conn:
        row = conn.execute("""
            SELECT COUNT(*) AS total,
                   SUM(CASE WHEN decision = 'Approved' THEN 1 ELSE 0 END) AS approved
            FROM applications
        """).fetchone()
    total = row["total"] or 0
    approved = row["approved"] or 0
    return {
        "total": total,
        "approved": approved,
        "not_approved": total - approved,
        "approval_rate": round(100 * approved / total, 1) if total else 0.0,
    }


def delete_application(reference):
    with closing(_connect()) as conn:
        with conn:
            cur = conn.execute("DELETE FROM applications WHERE reference = ?", (reference,))
    return cur.rowcount > 0
