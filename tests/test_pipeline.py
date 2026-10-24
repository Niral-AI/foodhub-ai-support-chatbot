import os
import sqlite3
import py_compile

def test_database_exists():
    assert os.path.exists("customer_orders.db"), "customer_orders.db missing."

def test_database_schema():
    conn = sqlite3.connect("customer_orders.db")
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='orders';")
    assert cur.fetchone() is not None, "Table 'orders' not found."
    cur.execute("PRAGMA table_info(orders);")
    cols = [c[1] for c in cur.fetchall()]
    conn.close()
    for col in ["order_id", "cust_id", "order_time", "order_status", "payment_status"]:
        assert col in cols, f"Column '{col}' missing."

def test_app_compiles():
    assert py_compile.compile("app.py") is not None, "app.py failed compilation."
