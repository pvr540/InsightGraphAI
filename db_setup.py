# db_setup.py
"""
Creates sales.db with sample tables and data.
Run once: python db_setup.py
"""

import sqlite3
from logger import log_step, log_success, log_error

DB_PATH = "sales.db"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id  INTEGER PRIMARY KEY,
    name         TEXT    NOT NULL,
    email        TEXT,
    region       TEXT
);

CREATE TABLE IF NOT EXISTS products (
    product_id   INTEGER PRIMARY KEY,
    name         TEXT    NOT NULL,
    category     TEXT,
    unit_price   REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    order_id     INTEGER PRIMARY KEY,
    customer_id  INTEGER REFERENCES customers(customer_id),
    product_id   INTEGER REFERENCES products(product_id),
    quantity     INTEGER NOT NULL DEFAULT 1,
    total_price  REAL    NOT NULL,
    created_at   TEXT    NOT NULL
);
"""

SEED_SQL = """
INSERT OR IGNORE INTO customers VALUES (1,'Alice','alice@mail.com','North');
INSERT OR IGNORE INTO customers VALUES (2,'Bob',  'bob@mail.com',  'South');
INSERT OR IGNORE INTO customers VALUES (3,'Carol','carol@mail.com','East');
INSERT OR IGNORE INTO customers VALUES (4,'Dave', 'dave@mail.com', 'West');

INSERT OR IGNORE INTO products VALUES (1,'Laptop',  'Electronics',999.99);
INSERT OR IGNORE INTO products VALUES (2,'Phone',   'Electronics',599.99);
INSERT OR IGNORE INTO products VALUES (3,'Desk',    'Furniture',  299.99);
INSERT OR IGNORE INTO products VALUES (4,'Chair',   'Furniture',  149.99);
INSERT OR IGNORE INTO products VALUES (5,'Monitor', 'Electronics',399.99);
INSERT OR IGNORE INTO products VALUES (6,'Keyboard','Accessories', 89.99);
INSERT OR IGNORE INTO products VALUES (7,'Webcam',  'Accessories',129.99);

INSERT OR IGNORE INTO orders VALUES (1,1,1,2,1999.98,'2024-01-10');
INSERT OR IGNORE INTO orders VALUES (2,2,2,1, 599.99,'2024-01-15');
INSERT OR IGNORE INTO orders VALUES (3,1,5,3,1199.97,'2024-02-05');
INSERT OR IGNORE INTO orders VALUES (4,2,3,1, 299.99,'2024-02-20');
INSERT OR IGNORE INTO orders VALUES (5,1,4,4, 599.96,'2024-03-01');
INSERT OR IGNORE INTO orders VALUES (6,3,6,2, 179.98,'2024-03-10');
INSERT OR IGNORE INTO orders VALUES (7,4,7,1, 129.99,'2024-03-18');
INSERT OR IGNORE INTO orders VALUES (8,3,1,1, 999.99,'2024-04-02');
INSERT OR IGNORE INTO orders VALUES (9,4,2,2,1199.98,'2024-04-15');
INSERT OR IGNORE INTO orders VALUES(10,1,6,3, 269.97,'2024-04-20');
"""

def setup():
    log_step("DB_SETUP", f"Creating database: {DB_PATH}")
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.executescript(SCHEMA_SQL)
        conn.executescript(SEED_SQL)
        conn.commit()
        conn.close()
        log_success("DB_SETUP", "Database created and seeded successfully.",
                    {"db": DB_PATH, "tables": ["customers","products","orders"]})
    except Exception as e:
        log_error("DB_SETUP", f"Failed: {e}")
        raise

if __name__ == "__main__":
    setup()
