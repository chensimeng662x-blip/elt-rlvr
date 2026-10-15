"""Generate the credential-free fixture task ``fixtures/mini_shop``.

Three source types (CSV, JSONL, SQLite), two data models, and the edge cases that
trip up naive SQL: a customer with no orders, a NULL quantity, a tie for the
favourite category, and a flag defined relative to an aggregate.
Ground truth is produced by the reference SQL below on DuckDB.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "mini_shop"

CUSTOMERS = pd.DataFrame(
    [
        (1, "Alice", "Seattle", "2024-01-05"),
        (2, "Bob", "Portland", "2024-01-09"),
        (3, "Carol", "Seattle", "2024-02-11"),
        (4, "Dan", "Boise", "2024-02-20"),
        (5, "Eve", "Portland", "2024-03-02"),
        (6, "Frank", "Seattle", "2024-03-15"),
        (7, "Grace", "Boise", "2024-04-01"),
        (8, "Heidi", "Portland", "2024-04-18"),
    ],
    columns=["customer_id", "name", "city", "signup_date"],
)

PRODUCTS = pd.DataFrame(
    [
        (101, "Laptop", "electronics", 1200.00),
        (102, "Mouse", "electronics", 25.50),
        (103, "Desk", "furniture", 300.00),
        (104, "Chair", "furniture", 150.00),
        (105, "Notebook", "stationery", 4.25),
        (106, "Pen", "stationery", 1.75),
    ],
    columns=["product_id", "product_name", "category", "unit_price"],
)

ORDERS = pd.DataFrame(
    [
        (1001, 1, 101, 1, "2024-02-01"),
        (1002, 1, 105, 10, "2024-02-03"),
        (1003, 2, 103, 1, "2024-02-10"),
        (1004, 2, 104, 1, "2024-02-10"),   # Bob: furniture 2 units
        (1005, 2, 106, 2, "2024-02-12"),   # Bob: stationery 2 units -> tie, 'furniture' wins alphabetically
        (1006, 3, 102, 3, "2024-03-01"),
        (1007, 3, 102, None, "2024-03-02"),  # NULL quantity counts as 0 units
        (1008, 4, 105, 5, "2024-03-05"),
        (1009, 4, 106, 20, "2024-03-06"),
        (1010, 5, 101, 2, "2024-03-20"),
        (1011, 5, 104, 4, "2024-03-21"),
        (1012, 6, 103, 2, "2024-04-02"),
        (1013, 6, 106, 1, "2024-04-02"),
        (1014, 7, 105, 1, "2024-04-10"),
        (1015, 7, 102, 1, "2024-04-11"),
    ],
    columns=["order_id", "customer_id", "product_id", "quantity", "order_date"],
)
ORDERS["quantity"] = ORDERS["quantity"].astype("Int64")

REFERENCE_SQL = {
    "customer_orders": """
with o as (
  select o.customer_id, o.product_id, coalesce(o.quantity, 0) as qty, p.category, p.unit_price
  from mini_shop.orders o join mini_shop.products p on o.product_id = p.product_id
), cat as (
  select customer_id, category, sum(qty) as units from o group by 1, 2
), fav as (
  select customer_id, category from (
    select customer_id, category,
           row_number() over (partition by customer_id order by units desc, category asc) as rn
    from cat) where rn = 1
)
select c.customer_id, c.name,
       count(o.product_id) as num_orders,
       round(coalesce(sum(o.qty * o.unit_price), 0), 2) as total_spent,
       f.category as favorite_category
from mini_shop.customers c
left join o on c.customer_id = o.customer_id
left join fav f on f.customer_id = c.customer_id
group by 1, 2, 5
order by 1
""",
    "category_sales": """
with s as (
  select p.category,
         sum(coalesce(o.quantity, 0)) as units_sold,
         round(sum(coalesce(o.quantity, 0) * p.unit_price), 2) as revenue
  from mini_shop.products p left join mini_shop.orders o on o.product_id = p.product_id
  group by 1)
select category, units_sold, revenue,
       case when revenue > (select avg(revenue) from s) then 1 else 0 end as above_average
from s order by 1
""",
}

DATA_MODEL_YAML = """models:
  - name: customer_orders
    description: One record per customer, including customers who never ordered.
    columns:
      - name: customer_id
        description: the customer id
      - name: name
        description: the customer name
      - name: num_orders
        description: the number of orders placed by the customer, 0 if none
      - name: total_spent
        description: the sum of quantity times unit_price over the customer's orders, treating a NULL quantity as 0, rounded to 2 decimals, 0 if no orders
      - name: favorite_category
        description: the product category with the most units bought by the customer, treating a NULL quantity as 0, with ties broken by the ascending order of the category name; NULL if the customer has no orders
  - name: category_sales
    description: One record per product category.
    columns:
      - name: category
        description: the product category
      - name: units_sold
        description: the total quantity ordered across products in the category, treating a NULL quantity as 0
      - name: revenue
        description: the sum of quantity times unit_price across products in the category, treating a NULL quantity as 0, rounded to 2 decimals
      - name: above_average
        description: set to 1 if the category's revenue is strictly greater than the average revenue across all categories, otherwise 0
"""

CONFIG_YAML = """duckdb:
  config:
    database: "{DESTINATION_DATABASE}"
    schema: "{DESTINATION_SCHEMA}"
    dbt_profile: elt
    dbt_target: dev
flat_files:
  - format: csv
    path: "{SOURCES}/customers.csv"
    table: customers
    sync_mode: full_refresh_overwrite
  - format: jsonl
    path: "{SOURCES}/orders.jsonl"
    table: orders
    sync_mode: full_refresh_overwrite
sqlite:
  config:
    path: "{SOURCES}/products.sqlite"
    tables:
      - products
    sync_mode: full_refresh_overwrite
"""

SCHEMA_DOCS = {
    "customers": [("customer_id", "unique id of the customer"), ("name", "customer name"), ("city", "city of residence"), ("signup_date", "date the customer signed up")],
    "orders": [("order_id", "unique id of the order"), ("customer_id", "id of the ordering customer"), ("product_id", "id of the product ordered"), ("quantity", "units ordered; may be NULL when unknown"), ("order_date", "date of the order")],
    "products": [("product_id", "unique id of the product"), ("product_name", "product name"), ("category", "product category"), ("unit_price", "price per unit")],
}


def main() -> None:
    (ROOT / "sources").mkdir(parents=True, exist_ok=True)
    (ROOT / "agent" / "schemas").mkdir(parents=True, exist_ok=True)
    (ROOT / "agent" / "documentation").mkdir(parents=True, exist_ok=True)
    (ROOT / "agent" / "elt").mkdir(parents=True, exist_ok=True)
    (ROOT / "grader" / "gt").mkdir(parents=True, exist_ok=True)
    (ROOT / "grader" / "sql").mkdir(parents=True, exist_ok=True)

    CUSTOMERS.to_csv(ROOT / "sources/customers.csv", index=False)
    ORDERS.to_json(ROOT / "sources/orders.jsonl", orient="records", lines=True)
    sq = ROOT / "sources/products.sqlite"
    if sq.exists():
        sq.unlink()
    with sqlite3.connect(sq) as c:
        PRODUCTS.to_sql("products", c, index=False)

    (ROOT / "agent/config.yaml").write_text(CONFIG_YAML)
    (ROOT / "agent/data_model.yaml").write_text(DATA_MODEL_YAML)
    (ROOT / "agent/elt/README.md").write_text("Write your EL configuration (el_config.yaml) and dbt project under this folder.\n")
    for table, cols in SCHEMA_DOCS.items():
        pd.DataFrame(cols, columns=["column_name", "description"]).to_csv(ROOT / f"agent/schemas/{table}.csv", index=False)

    con = duckdb.connect()
    con.execute("create schema mini_shop")
    con.register("customers_df", CUSTOMERS)
    con.register("orders_df", ORDERS)
    con.register("products_df", PRODUCTS)
    con.execute("create table mini_shop.customers as select * from customers_df")
    con.execute("create table mini_shop.orders as select * from orders_df")
    con.execute("create table mini_shop.products as select * from products_df")
    for name, sql in REFERENCE_SQL.items():
        df = con.execute(sql).df()
        df.to_csv(ROOT / f"grader/gt/{name}.csv", index=False)
        (ROOT / f"grader/sql/{name}.sql").write_text(f"select * from mini_shop.{name}\n")
    (ROOT / "grader/reference_models.sql").write_text(
        "\n\n".join(f"-- {k}\n{v.strip()}" for k, v in REFERENCE_SQL.items()) + "\n"
    )
    (ROOT / "grader/table.json").write_text(json.dumps({"customers": len(CUSTOMERS), "orders": len(ORDERS), "products": len(PRODUCTS)}, indent=2))
    (ROOT / "grader/sort_key.json").write_text(json.dumps({"customer_orders": ["customer_id"], "category_sales": ["category"]}, indent=2))
    print("fixture written to", ROOT)
    for name in REFERENCE_SQL:
        print(f"--- {name} ---")
        print((ROOT / f"grader/gt/{name}.csv").read_text())


if __name__ == "__main__":
    main()
