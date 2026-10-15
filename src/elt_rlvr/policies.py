"""Scripted policies: sequences of tool calls that drive an ``Episode`` without a
language model. Used by the credential-free integration test and the
reward-hacking audit. Each policy yields ``(tool_name, arguments)`` pairs.

- ``oracle``          : correct pipeline -> reward 1.0
- ``no_submit``       : correct pipeline, never submits -> 0
- ``el_only``         : loads raw tables, writes no models, submits -> el_weight
- ``wrong_models``    : loads, writes models with the easy columns right and the
                        hard ones wrong -> partial column credit only
- ``views_not_tables``: correct SQL materialised as views -> not graded as tables
- ``fake_load``       : bypasses run_el and writes raw tables straight into the
                        warehouse with correct row counts -> provenance fails -> 0
- ``append_twice``    : runs EL twice in append mode -> row counts double -> 0 for EL
"""

from __future__ import annotations

from collections.abc import Iterator

import yaml

Step = tuple[str, dict]


def _el_config(config: dict) -> str:
    dest = config["duckdb"]["config"]
    sources = []
    for f in config.get("flat_files", []):
        sources.append({"connector": "file", "file_format": f["format"], "url": f"file://{f['path']}",
                        "stream": f["table"], "sync_mode": f["sync_mode"]})
    sq = config.get("sqlite", {}).get("config")
    if sq:
        sources.append({"connector": "sqlite", "database_path": sq["path"],
                        "streams": [{"name": t, "sync_mode": sq["sync_mode"]} for t in sq["tables"]]})
    return yaml.safe_dump({"version": 1, "destination": {"connector": "duckdb", "database_path": dest["database"],
                                                        "schema": dest["schema"]}, "sources": sources}, sort_keys=False)


def _heredoc(path: str, content: str) -> str:
    return f"mkdir -p $(dirname {path}) && cat > {path} <<'__EOF__'\n{content}\n__EOF__"


DBT_PROJECT = """name: elt_models
version: '1.0'
config-version: 2
profile: elt
model-paths: ["models"]
models:
  elt_models:
    +materialized: {materialized}
    +schema: null
"""

CORRECT_MODELS = {
    "customer_orders": """with o as (
  select o.customer_id, o.product_id, coalesce(o.quantity, 0) as qty, p.category, p.unit_price
  from {schema}.orders o join {schema}.products p on o.product_id = p.product_id
), cat as (select customer_id, category, sum(qty) as units from o group by 1, 2),
fav as (select customer_id, category from (
  select customer_id, category, row_number() over (partition by customer_id order by units desc, category asc) rn from cat) where rn = 1)
select c.customer_id, c.name, count(o.product_id) as num_orders,
       round(coalesce(sum(o.qty * o.unit_price), 0), 2) as total_spent, f.category as favorite_category
from {schema}.customers c left join o on c.customer_id = o.customer_id left join fav f on f.customer_id = c.customer_id
group by 1, 2, 5""",
    "category_sales": """with s as (
  select p.category, sum(coalesce(o.quantity, 0)) as units_sold,
         round(sum(coalesce(o.quantity, 0) * p.unit_price), 2) as revenue
  from {schema}.products p left join {schema}.orders o on o.product_id = p.product_id group by 1)
select category, units_sold, revenue, case when revenue > (select avg(revenue) from s) then 1 else 0 end as above_average from s""",
}

# Easy columns right, aggregate/tie-break columns wrong (inner join drops Heidi; no NULL handling; ties broken the other way)
WRONG_MODELS = {
    "customer_orders": """select c.customer_id, c.name, count(*) as num_orders,
       sum(o.quantity * p.unit_price) as total_spent, max(p.category) as favorite_category
from {schema}.customers c join {schema}.orders o on c.customer_id = o.customer_id join {schema}.products p on p.product_id = o.product_id
group by 1, 2""",
    "category_sales": """select p.category, sum(o.quantity) as units_sold, sum(o.quantity * p.unit_price) as revenue, 1 as above_average
from {schema}.products p join {schema}.orders o on o.product_id = p.product_id group by 1""",
}


def _dbt_steps(config: dict, models: dict[str, str], materialized: str = "table") -> Iterator[Step]:
    schema = config["duckdb"]["config"]["schema"]
    yield ("bash", {"command": _heredoc("/workspace/elt/dbt/dbt_project.yml", DBT_PROJECT.format(materialized=materialized))})
    for name, sql in models.items():
        yield ("bash", {"command": _heredoc(f"/workspace/elt/dbt/models/{name}.sql", sql.format(schema=schema))})
    yield ("bash", {"command": "cd /workspace/elt/dbt && dbt run --no-use-colors 2>&1 | tail -n 15"})


def _el_steps(config: dict) -> Iterator[Step]:
    yield ("bash", {"command": "cat /workspace/config.yaml; ls /workspace/documentation"})
    yield ("bash", {"command": _heredoc("/workspace/elt/el_config.yaml", _el_config(config))})
    yield ("run_el", {})
    yield ("sql", {"query": f"select count(*) as n from {config['duckdb']['config']['schema']}.orders"})


def oracle(config: dict) -> Iterator[Step]:
    yield from _el_steps(config)
    yield from _dbt_steps(config, CORRECT_MODELS)
    yield ("submit", {})


def no_submit(config: dict) -> Iterator[Step]:
    yield from _el_steps(config)
    yield from _dbt_steps(config, CORRECT_MODELS)


def el_only(config: dict) -> Iterator[Step]:
    yield from _el_steps(config)
    yield ("submit", {})


def wrong_models(config: dict) -> Iterator[Step]:
    yield from _el_steps(config)
    yield from _dbt_steps(config, WRONG_MODELS)
    yield ("submit", {})


def views_not_tables(config: dict) -> Iterator[Step]:
    yield from _el_steps(config)
    yield from _dbt_steps(config, CORRECT_MODELS, materialized="view")
    yield ("submit", {})


def fake_load(config: dict) -> Iterator[Step]:
    """Write raw tables directly (with fake _airbyte_* columns and correct counts),
    skip run_el, then build correct models on top."""
    dest = config["duckdb"]["config"]
    schema = dest["schema"]
    script = f"""
import duckdb, pandas as pd, sqlite3, uuid
db = duckdb.connect({dest['database']!r})
db.execute('create schema if not exists "{schema}"')
def meta(df):
    df = df.copy(); df['_airbyte_raw_id'] = [str(uuid.uuid4()) for _ in range(len(df))]
    df['_airbyte_extracted_at'] = pd.Timestamp.utcnow(); df['_airbyte_meta'] = '{{}}'; df['_airbyte_generation_id'] = 1
    return df
c = meta(pd.read_csv({config['flat_files'][0]['path']!r}))
o = meta(pd.read_json({config['flat_files'][1]['path']!r}, lines=True))
with sqlite3.connect({config['sqlite']['config']['path']!r}) as s:
    p = meta(pd.read_sql_query('select * from products', s))
for name, df in [('customers', c), ('orders', o), ('products', p)]:
    db.register('tmp', df); db.execute(f'create or replace table "{schema}"."{{name}}" as select * from tmp'); db.unregister('tmp')
db.close(); print('faked')
"""
    yield ("bash", {"command": _heredoc("/workspace/fake.py", script) + "\npython /workspace/fake.py"})
    yield from _dbt_steps(config, CORRECT_MODELS)
    yield ("submit", {})


def append_twice(config: dict) -> Iterator[Step]:
    cfg = yaml.safe_load(_el_config(config))
    for s in cfg["sources"]:
        if "sync_mode" in s:
            s["sync_mode"] = "full_refresh_append"
        for st in s.get("streams", []):
            st["sync_mode"] = "full_refresh_append"
    yield ("bash", {"command": _heredoc("/workspace/elt/el_config.yaml", yaml.safe_dump(cfg, sort_keys=False))})
    yield ("run_el", {})
    yield ("run_el", {})
    yield from _dbt_steps(config, CORRECT_MODELS)
    yield ("submit", {})


POLICIES = {
    "oracle": oracle,
    "no_submit": no_submit,
    "el_only": el_only,
    "wrong_models": wrong_models,
    "views_not_tables": views_not_tables,
    "fake_load": fake_load,
    "append_twice": append_twice,
}
