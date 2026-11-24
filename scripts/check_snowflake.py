"""Smoke-test the Snowflake destination: connect, write, read, sandbox, clean up.

Usage (from the repo root, with .env filled in):
    uv run python scripts/check_snowflake.py
"""

import pandas as pd

from elt_env.warehouse import SnowflakeDestination


def main() -> None:
    dest = SnowflakeDestination.from_env()
    wh = dest.create_warehouse("smoke_test")
    print(f"connected; using database {wh.database}")
    try:
        cur = wh._exec("SELECT CURRENT_USER(), CURRENT_ROLE(), CURRENT_WAREHOUSE(), CURRENT_VERSION()")
        print("user/role/warehouse/version:", cur.fetchone())

        n = wh.write_table("raw", "cities", pd.DataFrame({"city": ["Urbana", "Chicago"], "pop": [38000, 2700000]}))
        print(f"wrote raw.cities ({n} rows)")
        print("tables:", wh.list_tables())
        print("fetch_table:\n", wh.fetch_table("raw", "cities"))

        df, truncated = wh.query("select city from raw.cities order by pop desc", max_rows=10)
        print("agent query:", df.to_dict("records"), "truncated:", truncated)

        try:
            wh.query("drop table raw.cities", max_rows=10)
            print("ERROR: a DROP statement was allowed")
        except PermissionError:
            print("read-only sandbox: DROP rejected (good)")
    finally:
        wh.drop()
        print(f"dropped {wh.database}")


if __name__ == "__main__":
    main()
