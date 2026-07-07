"""
DuckDB helper functions for the hard drive reliability project.

Key idea: DuckDB queries the raw CSVs directly via a glob pattern.
No separate "load into a database" step is needed — DuckDB scans
files on demand, so this works even when the CSVs are far bigger
than available RAM.
"""

import duckdb
import pandas as pd


def get_connection(db_path: str = None) -> duckdb.DuckDBPyConnection:
    """
    Returns a DuckDB connection.
    db_path=None -> in-memory (data in CSVs, not in DuckDB itself, so persistence is not necessary).
    """
    return duckdb.connect(db_path) if db_path else duckdb.connect()


def raw_glob(data_dir: str) -> str:
    """Build the glob pattern of DuckDB to turn folder path into wildcard pattern, so that DuckDB treat the folder of CSVs as one table."""
    return f"{data_dir.rstrip('/')}/*.csv"




""" What do the data looks like?"""


def preview(con: duckdb.DuckDBPyConnection, data_dir: str, n: int = 5) -> pd.DataFrame:
    """Quick look at the first n rows across all files."""
    query = f"SELECT * FROM read_csv_auto('{raw_glob(data_dir)}') LIMIT {n}"
    return con.sql(query).df()


def schema(con: duckdb.DuckDBPyConnection, data_dir: str) -> pd.DataFrame:
    """Inferred column names and types."""
    query = f"DESCRIBE SELECT * FROM read_csv_auto('{raw_glob(data_dir)}')"
    return con.sql(query).df()




"""How much data over certain period?"""


def row_count(con: duckdb.DuckDBPyConnection, data_dir: str) -> int:
    query = f"SELECT COUNT(*) AS n FROM read_csv_auto('{raw_glob(data_dir)}')"
    return con.sql(query).df()["n"][0]


def date_range(con: duckdb.DuckDBPyConnection, data_dir: str, date_col: str = "date") -> pd.DataFrame:
    query = f"""
        SELECT MIN({date_col}) AS min_date, MAX({date_col}) AS max_date,
               COUNT(DISTINCT {date_col}) AS n_days
        FROM read_csv_auto('{raw_glob(data_dir)}')
    """
    return con.sql(query).df()




"""Summary broken down by group"""


def unique_models(con: duckdb.DuckDBPyConnection, data_dir: str, model_col: str = "model") -> pd.DataFrame:
    query = f"""
        SELECT {model_col} AS model, COUNT(DISTINCT serial_number) AS n_drives, COUNT(*) AS n_rows
        FROM read_csv_auto('{raw_glob(data_dir)}')
        GROUP BY {model_col}
        ORDER BY n_rows DESC
    """
    return con.sql(query).df()


def failure_summary(con: duckdb.DuckDBPyConnection, data_dir: str) -> pd.DataFrame:
    """Failure counts and rate, overall and by model for sanity check."""
    query = f"""
        SELECT
            model,
            COUNT(*) AS drive_days,
            SUM(failure) AS failures,
            ROUND(100.0 * SUM(failure) / COUNT(*), 4) AS failure_rate_pct
        FROM read_csv_auto('{raw_glob(data_dir)}')
        GROUP BY model
        ORDER BY failures DESC
    """
    return con.sql(query).df()




"""Identify usable groups of data"""


def threshold_sensitivity(con: duckdb.DuckDBPyConnection, data_dir: str,
                           thresholds: list = None) -> pd.DataFrame:
    """
    For each candidate minimum-failure threshold Y, report how many models
    are eligible and what fraction of total failures they cover. Used to determine 
    the minimum failure threshold [Y] by looking at the 'elbow' — where raising 
    [Y] further stops meaningfully changing failure coverage.
    """
    if thresholds is None:
        thresholds = [5, 10, 15, 20, 30, 50]

    summary = failure_summary(con, data_dir)
    total_failures = summary["failures"].sum()

    rows = []
    for y in thresholds:
        eligible = summary[summary["failures"] >= y]
        covered = eligible["failures"].sum()
        rows.append({
            "Y": y,
            "n_models_eligible": len(eligible),
            "failures_covered": covered,
            "pct_failures_covered": round(100 * covered / total_failures, 1) if total_failures else 0
        })
    return pd.DataFrame(rows)


def drives_with_enough_failures(con: duckdb.DuckDBPyConnection, data_dir: str, min_failures: int = 5) -> pd.DataFrame:
    """
    Which models have enough failure EVENTS to display anything statistically meaningful information? 
    Models below the threshold are candidates to exclude or merge into an 'other' bucket.
    """
    summary = failure_summary(con, data_dir)
    return summary[summary["failures"] >= min_failures]


def sample_to_parquet(con: duckdb.DuckDBPyConnection, data_dir: str, out_path: str, where: str = "1=1"):
    """
    Pull a filtered/aggregated slice down to a Parquet file small enough for plain pandas to handle.
    `where` is a raw SQL filter, e.g. "failure = 1" or "model = 'ST4000DM000'".
    """
    query = f"""
        COPY (
            SELECT * FROM read_csv_auto('{raw_glob(data_dir)}')
            WHERE {where}
        ) TO '{out_path}' (FORMAT PARQUET)
    """
    con.sql(query)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    # Quick smoke test against the synthetic sample data
    con = get_connection()
    data_dir = "data/raw/sample"

    print("Schema:")
    print(schema(con, data_dir))
    print("\nRow count:", row_count(con, data_dir))
    print("\nDate range:")
    print(date_range(con, data_dir))
    print("\nFailure summary by model:")
    print(failure_summary(con, data_dir))
