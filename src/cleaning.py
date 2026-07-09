"""
Cleaning utilities — Week 2.

Everything here works via DuckDB SQL rather than pandas, so it scales to
the full ~28M-row dataset without ever loading raw drive-days into memory.
The only thing that comes back as a pandas DataFrame is the small, final
per-drive snapshot table (one row per drive, not per drive-day).
"""

import duckdb
import pandas as pd
import load_data as ld


def null_fraction_by_column(con: duckdb.DuckDBPyConnection, data_dir: str, columns: list = None) -> pd.DataFrame:
    """
    Fraction of NULLs per column, computed entirely in SQL (no full load into pandas).
    If `columns` is None, checks every smart_* column found in the schema.
    Used to remove mostly empty columns.
    """
    if columns is None:
        schema = ld.schema(con, data_dir)
        columns = [c for c in schema["column_name"] if c.startswith("smart_")]

    exprs = ", ".join(
        f"ROUND(AVG(CASE WHEN {c} IS NULL THEN 1.0 ELSE 0.0 END), 4) AS {c}" for c in columns  # percent null
    )
    query = f"SELECT {exprs} FROM read_csv_auto('{ld.raw_glob(data_dir)}')"
    wide = con.sql(query).df()

    # reshape from wide (one row, many columns) to a long (one column, many rows) tidy column/null_fraction table,
    # so that it is easy to read and sort
    long = wide.T.reset_index()
    long.columns = ["column", "null_fraction"]
    return long.sort_values("null_fraction", ascending=False).reset_index(drop=True)


def sparse_columns(con: duckdb.DuckDBPyConnection, data_dir: str, threshold: float = 0.5,
                    columns: list = None) -> list:
    """Return column names whose null fraction exceeds the given `threshold` as candidates to drop."""
    nf = null_fraction_by_column(con, data_dir, columns)
    return nf.loc[nf["null_fraction"] > threshold, "column"].tolist()


def raw_normalized_pairs(columns: list) -> list:
    """
    Find smart_N_raw / smart_N_normalized pairs present in a column list (DUPLICATES).
    Returns a list of (raw_col, normalized_col) tuples for pairs where BOTH exist.
    """
    raws = {c for c in columns if c.endswith("_raw")}
    norms = {c for c in columns if c.endswith("_normalized")}
    pairs = []
    for r in raws:
        base = r[: -len("_raw")]
        n = f"{base}_normalized"
        if n in norms:
            pairs.append((r, n))
    return sorted(pairs)


def build_snapshot_dataset(con: duckdb.DuckDBPyConnection, data_dir: str,
                            eligible_models: list, feature_columns: list,
                            out_path: str = None) -> pd.DataFrame:
    """
    Build the target-variable decision as a table:
    one row per drive, restricted to eligible_models, with:
      - label = 1 if the drive ever failed in the quarter, else 0
      - feature snapshot = the reading on the day BEFORE failure (failed drives)
                           or the last day in the dataset (healthy drives)

    This process collapse 90 rows-per-drive into 1.

    Done with SQL window functions so it works directly against the full
    CSV folder without ever materializing all drive-days in pandas.
    """
    model_list = ", ".join(f"'{m}'" for m in eligible_models)
    feat_cols = ", ".join(feature_columns)

    query = f"""
    WITH raw AS (
        SELECT serial_number, model, date, failure, {feat_cols}
        FROM read_csv_auto('{ld.raw_glob(data_dir)}')
        WHERE model IN ({model_list})
    ),
    failure_flag AS (
        SELECT serial_number, MAX(failure) AS ever_failed
        FROM raw
        GROUP BY serial_number
    ), 
    failure_date AS (
        SELECT serial_number, MIN(date) AS failure_date
        FROM raw
        WHERE failure = 1
        GROUP BY serial_number
    ),
    ranked_healthy AS (
        SELECT r.*, ROW_NUMBER() OVER (PARTITION BY r.serial_number ORDER BY r.date DESC) AS rn
        FROM raw r
        JOIN failure_flag f ON r.serial_number = f.serial_number
        WHERE f.ever_failed = 0
    ),
    ranked_failed AS (
        SELECT r.*, ROW_NUMBER() OVER (PARTITION BY r.serial_number ORDER BY r.date DESC) AS rn
        FROM raw r
        JOIN failure_date fd ON r.serial_number = fd.serial_number
        WHERE r.date < fd.failure_date
    )
    SELECT serial_number, model, date AS snapshot_date, {feat_cols}, 0 AS label
    FROM ranked_healthy WHERE rn = 1
    UNION ALL
    SELECT serial_number, model, date AS snapshot_date, {feat_cols}, 1 AS label
    FROM ranked_failed WHERE rn = 1
    """
    result = con.sql(query).df()

    if out_path:
        con.sql(f"COPY ({query}) TO '{out_path}' (FORMAT PARQUET)")
        print(f"Wrote {out_path}")

    return result


def class_balance_summary(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """Display class imbalances on the snapshot dataset."""
    counts = df[label_col].value_counts().rename_axis("label").reset_index(name="n")
    counts["pct"] = round(100 * counts["n"] / counts["n"].sum(), 2)
    return counts
