"""This script:
1. Loads data/stock_market.csv with Polars
2. Inspects shape, previews rows, prints a quick schema/null summary
3. Normalizes headers to snake_case, trims whitespace, standardizes text case,
   maps obvious missing tokens ("", "NA", "N/A", "null", "-") to NULL
4. Finds a date-like column, coerces it to yyyy-MM-dd
5. Defines & applies a target schema (dates, strings, ints, floats, bools)
6. Deduplicates rows, writes cleaned.parquet
7. Creates aggregations and writes agg1.parquet, agg2.parquet, agg3.parquet
"""
import re
from datetime import datetime
from dateutil import parser as dateparser
import polars as pl
import pyarrow as pa
import os

INPUT_PATH = "data/stock_market.csv"
CLEANED_PARQUET = "data/cleaned.parquet"
AGG1 = "data/agg1.parquet"
AGG2 = "data/agg2.parquet"
AGG3 = "data/agg3.parquet"

# Standard missing tokens — these will be mapped to None
MISSING_TOKENS = {"", "na", "n/a", "null", "-", "none", "nan"}


def to_snake_case(s: str) -> str:
    # basic snake_case conversion
    s = s.strip()
    s = re.sub(r"[^\w\s]", "_", s)            # punctuation -> underscore
    s = re.sub(r"\s+", "_", s)                # spaces -> underscore
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)  # camelCase -> snake_case
    s = re.sub(r"_+", "_", s)
    s = s.strip("_").lower()
    return s


def normalize_headers(df: pl.DataFrame) -> pl.DataFrame:
    new_cols = [to_snake_case(c) for c in df.columns]
    return df.rename(dict(zip(df.columns, new_cols)))


def map_missing_tokens(df: pl.DataFrame) -> pl.DataFrame:
    # Normalize tokens to lowercase for matching
    missing = {m.lower() for m in MISSING_TOKENS}

    # Identify string columns
    str_cols = [c for c, t in zip(df.columns, df.dtypes) if t == pl.Utf8]

    df = df.with_columns([
        (
            pl.col(c)
            .str.strip_chars()
            .map_elements(
                lambda x: None
                if (x is None or x.lower() in missing)
                else x
            )
            .alias(c)
        )
        for c in str_cols
    ])

    return df


def quick_schema_summary(df: pl.DataFrame):
    print("=== SHAPE ===")
    print(df.shape)
    print("\n=== PREVIEW (first 10 rows) ===")
    print(df.head(10).to_pandas())
    print("\n=== SCHEMA & NULL SUMMARY ===")
    # show dtype and count nulls per column
    summary = []
    for c, t in zip(df.columns, df.dtypes):
        nulls = df.select(pl.col(c).is_null().sum()).to_series()[0]
        unique = df.select(pl.col(c).n_unique()).to_series()[0]
        summary.append((c, str(t), int(nulls), int(unique)))
    print("col_name, dtype, null_count, unique_count")
    for row in summary:
        print(row)


def find_date_column(df: pl.DataFrame):
    # Heuristics: column names commonly used for dates
    candidates = []
    for c in df.columns:
        lname = c.lower()
        if any(x in lname for x in ("date", "dt", "day", "timestamp")):
            candidates.append(c)
    if candidates:
        return candidates[0]  # return first match
    # otherwise try to detect a column that looks like a date by sampling
    for c, t in zip(df.columns, df.dtypes):
        if t == pl.Utf8 or t == pl.Int64 or t == pl.Float64:
            sample = df.select(pl.col(c).cast(pl.Utf8)).head(50).to_series().to_list()
            parsed = 0
            total = 0
            for v in sample:
                total += 1
                try:
                    if v is None:
                        continue
                    dateparser.parse(str(v))
                    parsed += 1
                except Exception:
                    pass
            if total and parsed / total > 0.6:
                return c
    return None


def normalize_date_column(df: pl.DataFrame, date_col: str) -> pl.DataFrame:
    # Attempt to standardize to yyyy-MM-dd
    # First trim spaces and replace common separators
    def normalize_date_str(s):
        if s is None:
            return None
        s = str(s).strip()
        if s.lower() in MISSING_TOKENS:
            return None
        # handle timezone or time portion by parsing via dateutil
        try:
            dt = dateparser.parse(s)
            return dt.date().isoformat()
        except Exception:
            # best-effort: return None
            return None

    # apply conversion to a new column then cast to Date
    df = df.with_columns([
    pl.col(date_col)
        .cast(pl.Utf8)
        .map_elements(normalize_date_str)
        .alias(date_col)
    ])

    # parse string into Date type in Polars
    df = df.with_columns([
    pl.col(date_col)
        .str.strptime(pl.Date, format="%Y-%m-%d", strict=False)
        .alias(date_col)
    ])
    return df


def coerce_target_schema(df: pl.DataFrame) -> pl.DataFrame:
    """
    Heuristically infer target schema and safely cast columns.
    Compatible with Polars ≥ 1.0.
    """
    casts = {}

    for c, t in zip(df.columns, df.dtypes):
        lname = c.lower()

        # date-like
        if "date" in lname or lname in ("dt", "day", "timestamp"):
            casts[c] = pl.Date
            continue

        # text columns
        if any(k in lname for k in ("ticker", "symbol", "company", "name", "sector", "industry")):
            casts[c] = pl.Utf8
            continue

        # boolean-like
        if lname.startswith(("is_", "has_", "flag_")):
            casts[c] = pl.Boolean
            continue

        # integer-like
        if any(k in lname for k in ("volume", "qty", "quantity", "count")):
            casts[c] = pl.Int64
            continue

        # float-like
        if any(k in lname for k in ("price", "open", "close", "high", "low", "adj", "pct", "percent", "change")):
            casts[c] = pl.Float64
            continue

        # default string
        casts[c] = pl.Utf8

    # ===== PER-COLUMN CASTING =====

    for c, target in casts.items():
        try:

            # -----------------------
            # DATE
            # -----------------------
            if target == pl.Date:
                df = df.with_columns(
                    pl.col(c)
                    .cast(pl.Utf8, strict=False)
                    .str.strptime(pl.Date, strict=False)
                    .alias(c)
                )
                continue

            # -----------------------
            # INTEGER
            # -----------------------
            if target == pl.Int64:
                df = df.with_columns(
                    pl.col(c)
                    .cast(pl.Utf8, strict=False)
                    .str.replace_all(r"[^0-9]", "")          # remove all non-digits
                    .str.replace_all(r"^\s*$", None)         # empty → null
                    .cast(pl.Int64, strict=False)
                    .alias(c)
                )
                continue

            # -----------------------
            # FLOAT
            # -----------------------
            if target == pl.Float64:
                df = df.with_columns(
                    pl.col(c)
                    .cast(pl.Utf8, strict=False)
                    .str.replace_all(r"[^0-9.\-]", "")        # keep only valid float chars
                    .str.replace_all(r"^\s*$", None)
                    .cast(pl.Float64, strict=False)
                    .alias(c)
                )
                continue

            # -----------------------
            # BOOLEAN
            # -----------------------
            if target == pl.Boolean:
                df = df.with_columns(
                    pl.col(c)
                    .cast(pl.Utf8, strict=False)
                    .str.to_lowercase()
                    .alias(c)
                )

                df = df.with_columns(
                    pl.when(pl.col(c).is_in(["true", "1", "yes", "y", "t"]))
                      .then(True)
                    .when(pl.col(c).is_in(["false", "0", "no", "n", "f"]))
                      .then(False)
                    .otherwise(None)
                    .alias(c)
                )
                continue

            # -----------------------
            # STRING DEFAULT
            # -----------------------
            df = df.with_columns(pl.col(c).cast(pl.Utf8, strict=False).alias(c))

        except Exception as e:
            print(f"Warning: failed to cast column {c} to {target}. Error: {e}")

    return df

def create_aggregations(df: pl.DataFrame):
    # For the following aggregations we expect columns like: date (Date), ticker (Utf8), close (Float), volume (Int), sector (Utf8)
    # Agg1: daily average close by ticker -> date, ticker, avg_close
    # Agg2: avg volume by sector (over full range) -> sector, avg_volume
    # Agg3: simple daily return by ticker -> date, ticker, close, prev_close, daily_return

    # Ensure names exist; try to find best matches
    cols_lower = [c.lower() for c in df.columns]
    def find_candidate(key_list):
        for key in key_list:
            for c in df.columns:
                if key in c.lower():
                    return c
        return None

    date_col = find_candidate(["date", "dt", "day", "timestamp"])
    ticker_col = find_candidate(["ticker", "symbol", "code"])
    close_col = find_candidate(["close", "adj_close", "adjclose", "last", "price"])
    volume_col = find_candidate(["volume", "vol", "qty", "quantity"])
    sector_col = find_candidate(["sector", "industry"])

    print("Using for aggregations:", dict(date=date_col, ticker=ticker_col, close=close_col, volume=volume_col, sector=sector_col))

    # Agg1
    if date_col and ticker_col and close_col:
        agg1 = (
            df.lazy()
            # ---- CLEAN close_col BEFORE aggregation ----
            .with_columns([
                pl.col(close_col)
                    .cast(pl.Utf8)
                    .str.replace_all(",", "")
                    .str.replace_all(r"\s+", "")
                    .map_elements(lambda x: None if x in ("", None, "-") else x)
                    .cast(pl.Float64, strict=False)
                    .alias(close_col)
            ])
            # --------------------------------------------
            .group_by([pl.col(date_col), pl.col(ticker_col)])
            .agg(pl.col(close_col).mean().alias("avg_close"))
            .sort([pl.col(date_col), pl.col(ticker_col)])
            .collect()
        )
    else:
        agg1 = pl.DataFrame()
        print("Skipping agg1: missing required columns")

    # Agg2
    if sector_col and volume_col:

        df_clean = (
            df
            # --- Remove null ticker rows (if ticker_col exists) ---
            .filter(
                pl.col(ticker_col).is_not_null() 
                if ticker_col in df.columns else True
            )
            # --- Remove null sector rows ---
            .filter(pl.col(sector_col).is_not_null())

            # --- Clean + convert volume column ---
            .with_columns(
                pl.col(volume_col)
                .cast(pl.Utf8, strict=False)            # ensure string for cleaning
                .str.replace_all(",", "")               # remove commas
                .str.replace_all(r"[^0-9]", "")         # keep digits only
                .map_elements(lambda x: None if x in ("", None) else x)
                .cast(pl.Float64, strict=False)         # safe numeric cast
                .alias("volume_float")
            )
            .drop_nulls("volume_float")                 # remove bad/missing values
        )

        agg2 = (
            df_clean.lazy()
            .group_by(pl.col(sector_col))
            .agg(
                pl.col("volume_float").mean().alias("avg_volume")
            )
            .sort("avg_volume", descending=True)
            .collect()
        )

    else:
        agg2 = pl.DataFrame()
        print("Skipping agg2: missing required columns")

    # Agg3: daily return by ticker
    if date_col and ticker_col and close_col:
        agg3 = (
            df.lazy()
            # --- Remove rows with null ticker ---
            .filter(pl.col(ticker_col).is_not_null())

            # --- Ensure close_col is clean numeric before computing returns ---
            .with_columns([
                pl.col(close_col)
                    .cast(pl.Utf8)
                    .str.replace_all(",", "")
                    .str.replace_all(r"\s+", "")
                    .map_elements(lambda x: None if x in ("", "-", None) else x)
                    .cast(pl.Float64, strict=False)
                    .alias("close_val")
            ])

            # --- Compute previous close per ticker ---
            .with_columns([
                pl.col("close_val")
                    .shift(1)
                    .over(pl.col(ticker_col))
                    .alias("prev_close")
            ])

            # --- Daily return: (close / prev_close) - 1 ---
            .with_columns([
                (pl.col("close_val") / pl.col("prev_close") - 1)
                    .alias("daily_return")
            ])

            .select([
                pl.col(date_col),
                pl.col(ticker_col),
                pl.col("close_val").alias("close"),
                pl.col("prev_close"),
                pl.col("daily_return")
            ])

            .sort([pl.col(ticker_col), pl.col(date_col)])
            .collect()
        )
    else:
        agg3 = pl.DataFrame()
        print("Skipping agg3: missing required columns")

    return agg1, agg2, agg3


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input file not found at {INPUT_PATH}. Place the CSV at that path.")

    print("Loading CSV into Polars...")
    # read csv, eager to inspect quickly
    df = pl.read_csv(INPUT_PATH, ignore_errors=True, try_parse_dates=False)

    print("\nInitial load done.")
    quick_schema_summary(df)

    # 1) Normalize headers
    df = normalize_headers(df)
    print("\nHeaders normalized to snake_case.")

    # 2) Trim whitespace for all string columns and map missing tokens
    # convert any bytes to utf8 if necessary
    string_cols = [c for c, t in zip(df.columns, df.dtypes) if t == pl.Utf8]
    df = df.with_columns([
        pl.col(col).str.strip_chars().alias(col)
        for col in string_cols
    ])

    df = map_missing_tokens(df)
    print("\nMapped missing tokens to NULL and trimmed strings.")

    # 3) Find date column and normalize date format to yyyy-MM-dd
    date_col = find_date_column(df)
    if date_col:
        print(f"Detected date column: '{date_col}'. Normalizing...")
        df = normalize_date_column(df, date_col)
    else:
        print("No date-like column detected. Please inspect file manually.")

    # 4) Coerce to target schema heuristically
    df = coerce_target_schema(df)
    print("\nCoerced to target schema where possible.")

    # 5) Deduplicate rows
    original_count = df.height
    df = df.unique()
    deduped_count = df.height
    print(f"Deduplicated rows: {original_count} -> {deduped_count}")

    # 6) Save cleaned.parquet
    print(f"Writing cleaned parquet to {CLEANED_PARQUET} ...")
    df.write_parquet(CLEANED_PARQUET)
    print("Saved cleaned.parquet")

    # 7) Create aggregations
    print("Creating aggregations...")
    agg1, agg2, agg3 = create_aggregations(df)

    if not agg1.is_empty():
        print(f"Writing agg1 -> {AGG1}")
        agg1.write_parquet(AGG1)
    if not agg2.is_empty():
        print(f"Writing agg2 -> {AGG2}")
        agg2.write_parquet(AGG2)
    if not agg3.is_empty():
        print(f"Writing agg3 -> {AGG3}")
        agg3.write_parquet(AGG3)

    print("All done. Files produced:")
    for p in [CLEANED_PARQUET, AGG1, AGG2, AGG3]:
        if os.path.exists(p):
            print(" -", p)

if __name__ == "__main__":
    main()