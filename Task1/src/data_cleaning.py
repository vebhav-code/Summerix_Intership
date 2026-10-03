"""
data_cleaning.py
================
Reusable helper functions for loading and cleaning the Student Performance
dataset.

Every function takes a DataFrame (or a path) and returns a NEW object, so the
original raw data is never modified in place. The notebook calls these
functions one at a time so each cleaning step can be inspected; the
``clean_dataset`` function chains them together for one-shot use.

Run this file directly to regenerate the cleaned CSV from the raw CSV:

    python src/data_cleaning.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths (relative to the project root, so the project works on any computer)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "student_data.csv"
CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "cleaned_student_data.csv"

# Shorter, friendlier names applied AFTER the generic snake_case conversion.
# (The generic step turns "parental level of education" into
#  "parental_level_of_education"; this map shortens the two longest names.)
COLUMN_RENAME_MAP = {
    "parental_level_of_education": "parental_education",
    "test_preparation_course": "test_preparation",
}

# Valid range of an exam score in this dataset (documented in data/README.md)
SCORE_MIN, SCORE_MAX = 0, 100


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_dataset(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load a CSV file into a DataFrame.

    Parameters
    ----------
    path : str or Path
        Location of the CSV file. Defaults to the raw student dataset.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Check the path or run the "
            "notebook from the 'notebooks/' folder."
        )
    return pd.read_csv(path)


def get_column_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return ``(numerical_columns, categorical_columns)`` for a DataFrame."""
    numerical = df.select_dtypes(include="number").columns.tolist()
    categorical = df.select_dtypes(exclude="number").columns.tolist()
    return numerical, categorical


# ---------------------------------------------------------------------------
# Quality checks (these only REPORT problems; they never change the data)
# ---------------------------------------------------------------------------
def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Return a table with the count and percentage of missing values per column."""
    missing_count = df.isna().sum()
    report = pd.DataFrame(
        {
            "missing_count": missing_count,
            "missing_percent": (missing_count / len(df) * 100).round(2),
        }
    )
    return report.sort_values("missing_count", ascending=False)


def count_duplicates(df: pd.DataFrame) -> int:
    """Return the number of fully duplicated rows (the first copy is not counted)."""
    return int(df.duplicated().sum())


def find_score_range_violations(
    df: pd.DataFrame,
    score_columns: list[str],
    low: int = SCORE_MIN,
    high: int = SCORE_MAX,
) -> pd.DataFrame:
    """Return the rows whose score lies outside the valid ``[low, high]`` range.

    A score outside the valid range is *incorrect data* (not merely an
    unusual value), so it is worth checking before any outlier analysis.
    """
    out_of_range = (df[score_columns] < low) | (df[score_columns] > high)
    return df[out_of_range.any(axis=1)]


def find_whitespace_issues(df: pd.DataFrame) -> dict[str, int]:
    """Count text values with leading/trailing spaces, per text column."""
    _, categorical = get_column_types(df)
    return {
        col: int((df[col].dropna() != df[col].dropna().str.strip()).sum())
        for col in categorical
    }


# ---------------------------------------------------------------------------
# Cleaning steps (each returns a new DataFrame)
# ---------------------------------------------------------------------------
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop fully duplicated rows, keeping the first occurrence."""
    return df.drop_duplicates(keep="first").reset_index(drop=True)


def clean_column_names(
    df: pd.DataFrame, rename_map: dict[str, str] | None = None
) -> pd.DataFrame:
    """Convert column names to lowercase snake_case, then apply ``rename_map``.

    Example: ``"math score"`` -> ``"math_score"``, ``"race/ethnicity"`` ->
    ``"race_ethnicity"``.
    """
    rename_map = COLUMN_RENAME_MAP if rename_map is None else rename_map
    cleaned = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^0-9a-z]+", "_", regex=True)  # spaces, "/" etc. -> "_"
        .str.strip("_")
    )
    result = df.copy()
    result.columns = cleaned
    return result.rename(columns=rename_map)


def standardize_text_values(df: pd.DataFrame) -> pd.DataFrame:
    """Strip surrounding spaces and lowercase every text column.

    This prevents the same category from being counted twice because of
    spelling differences such as ``"Group B"`` vs ``"group b "``.
    """
    result = df.copy()
    _, categorical = get_column_types(result)
    for col in categorical:
        result[col] = result[col].str.strip().str.lower()
    return result


def handle_missing_values(
    df: pd.DataFrame,
    numeric_strategy: str = "median",
    categorical_strategy: str = "mode",
) -> pd.DataFrame:
    """Fill missing values instead of deleting rows.

    * numerical columns   -> median (robust to outliers) or mean
    * categorical columns -> most frequent value (mode)

    Columns without missing values are left untouched, so calling this
    function on a complete dataset returns an identical copy.
    """
    if numeric_strategy not in {"median", "mean"}:
        raise ValueError("numeric_strategy must be 'median' or 'mean'")
    if categorical_strategy != "mode":
        raise ValueError("categorical_strategy must be 'mode'")

    result = df.copy()
    numerical, categorical = get_column_types(result)

    for col in numerical:
        if result[col].isna().any():
            fill_value = (
                result[col].median()
                if numeric_strategy == "median"
                else result[col].mean()
            )
            result[col] = result[col].fillna(fill_value)

    for col in categorical:
        if result[col].isna().any():
            result[col] = result[col].fillna(result[col].mode().iloc[0])

    return result


# ---------------------------------------------------------------------------
# One-shot pipeline
# ---------------------------------------------------------------------------
def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Run the full cleaning pipeline and return ``(clean_df, log)``.

    Order of operations:
        1. clean column names
        2. standardise text values
        3. fill missing values (if any)
        4. remove duplicate rows

    ``log`` is a dictionary that summarises what changed, so the effect of
    cleaning is documented rather than silent.
    """
    log = {"rows_before": len(df), "columns_before": df.shape[1]}
    log["missing_before"] = int(df.isna().sum().sum())
    log["duplicates_before"] = count_duplicates(df)

    cleaned = clean_column_names(df)
    log["renamed_columns"] = {
        old: new for old, new in zip(df.columns, cleaned.columns) if old != new
    }

    standardized = standardize_text_values(cleaned)
    changed_cells = int((standardized.ne(cleaned) & cleaned.notna()).sum().sum())
    log["text_values_changed"] = changed_cells

    filled = handle_missing_values(standardized)
    log["missing_after"] = int(filled.isna().sum().sum())

    final = remove_duplicates(filled)
    log["rows_after"] = len(final)
    log["duplicates_removed"] = log["rows_before"] - log["rows_after"]
    return final, log


def save_dataset(df: pd.DataFrame, path: str | Path = CLEAN_DATA_PATH) -> Path:
    """Save a DataFrame to CSV (without the index) and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    raw = load_dataset()
    clean, cleaning_log = clean_dataset(raw)
    out_path = save_dataset(clean)
    print("Cleaning summary")
    for key, value in cleaning_log.items():
        print(f"  {key}: {value}")
    print(f"Saved cleaned dataset to: {out_path.relative_to(PROJECT_ROOT)}")
