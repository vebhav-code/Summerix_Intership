"""
data_analysis.py
================
Reusable functions for descriptive statistics, correlation analysis,
outlier analysis and plotting for the Student Performance dataset.

The plotting functions each:
    * draw one figure with a title and labelled axes,
    * optionally save it as a PNG (``save_path``),
    * show it (``show=True``) or close it (``show=False``).

Run this file directly to regenerate all images in the ``images/`` folder from
the cleaned dataset:

    python src/data_analysis.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = PROJECT_ROOT / "images"

SCORE_COLUMNS = ["math_score", "reading_score", "writing_score"]

# Consistent colour per subject across every chart
SUBJECT_COLORS = {
    "math_score": "#2F6690",
    "reading_score": "#3A7D44",
    "writing_score": "#C8553D",
}

# Natural (low -> high) order of parental education, used to order plots
EDUCATION_ORDER = [
    "some high school",
    "high school",
    "some college",
    "associate's degree",
    "bachelor's degree",
    "master's degree",
]


def set_plot_style() -> None:
    """Apply one clean, readable plotting style to every chart."""
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": 150,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "legend.fontsize": 10,
        }
    )


def nice_name(column: str) -> str:
    """'math_score' -> 'Math Score' (for chart titles and axis labels)."""
    return column.replace("_", " ").title().replace("Race Ethnicity", "Race/Ethnicity")


def _finish(fig: plt.Figure, save_path: str | Path | None, show: bool) -> None:
    """Save the figure if requested, then show or close it."""
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------
def descriptive_statistics(
    df: pd.DataFrame, columns: list[str] | None = None
) -> pd.DataFrame:
    """Return mean, median, mode, std, min, quartiles, max and skewness per column.

    For columns with several equally common values, the *smallest* mode is
    reported (``Series.mode`` returns the modes sorted in ascending order).
    """
    columns = columns or df.select_dtypes(include="number").columns.tolist()
    data = df[columns]
    stats = pd.DataFrame(
        {
            "mean": data.mean(),
            "median": data.median(),
            "mode": [data[c].mode().iloc[0] for c in columns],
            "std": data.std(),
            "min": data.min(),
            "q1": data.quantile(0.25),
            "q3": data.quantile(0.75),
            "max": data.max(),
            "skewness": data.skew(),
        }
    )
    return stats.round(3)


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------
def correlation_matrix(
    df: pd.DataFrame, columns: list[str] | None = None, method: str = "pearson"
) -> pd.DataFrame:
    """Return the correlation matrix of the numerical columns."""
    columns = columns or df.select_dtypes(include="number").columns.tolist()
    return df[columns].corr(method=method)


def ranked_correlations(corr: pd.DataFrame) -> pd.DataFrame:
    """List each unique column pair once, sorted from strongest positive to
    strongest negative correlation."""
    rows = []
    cols = corr.columns.tolist()
    for i, first in enumerate(cols):
        for second in cols[i + 1 :]:
            rows.append({"variable_1": first, "variable_2": second, "correlation": corr.loc[first, second]})
    ranked = pd.DataFrame(rows).sort_values("correlation", ascending=False)
    ranked["correlation"] = ranked["correlation"].round(3)
    return ranked.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Outlier analysis (IQR rule)
# ---------------------------------------------------------------------------
def iqr_bounds(series: pd.Series, k: float = 1.5) -> tuple[float, float]:
    """Return the lower and upper fence of the IQR rule (Q1 - k*IQR, Q3 + k*IQR)."""
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def iqr_outlier_summary(
    df: pd.DataFrame, columns: list[str], k: float = 1.5
) -> pd.DataFrame:
    """Summarise potential outliers (by the IQR rule) for each column."""
    rows = []
    for col in columns:
        lower, upper = iqr_bounds(df[col], k)
        outliers = df.loc[(df[col] < lower) | (df[col] > upper), col]
        rows.append(
            {
                "column": col,
                "lower_fence": round(lower, 3),
                "upper_fence": round(upper, 3),
                "n_outliers": len(outliers),
                "percent_of_rows": round(len(outliers) / len(df) * 100, 2),
                "lowest_value": outliers.min() if len(outliers) else np.nan,
                "highest_value": outliers.max() if len(outliers) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def get_outlier_rows(df: pd.DataFrame, column: str, k: float = 1.5) -> pd.DataFrame:
    """Return the rows that are potential outliers in ``column``."""
    lower, upper = iqr_bounds(df[column], k)
    return df[(df[column] < lower) | (df[column] > upper)]


# ---------------------------------------------------------------------------
# Group comparison
# ---------------------------------------------------------------------------
def group_mean_table(
    df: pd.DataFrame, group_column: str, value_columns: list[str]
) -> pd.DataFrame:
    """Mean of each value column per category, plus the group size."""
    table = df.groupby(group_column)[value_columns].mean().round(2)
    table["n_students"] = df.groupby(group_column).size()
    return table


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_histograms(df, columns=SCORE_COLUMNS, save_path=None, show=True, bins=20):
    """Histogram + density curve for each column, with mean and median lines."""
    fig, axes = plt.subplots(1, len(columns), figsize=(6 * len(columns), 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, col in zip(axes, columns):
        color = SUBJECT_COLORS.get(col, "#2F6690")
        sns.histplot(df[col], bins=bins, kde=True, color=color, edgecolor="white", ax=ax)
        ax.axvline(df[col].mean(), color="black", linestyle="-", linewidth=1.6,
                   label=f"Mean = {df[col].mean():.1f}")
        ax.axvline(df[col].median(), color="black", linestyle="--", linewidth=1.6,
                   label=f"Median = {df[col].median():.1f}")
        ax.set_title(f"Distribution of {nice_name(col)}")
        ax.set_xlabel("Score (0-100)")
        ax.set_ylabel("Number of students")
        ax.legend(loc="upper left")
    _finish(fig, save_path, show)


def plot_countplots(df, columns, save_path=None, show=True, ncols=3):
    """Bar chart of category counts for each categorical column."""
    nrows = int(np.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.3 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, columns):
        order = EDUCATION_ORDER if col == "parental_education" else df[col].value_counts().index
        sns.countplot(data=df, x=col, order=order, color="#5B8DB8", edgecolor="white", ax=ax)
        ax.bar_label(ax.containers[0], padding=2)
        ax.margins(y=0.12)  # head-room so the count labels are not clipped
        ax.set_title(f"Students by {nice_name(col)}")
        ax.set_xlabel(nice_name(col))
        ax.set_ylabel("Number of students")
        if col == "parental_education":
            ax.tick_params(axis="x", rotation=30)
            for label in ax.get_xticklabels():
                label.set_ha("right")
    for ax in axes[len(columns):]:
        ax.set_visible(False)  # hide unused panels
    _finish(fig, save_path, show)


def plot_scatter_pairs(df, pairs, save_path=None, show=True):
    """Scatter plots with a fitted line for each (x, y) column pair.

    The Pearson correlation coefficient r is shown in each panel title.
    """
    fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 5.2))
    axes = np.atleast_1d(axes)
    for ax, (x_col, y_col) in zip(axes, pairs):
        r = df[x_col].corr(df[y_col])
        sns.regplot(data=df, x=x_col, y=y_col, ax=ax,
                    scatter_kws={"alpha": 0.45, "s": 28, "color": SUBJECT_COLORS.get(x_col, "#2F6690")},
                    line_kws={"color": "black", "linewidth": 1.8})
        ax.set_title(f"{nice_name(x_col)} vs {nice_name(y_col)} (r = {r:.2f})")
        ax.set_xlabel(f"{nice_name(x_col)} (0-100)")
        ax.set_ylabel(f"{nice_name(y_col)} (0-100)")
    _finish(fig, save_path, show)


def plot_boxplots(df, columns=SCORE_COLUMNS, save_path=None, show=True):
    """Box plots of several columns; tick labels show the IQR outlier count."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    sns.boxplot(data=df[columns], ax=ax, palette=[SUBJECT_COLORS.get(c, "#2F6690") for c in columns],
                hue=None, flierprops={"marker": "o", "markersize": 5, "markerfacecolor": "white",
                                      "markeredgecolor": "black"})
    counts = iqr_outlier_summary(df, columns).set_index("column")["n_outliers"]
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([f"{nice_name(c)}\n({counts[c]} potential outliers)" for c in columns])
    ax.set_title("Box Plots of Exam Scores (outliers by the 1.5 x IQR rule)")
    ax.set_xlabel("Subject")
    ax.set_ylabel("Score (0-100)")
    _finish(fig, save_path, show)


def plot_correlation_heatmap(corr, save_path=None, show=True):
    """Annotated heatmap of a correlation matrix on the full -1 to +1 scale."""
    fig, ax = plt.subplots(figsize=(7, 5.8))
    labels = [nice_name(c) for c in corr.columns]
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, square=True,
                linewidths=0.8, cbar_kws={"label": "Pearson correlation (r)"},
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title("Correlation Heatmap of Exam Scores")
    _finish(fig, save_path, show)


def plot_correlation_table(corr, save_path=None, show=True):
    """Render the correlation matrix as a shaded table image.

    Shading is proportional to |r| (darker blue = stronger relationship).
    """
    labels = [nice_name(c) for c in corr.columns]
    fig, ax = plt.subplots(figsize=(7.5, 2.3))
    ax.axis("off")
    cell_text = [[f"{v:.3f}" for v in row] for row in corr.values]
    cmap = plt.get_cmap("Blues")
    cell_colors = [[cmap(0.1 + 0.55 * abs(v)) for v in row] for row in corr.values]
    table = ax.table(cellText=cell_text, rowLabels=labels, colLabels=labels,
                     cellColours=cell_colors, cellLoc="center", loc="upper center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.9)
    ax.set_title("Correlation Matrix of Exam Scores (Pearson r)", pad=6)
    _finish(fig, save_path, show)


def plot_group_boxplots(df, category_columns, value_column, save_path=None, show=True, ncols=3):
    """One box plot per categorical column, showing ``value_column`` by category."""
    nrows = int(np.ceil(len(category_columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.8 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, category_columns):
        order = EDUCATION_ORDER if col == "parental_education" else sorted(df[col].unique())
        sns.boxplot(data=df, x=col, y=value_column, order=order, color="#8FB8DE", ax=ax,
                    flierprops={"marker": "o", "markersize": 4, "markerfacecolor": "white",
                                "markeredgecolor": "black"})
        ax.set_title(f"{nice_name(value_column)} by {nice_name(col)}")
        ax.set_xlabel(nice_name(col))
        ax.set_ylabel(f"{nice_name(value_column)} (0-100)")
        if col == "parental_education":
            ax.tick_params(axis="x", rotation=30)
            for label in ax.get_xticklabels():
                label.set_ha("right")
    for ax in axes[len(category_columns):]:
        ax.set_visible(False)
    _finish(fig, save_path, show)


def plot_grouped_means(df, group_column, value_columns=SCORE_COLUMNS, save_path=None, show=True):
    """Grouped bar chart: mean of each score column for every category."""
    means = df.groupby(group_column)[value_columns].mean()
    if group_column == "parental_education":
        means = means.reindex(EDUCATION_ORDER)
    means.columns = [nice_name(c) for c in means.columns]
    colors = [SUBJECT_COLORS[c] for c in value_columns]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    means.plot(kind="bar", ax=ax, color=colors, edgecolor="white", width=0.8)
    ax.set_title(f"Mean Score by {nice_name(group_column)}")
    ax.set_xlabel(nice_name(group_column))
    ax.set_ylabel("Mean score (0-100)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=25 if group_column == "parental_education" else 0)
    ax.grid(axis="x", visible=False)  # horizontal gridlines only
    ax.legend(title="Subject", loc="upper center", ncol=3)  # bars stay below ~80, so the top is free
    _finish(fig, save_path, show)


# ---------------------------------------------------------------------------
# Regenerate all core images from the cleaned dataset
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from data_cleaning import CLEAN_DATA_PATH, get_column_types, load_dataset

    plt.switch_backend("Agg")  # no window needed when run as a script
    set_plot_style()
    data = load_dataset(CLEAN_DATA_PATH)
    _, categorical_cols = get_column_types(data)
    corr_matrix = correlation_matrix(data, SCORE_COLUMNS)

    plot_histograms(data, save_path=IMAGES_DIR / "histogram.png", show=False)
    plot_scatter_pairs(
        data,
        [("math_score", "reading_score"), ("math_score", "writing_score"), ("reading_score", "writing_score")],
        save_path=IMAGES_DIR / "scatter_plot.png",
        show=False,
    )
    plot_boxplots(data, save_path=IMAGES_DIR / "box_plot.png", show=False)
    plot_correlation_heatmap(corr_matrix, save_path=IMAGES_DIR / "correlation_heatmap.png", show=False)
    plot_correlation_table(corr_matrix, save_path=IMAGES_DIR / "correlation_matrix.png", show=False)
    plot_countplots(data, categorical_cols, save_path=IMAGES_DIR / "categorical_counts.png", show=False)
    print(f"Images written to: {IMAGES_DIR.relative_to(PROJECT_ROOT)}")
