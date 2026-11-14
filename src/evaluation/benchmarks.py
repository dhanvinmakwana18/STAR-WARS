"""Published NASA C-MAPSS benchmark results and comparison utilities.

Includes literature reference numbers across all four sub-datasets (FD001-FD004)
from Babu et al. (2016), Zheng et al. (2017), and Li et al. (2018).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DATASETS: tuple[str, ...] = ("FD001", "FD002", "FD003", "FD004")

PUBLISHED_BENCHMARKS: dict[str, dict[str, dict[str, float | None]]] = {
    "Babu et al. (2016) CNN": {
        "rmse": {"FD001": 18.45, "FD002": 30.29, "FD003": 19.82, "FD004": 29.16},
        "nasa_score": {"FD001": None, "FD002": None, "FD003": None, "FD004": None},
    },
    "Zheng et al. (2017) Deep LSTM": {
        "rmse": {"FD001": 16.14, "FD002": 24.49, "FD003": 16.18, "FD004": 28.17},
        "nasa_score": {"FD001": 338.0, "FD002": 4450.0, "FD003": 852.0, "FD004": 5550.0},
    },
    "Li et al. (2018) DCNN": {
        "rmse": {"FD001": 12.61, "FD002": 22.36, "FD003": 12.64, "FD004": 23.31},
        "nasa_score": {"FD001": 273.7, "FD002": 10412.0, "FD003": 284.1, "FD004": 12466.0},
    },
}


def benchmarks_to_dataframe() -> pd.DataFrame:
    """Return ``PUBLISHED_BENCHMARKS`` as a tidy DataFrame."""
    rows: list[dict[str, object]] = []
    for model_name, metrics in PUBLISHED_BENCHMARKS.items():
        for ds in DATASETS:
            rows.append(
                {
                    "model": model_name,
                    "dataset_id": ds,
                    "rmse": metrics["rmse"][ds],
                    "nasa_score": metrics["nasa_score"][ds],
                }
            )
    return pd.DataFrame(rows)


PUBLISHED_BENCHMARKS_DF: pd.DataFrame = benchmarks_to_dataframe()


def _resolve_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lower_map = {col.lower(): col for col in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _pct_diff(ours: pd.Series, ref: pd.Series) -> pd.Series:
    """Compute signed percentage difference ``100 * (ours - ref) / ref``.

    Negative values indicate lower error (better) than the benchmark; positive
    values indicate higher error (worse) than the benchmark.
    """
    ours_f = pd.to_numeric(ours, errors="coerce")
    ref_f = pd.to_numeric(ref, errors="coerce")
    return np.where(
        ref_f.notna() & (ref_f != 0.0) & ours_f.notna(),
        100.0 * (ours_f - ref_f) / ref_f,
        np.nan,
    )


def compare_against_benchmarks(our_results: pd.DataFrame) -> pd.DataFrame:
    """Compare model evaluation rows against Zheng (2017) and Li (2018) (plus Babu 2016).

    Parameters
    ----------
    our_results:
        DataFrame with one row per (model, dataset_id) evaluation. Must contain
        a dataset identifier column (``dataset_id`` or ``dataset``) or dataset
        index values in ``FD001``..``FD004``, and an RMSE column (``rmse`` or
        ``rmse_clipped``). Optionally includes a NASA score column
        (``nasa_score``, ``score``, or ``nasa_asymmetric_score``).

    Returns
    -------
    pd.DataFrame
        Copy of ``our_results`` enriched with reference benchmark columns and
        percentage differences (``100 * (ours - ref) / ref``, where negative
        means lower/better error than the published benchmark).
    """
    if not isinstance(our_results, pd.DataFrame):
        raise TypeError("our_results must be a pandas DataFrame.")

    df = our_results.copy()
    if df.empty:
        return df

    ds_col = _resolve_column(df, ("dataset_id", "dataset"))
    if ds_col is not None:
        ds_series = df[ds_col].astype(str).str.upper().str.strip()
    elif df.index.astype(str).str.upper().isin(DATASETS).any():
        ds_series = df.index.astype(str).str.upper().str.strip()
        df["dataset_id"] = ds_series
    else:
        raise ValueError(
            "our_results must contain a 'dataset_id' or 'dataset' column "
            "(or dataset IDs FD001-FD004 in the index)."
        )

    rmse_col = _resolve_column(df, ("rmse", "rmse_clipped"))
    if rmse_col is None:
        raise ValueError("our_results must contain an 'rmse' (or 'rmse_clipped') column.")

    score_col = _resolve_column(
        df, ("nasa_score", "nasa_score_clipped", "score", "nasa_asymmetric_score")
    )

    babu_rmse = PUBLISHED_BENCHMARKS["Babu et al. (2016) CNN"]["rmse"]
    zheng_rmse = PUBLISHED_BENCHMARKS["Zheng et al. (2017) Deep LSTM"]["rmse"]
    zheng_score = PUBLISHED_BENCHMARKS["Zheng et al. (2017) Deep LSTM"]["nasa_score"]
    li_rmse = PUBLISHED_BENCHMARKS["Li et al. (2018) DCNN"]["rmse"]
    li_score = PUBLISHED_BENCHMARKS["Li et al. (2018) DCNN"]["nasa_score"]

    df["babu_2016_rmse"] = ds_series.map(babu_rmse).astype(float)
    df["zheng_2017_rmse"] = ds_series.map(zheng_rmse).astype(float)
    df["li_2018_rmse"] = ds_series.map(li_rmse).astype(float)

    df["rmse_pct_diff_vs_babu_2016"] = _pct_diff(df[rmse_col], df["babu_2016_rmse"])
    df["rmse_pct_diff_vs_zheng_2017"] = _pct_diff(df[rmse_col], df["zheng_2017_rmse"])
    df["rmse_pct_diff_vs_li_2018"] = _pct_diff(df[rmse_col], df["li_2018_rmse"])

    df["zheng_2017_nasa_score"] = ds_series.map(zheng_score).astype(float)
    df["li_2018_nasa_score"] = ds_series.map(li_score).astype(float)

    if score_col is not None:
        df["score_pct_diff_vs_zheng_2017"] = _pct_diff(
            df[score_col], df["zheng_2017_nasa_score"]
        )
        df["score_pct_diff_vs_li_2018"] = _pct_diff(
            df[score_col], df["li_2018_nasa_score"]
        )
    else:
        df["score_pct_diff_vs_zheng_2017"] = np.nan
        df["score_pct_diff_vs_li_2018"] = np.nan

    return df
