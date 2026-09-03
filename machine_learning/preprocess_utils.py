from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CAT_COLS = [
    "Class",
    "Order",
    "Family",
    "Genus",
    "plant_functional_group",
    "plant_root_system",
    "exposure_medium",
    "plant_condition",
]

DROP_COLS = [
    "Chemical",
    "SMILES",
    "Kingdom",
    "Division",
    "Species",
]

DEFAULT_EMBEDDING_PREFIXES = (
    "biot5",
    "chemberta",
)

NO_SCALE_COLS = [
    "plant_monocot",
]


def build_max_features_grid(n_features: int) -> List[int]:
    if n_features <= 1:
        return [1]

    current_range = list(range(2, min(12, n_features + 1), 1))
    expanded_candidates = [
        12,
        16,
        24,
        32,
        48,
        64,
        96,
        128,
        int(round(math.sqrt(n_features))),
    ]

    merged = current_range + expanded_candidates
    merged = [v for v in merged if 2 <= v <= n_features]
    return sorted(set(int(v) for v in merged))


def _existing_cols(df: pd.DataFrame, cols: Iterable[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def _make_onehot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _infer_object_cat_cols(
    df: pd.DataFrame,
    exclude_cols: Optional[Iterable[str]] = None,
) -> List[str]:
    exclude = set(exclude_cols or [])
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c]):
            cols.append(c)
    return cols


def _infer_numeric_cols(
    df: pd.DataFrame,
    cat_cols: List[str],
    exclude_cols: Optional[List[str]] = None,
) -> List[str]:
    exclude_cols = exclude_cols or []
    blocked = set(cat_cols) | set(exclude_cols)

    num_cols = []
    for c in df.columns:
        if c in blocked:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
    return num_cols


def _resolve_embedding_cols(
    df: pd.DataFrame,
    embedding_cols: Optional[List[str]] = None,
    embedding_prefix: Optional[str] = None,
) -> List[str]:
    if embedding_cols is not None:
        cols = [c for c in embedding_cols if c in df.columns]
        if cols:
            return cols

    if embedding_prefix is not None:
        cols = [c for c in df.columns if c.startswith(embedding_prefix)]
        if cols:
            return cols

    cols = [
        c for c in df.columns
        if any(c.startswith(prefix) for prefix in DEFAULT_EMBEDDING_PREFIXES)
    ]
    return cols


def infer_embedding_cols(
    df: pd.DataFrame,
    embedding_cols: Optional[List[str]] = None,
    embedding_prefix: Optional[str] = None,
) -> List[str]:
    return _resolve_embedding_cols(
        df,
        embedding_cols=embedding_cols,
        embedding_prefix=embedding_prefix,
    )


@lru_cache(maxsize=None)
def _read_csv_header_kind(path: str) -> Optional[str]:
    if path is None:
        return None

    path = str(path)
    if not path.lower().endswith(".csv"):
        return None
    if not os.path.exists(path):
        return None

    try:
        cols = pd.read_csv(path, nrows=0).columns
    except Exception:
        return None

    lowered = {str(c).strip().lower() for c in cols}
    has_rcf = "rcf" in lowered
    has_tscf = "tscf" in lowered

    if has_rcf and has_tscf:
        raise ValueError(
            f"CSV header for {path} contains both 'RCF' and 'TSCF' columns."
        )
    if has_rcf:
        return "rcf"
    if has_tscf:
        return "tscf"
    return None


def _infer_target_kind_from_paths(*paths: str) -> str:
    found = set()

    for path in paths:
        kind = _read_csv_header_kind(path)
        if kind is not None:
            found.add(kind)

    if len(found) > 1:
        raise ValueError(
            f"Conflicting target columns inferred from CSV headers: {sorted(found)}"
        )
    if len(found) == 1:
        return next(iter(found))

    raise ValueError(
        "Could not infer target type from CSV headers. "
        "Expected one of the provided CSV files to contain an 'RCF' or 'TSCF' column."
    )


def maybe_log10_target(y, *paths: str):
    if _infer_target_kind_from_paths(*paths) != "rcf":
        return y

    y_series = pd.Series(y, copy=False)
    if (y_series <= 0).any():
        raise ValueError("RCF target contains non-positive values, so log10 cannot be applied.")
    return y_series.astype(float).map(math.log10).to_numpy()


def is_rcf_paths(*paths: str) -> bool:
    return _infer_target_kind_from_paths(*paths) == "rcf"


def restore_target_scale(y, *paths: str):
    if not is_rcf_paths(*paths):
        return y
    return pd.Series(y, copy=False).astype(float).map(lambda v: 10 ** v).to_numpy()


def _safe_pearsonr(y_true, y_pred):
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return pearsonr(y_true, y_pred)[0]


def compute_metric_tuple(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    corr = _safe_pearsonr(y_true, y_pred)
    return mse, rmse, mae, r2, corr


def evaluate_model_pair(y_true, y_pred, *paths):
    current_scores = compute_metric_tuple(y_true, y_pred)
    original_scores = None
    if is_rcf_paths(*paths):
        y_true_original = restore_target_scale(y_true, *paths)
        y_pred_original = restore_target_scale(y_pred, *paths)
        original_scores = compute_metric_tuple(y_true_original, y_pred_original)
    return {
        "current": current_scores,
        "original": original_scores,
    }


def _write_metric_block(f, title: str, prefix: str, scores):
    if scores is None:
        return
    f.write(f"{title}\n")
    f.write(f"{prefix} MSE:  {scores[0]}\n")
    f.write(f"{prefix} RMSE: {scores[1]}\n")
    f.write(f"{prefix} MAE:  {scores[2]}\n")
    f.write(f"{prefix} R2:   {scores[3]}\n")
    f.write(f"{prefix} Corr: {scores[4]}\n\n")


def write_results_txt_dual(
    path,
    model_label: str,
    best_params,
    cv_mean_current,
    cv_std_current,
    train_scores_current,
    test_scores_current,
    cv_mean_original=None,
    cv_std_original=None,
    train_scores_original=None,
    test_scores_original=None,
    search_summary=None,
):
    with open(path, "w") as f:
        f.write(f"{model_label} (CV-mean selection)\n\n")
        f.write(f"Best Hyperparameters: {best_params}\n\n")

        _write_metric_block(f, "CV Mean (current scale, across folds)", "CV Mean", cv_mean_current)
        _write_metric_block(f, "CV Std (current scale, across folds)", "CV Std", cv_std_current)
        _write_metric_block(f, "Train (current scale, refit on full train set)", "Train", train_scores_current)
        _write_metric_block(f, "Test (current scale, evaluated once on held-out test set)", "Test", test_scores_current)

        if cv_mean_original is not None:
            _write_metric_block(f, "CV Mean (original scale, across folds)", "Original CV Mean", cv_mean_original)
        if cv_std_original is not None:
            _write_metric_block(f, "CV Std (original scale, across folds)", "Original CV Std", cv_std_original)
        if train_scores_original is not None:
            _write_metric_block(f, "Train (original scale, refit on full train set)", "Original Train", train_scores_original)
        if test_scores_original is not None:
            _write_metric_block(f, "Test (original scale, evaluated once on held-out test set)", "Original Test", test_scores_original)

        if search_summary is not None:
            f.write("Search summary\n")
            f.write(f"total_combos:   {search_summary['total_combos']}\n")
            f.write(f"skipped_combos: {search_summary['skipped_combos']}\n")
            f.write(f"fold_failures:  {search_summary['fold_failures']}\n")
            f.write(f"logged_errors:  {search_summary['logged_errors']}\n")


def fit_transform_fold(
    X_train_df: pd.DataFrame,
    X_val_df: pd.DataFrame,
    cat_cols: Optional[List[str]] = None,
    embedding_cols: Optional[List[str]] = None,
    embedding_prefix: Optional[str] = None,
    do_scale: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    cat_cols = cat_cols or CAT_COLS

    Xtr = X_train_df.copy()
    Xva = X_val_df.copy()

    drop_cols = _existing_cols(Xtr, DROP_COLS)
    if drop_cols:
        Xtr = Xtr.drop(columns=drop_cols)
        Xva = Xva.drop(columns=drop_cols)

    object_cat_cols = _infer_object_cat_cols(Xtr)
    allowed_cat_cols = _existing_cols(Xtr, cat_cols)
    unexpected_cat_cols = [c for c in object_cat_cols if c not in allowed_cat_cols]
    if unexpected_cat_cols:
        raise ValueError(
            "Unexpected categorical columns found outside the allowed cat_cols list: "
            f"{unexpected_cat_cols}"
        )

    cat_cols_present = allowed_cat_cols
    emb_cols = _resolve_embedding_cols(
        Xtr,
        embedding_cols=embedding_cols,
        embedding_prefix=embedding_prefix,
    )

    fitted = {
        "cat_cols": cat_cols_present,
        "scaler": None,
        "encoder": None,
        "scaled_cols": [],
        "drop_cols": drop_cols,
        "encoded_cols": [],
    }

    if do_scale:
        no_scale_cols = _existing_cols(Xtr, NO_SCALE_COLS)
        scale_cols = _infer_numeric_cols(
            Xtr,
            cat_cols=cat_cols_present,
            exclude_cols=no_scale_cols,
        )

        if scale_cols:
            scaler = StandardScaler()
            Xtr[scale_cols] = scaler.fit_transform(Xtr[scale_cols])
            Xva[scale_cols] = scaler.transform(Xva[scale_cols])

            fitted["scaler"] = scaler
            fitted["scaled_cols"] = scale_cols

    if cat_cols_present:
        encoder = _make_onehot_encoder()
        tr_cat = encoder.fit_transform(Xtr[cat_cols_present].astype(str))
        va_cat = encoder.transform(Xva[cat_cols_present].astype(str))
        encoded_cols = encoder.get_feature_names_out(cat_cols_present).tolist()

        Xtr = Xtr.drop(columns=cat_cols_present)
        Xva = Xva.drop(columns=cat_cols_present)

        Xtr_cat_df = pd.DataFrame(tr_cat, columns=encoded_cols, index=Xtr.index)
        Xva_cat_df = pd.DataFrame(va_cat, columns=encoded_cols, index=Xva.index)

        Xtr = pd.concat([Xtr, Xtr_cat_df], axis=1)
        Xva = pd.concat([Xva, Xva_cat_df], axis=1)

        fitted["encoder"] = encoder
        fitted["encoded_cols"] = encoded_cols

    return Xtr.astype(float), Xva.astype(float), fitted


def fit_transform_train_test(
    X_train_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
    cat_cols: Optional[List[str]] = None,
    embedding_cols: Optional[List[str]] = None,
    embedding_prefix: Optional[str] = None,
    do_scale: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    return fit_transform_fold(
        X_train_df=X_train_df,
        X_val_df=X_test_df,
        cat_cols=cat_cols,
        embedding_cols=embedding_cols,
        embedding_prefix=embedding_prefix,
        do_scale=do_scale,
    )
