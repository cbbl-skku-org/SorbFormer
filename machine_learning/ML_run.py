#!/usr/bin/env python3
import argparse
import ast
import os
import subprocess
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import timedelta
from multiprocessing import cpu_count
from pathlib import Path

import pandas as pd
from tqdm import tqdm


DEFAULT_MODELS = ["Ridge", "KNN", "MLP", "DT", "RF", "ET", "GB", "AdaBoost", "XGBoost", "LGBM", "CatBoost"]
DEFAULT_DATASETS = ["tscf", "rcf"]
DEFAULT_EMBEDDING_MODES = ["biot5", "chemberta"]
EMBEDDING_PREFIXES = ("biot5", "chemberta")
DROP_ID_COLS = {"Chemical", "SMILES", "Kingdom", "Division", "Species"}
ALLOWED_CAT_COLS = {
    "Class",
    "Order",
    "Family",
    "Genus",
    "plant_functional_group",
    "plant_root_system",
    "exposure_medium",
    "plant_condition",
}

TARGET_CANDIDATES = {
    "rcf": ["RCF", "target", "y"],
    "tscf": ["TSCF", "target", "y"],
}


ENV = os.environ.copy()
ENV.update({
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
})


def parse_args():
    script_dir = Path(__file__).resolve().parent
    dataset_dir = script_dir.parent / "deep_learning" / "data"
    parser = argparse.ArgumentParser(description="Run ML models on train/test csv files.")
    parser.add_argument("--rcf_train_csv", type=str, default=str(dataset_dir / "rcf_train_val.csv"))
    parser.add_argument("--rcf_test_csv", type=str, default=str(dataset_dir / "rcf_test.csv"))
    parser.add_argument("--tscf_train_csv", type=str, default=str(dataset_dir / "tscf_train_val.csv"))
    parser.add_argument("--tscf_test_csv", type=str, default=str(dataset_dir / "tscf_test.csv"))
    parser.add_argument("--script_dir", type=str, default=str(script_dir))
    parser.add_argument("--prepared_root", type=str, default=str(script_dir / "prepared_inputs"))
    parser.add_argument("--results_root", type=str, default=str(script_dir / "results"))
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, choices=DEFAULT_DATASETS)
    parser.add_argument("--embedding_modes", nargs="+", default=DEFAULT_EMBEDDING_MODES, choices=DEFAULT_EMBEDDING_MODES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fold", type=int, default=5)
    parser.add_argument("--max_workers", type=int, default=48)
    parser.add_argument("--load_limit", type=float, default=90.0)
    parser.add_argument("--poll_sec", type=int, default=5)
    parser.add_argument("--skip_if_done", dest="skip_if_done", action="store_true")
    parser.add_argument("--no_skip_if_done", dest="skip_if_done", action="store_false")
    parser.set_defaults(skip_if_done=True)
    return parser.parse_args()


def load1():
    return os.getloadavg()[0]


def wait_until_load_ok(limit, poll_sec):
    while True:
        l1 = load1()
        if l1 <= limit:
            return l1
        time.sleep(poll_sec)


def is_done(results_dir: str) -> bool:
    p = os.path.join(results_dir, "results.txt")
    return os.path.exists(p) and os.path.getsize(p) > 0


def find_target_col(df: pd.DataFrame, dataset: str) -> str:
    for col in TARGET_CANDIDATES[dataset]:
        if col in df.columns:
            return col

    lowered = {c.lower(): c for c in df.columns}
    for col in TARGET_CANDIDATES[dataset]:
        if col.lower() in lowered:
            return lowered[col.lower()]

    raise ValueError(
        f"Could not find target column for dataset={dataset}. "
        f"Available columns: {df.columns.tolist()[:20]}"
    )


def select_feature_columns(x_df: pd.DataFrame, embedding_mode: str) -> pd.DataFrame:
    selected_embedding_cols = [c for c in x_df.columns if c.startswith(embedding_mode)]
    all_embedding_cols = [c for c in x_df.columns if any(c.startswith(prefix) for prefix in EMBEDDING_PREFIXES)]
    common_cols = [c for c in x_df.columns if c not in all_embedding_cols and c not in DROP_ID_COLS]

    missing = [mode for mode in DEFAULT_EMBEDDING_MODES if mode == embedding_mode and not selected_embedding_cols]
    if missing:
        raise ValueError(f"No embedding columns found for embedding_mode={embedding_mode}")

    selected_df = x_df[common_cols + selected_embedding_cols].copy()

    object_cols = selected_df.select_dtypes(include=["object", "category"]).columns.tolist()
    unexpected_cat_cols = [c for c in object_cols if c not in ALLOWED_CAT_COLS]
    if unexpected_cat_cols:
        raise ValueError(
            "Unexpected categorical columns found outside the allowed cat_cols list: "
            f"{unexpected_cat_cols}"
        )

    return selected_df


def prepare_dataset_files(dataset: str, train_csv: str, test_csv: str, prepared_dir: Path, embedding_mode: str):
    prepared_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for split_name, csv_path in [("train_val", train_csv), ("test", test_csv)]:
        df = pd.read_csv(csv_path)
        target_col = find_target_col(df, dataset)

        x_df = df.drop(columns=[target_col])
        y_df = df[[target_col]]

        x_df = select_feature_columns(x_df, embedding_mode=embedding_mode)

        x_out = prepared_dir / f"{dataset}_{split_name}_X_{embedding_mode}.csv"
        y_out = prepared_dir / f"{dataset}_{split_name}_y.csv"

        x_df.to_csv(x_out, index=False)
        y_df.to_csv(y_out, index=False)

        paths[f"{split_name}_x"] = str(x_out)
        paths[f"{split_name}_y"] = str(y_out)

    return paths


def run_job(model, dataset, args, prepared_paths, results_base: Path):
    results_dir = str(results_base / model / dataset)
    os.makedirs(results_dir, exist_ok=True)

    if args.skip_if_done and is_done(results_dir):
        return {"model": model, "dataset": dataset, "returncode": 0, "skipped": True, "results_dir": results_dir}

    script_path = f"{args.script_dir}/{model}_regression.py"
    cmd = [
        "python3", script_path,
        "--train_x_csv", prepared_paths["train_val_x"],
        "--train_y_csv", prepared_paths["train_val_y"],
        "--test_x_csv", prepared_paths["test_x"],
        "--test_y_csv", prepared_paths["test_y"],
        "--seed", str(args.seed),
        "--fold", str(args.fold),
        "--results_dir", results_dir,
    ]

    out_path = os.path.join(results_dir, "stdout.log")
    err_path = os.path.join(results_dir, "stderr.log")

    with open(out_path, "w") as fout, open(err_path, "w") as ferr:
        proc = subprocess.Popen(cmd, stdout=fout, stderr=ferr, text=True, env=ENV)
        rc = proc.wait()

    return {"model": model, "dataset": dataset, "returncode": rc, "skipped": False, "results_dir": results_dir}


def parse_results_txt(results_path: Path):
    metrics = {
        "best_hyperparameters": None,
        "current_cv_mean_mse": None,
        "current_cv_mean_rmse": None,
        "current_cv_mean_mae": None,
        "current_cv_mean_r2": None,
        "current_cv_mean_corr": None,
        "current_cv_std_mse": None,
        "current_cv_std_rmse": None,
        "current_cv_std_mae": None,
        "current_cv_std_r2": None,
        "current_cv_std_corr": None,
        "current_train_mse": None,
        "current_train_rmse": None,
        "current_train_mae": None,
        "current_train_r2": None,
        "current_train_corr": None,
        "current_test_mse": None,
        "current_test_rmse": None,
        "current_test_mae": None,
        "current_test_r2": None,
        "current_test_corr": None,
        "original_cv_mean_mse": None,
        "original_cv_mean_rmse": None,
        "original_cv_mean_mae": None,
        "original_cv_mean_r2": None,
        "original_cv_mean_corr": None,
        "original_cv_std_mse": None,
        "original_cv_std_rmse": None,
        "original_cv_std_mae": None,
        "original_cv_std_r2": None,
        "original_cv_std_corr": None,
        "original_train_mse": None,
        "original_train_rmse": None,
        "original_train_mae": None,
        "original_train_r2": None,
        "original_train_corr": None,
        "original_test_mse": None,
        "original_test_rmse": None,
        "original_test_mae": None,
        "original_test_r2": None,
        "original_test_corr": None,
    }

    with open(results_path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or ":" not in line:
                continue

            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()

            if key == "Best Hyperparameters":
                try:
                    parsed = ast.literal_eval(value)
                    metrics["best_hyperparameters"] = str(parsed)
                except Exception:
                    metrics["best_hyperparameters"] = value
            elif key == "CV Mean MSE":
                metrics["current_cv_mean_mse"] = float(value)
            elif key == "CV Mean RMSE":
                metrics["current_cv_mean_rmse"] = float(value)
            elif key == "CV Mean MAE":
                metrics["current_cv_mean_mae"] = float(value)
            elif key == "CV Mean R2":
                metrics["current_cv_mean_r2"] = float(value)
            elif key == "CV Mean Corr":
                metrics["current_cv_mean_corr"] = float(value)
            elif key == "CV Std MSE":
                metrics["current_cv_std_mse"] = float(value)
            elif key == "CV Std RMSE":
                metrics["current_cv_std_rmse"] = float(value)
            elif key == "CV Std MAE":
                metrics["current_cv_std_mae"] = float(value)
            elif key == "CV Std R2":
                metrics["current_cv_std_r2"] = float(value)
            elif key == "CV Std Corr":
                metrics["current_cv_std_corr"] = float(value)
            elif key == "Train MSE":
                metrics["current_train_mse"] = float(value)
            elif key == "Train RMSE":
                metrics["current_train_rmse"] = float(value)
            elif key == "Train MAE":
                metrics["current_train_mae"] = float(value)
            elif key == "Train R2":
                metrics["current_train_r2"] = float(value)
            elif key == "Train Corr":
                metrics["current_train_corr"] = float(value)
            elif key == "Test MSE":
                metrics["current_test_mse"] = float(value)
            elif key == "Test RMSE":
                metrics["current_test_rmse"] = float(value)
            elif key == "Test MAE":
                metrics["current_test_mae"] = float(value)
            elif key == "Test R2":
                metrics["current_test_r2"] = float(value)
            elif key == "Test Corr":
                metrics["current_test_corr"] = float(value)
            elif key == "Original CV Mean MSE":
                metrics["original_cv_mean_mse"] = float(value)
            elif key == "Original CV Mean RMSE":
                metrics["original_cv_mean_rmse"] = float(value)
            elif key == "Original CV Mean MAE":
                metrics["original_cv_mean_mae"] = float(value)
            elif key == "Original CV Mean R2":
                metrics["original_cv_mean_r2"] = float(value)
            elif key == "Original CV Mean Corr":
                metrics["original_cv_mean_corr"] = float(value)
            elif key == "Original CV Std MSE":
                metrics["original_cv_std_mse"] = float(value)
            elif key == "Original CV Std RMSE":
                metrics["original_cv_std_rmse"] = float(value)
            elif key == "Original CV Std MAE":
                metrics["original_cv_std_mae"] = float(value)
            elif key == "Original CV Std R2":
                metrics["original_cv_std_r2"] = float(value)
            elif key == "Original CV Std Corr":
                metrics["original_cv_std_corr"] = float(value)
            elif key == "Original Train MSE":
                metrics["original_train_mse"] = float(value)
            elif key == "Original Train RMSE":
                metrics["original_train_rmse"] = float(value)
            elif key == "Original Train MAE":
                metrics["original_train_mae"] = float(value)
            elif key == "Original Train R2":
                metrics["original_train_r2"] = float(value)
            elif key == "Original Train Corr":
                metrics["original_train_corr"] = float(value)
            elif key == "Original Test MSE":
                metrics["original_test_mse"] = float(value)
            elif key == "Original Test RMSE":
                metrics["original_test_rmse"] = float(value)
            elif key == "Original Test MAE":
                metrics["original_test_mae"] = float(value)
            elif key == "Original Test R2":
                metrics["original_test_r2"] = float(value)
            elif key == "Original Test Corr":
                metrics["original_test_corr"] = float(value)

    return metrics


def build_summary_df(results_base: Path, models, datasets):
    rows = []

    for model in models:
        for dataset in datasets:
            results_path = results_base / model / dataset / "results.txt"
            if not results_path.exists():
                continue

            parsed = parse_results_txt(results_path)
            rows.append({
                "dataset": dataset.upper(),
                "model": model,
                "Current CV Mean MSE": parsed["current_cv_mean_mse"],
                "Current CV Mean RMSE": parsed["current_cv_mean_rmse"],
                "Current CV Mean MAE": parsed["current_cv_mean_mae"],
                "Current CV Mean R2": parsed["current_cv_mean_r2"],
                "Current CV Mean Corr": parsed["current_cv_mean_corr"],
                "Current CV Std MSE": parsed["current_cv_std_mse"],
                "Current CV Std RMSE": parsed["current_cv_std_rmse"],
                "Current CV Std MAE": parsed["current_cv_std_mae"],
                "Current CV Std R2": parsed["current_cv_std_r2"],
                "Current CV Std Corr": parsed["current_cv_std_corr"],
                "Current Train MSE": parsed["current_train_mse"],
                "Current Train RMSE": parsed["current_train_rmse"],
                "Current Train MAE": parsed["current_train_mae"],
                "Current Train R2": parsed["current_train_r2"],
                "Current Train Corr": parsed["current_train_corr"],
                "Current Test MSE": parsed["current_test_mse"],
                "Current Test RMSE": parsed["current_test_rmse"],
                "Current Test MAE": parsed["current_test_mae"],
                "Current Test R2": parsed["current_test_r2"],
                "Current Test Corr": parsed["current_test_corr"],
                "Original CV Mean MSE": parsed["original_cv_mean_mse"],
                "Original CV Mean RMSE": parsed["original_cv_mean_rmse"],
                "Original CV Mean MAE": parsed["original_cv_mean_mae"],
                "Original CV Mean R2": parsed["original_cv_mean_r2"],
                "Original CV Mean Corr": parsed["original_cv_mean_corr"],
                "Original CV Std MSE": parsed["original_cv_std_mse"],
                "Original CV Std RMSE": parsed["original_cv_std_rmse"],
                "Original CV Std MAE": parsed["original_cv_std_mae"],
                "Original CV Std R2": parsed["original_cv_std_r2"],
                "Original CV Std Corr": parsed["original_cv_std_corr"],
                "Original Train MSE": parsed["original_train_mse"],
                "Original Train RMSE": parsed["original_train_rmse"],
                "Original Train MAE": parsed["original_train_mae"],
                "Original Train R2": parsed["original_train_r2"],
                "Original Train Corr": parsed["original_train_corr"],
                "Original Test MSE": parsed["original_test_mse"],
                "Original Test RMSE": parsed["original_test_rmse"],
                "Original Test MAE": parsed["original_test_mae"],
                "Original Test R2": parsed["original_test_r2"],
                "Original Test Corr": parsed["original_test_corr"],
                "best hyperparameters": parsed["best_hyperparameters"],
            })

    if not rows:
        return pd.DataFrame(columns=[
            "dataset",
            "model",
            "Current CV Mean MSE",
            "Current CV Mean RMSE",
            "Current CV Mean MAE",
            "Current CV Mean R2",
            "Current CV Mean Corr",
            "Current CV Std MSE",
            "Current CV Std RMSE",
            "Current CV Std MAE",
            "Current CV Std R2",
            "Current CV Std Corr",
            "Current Train MSE",
            "Current Train RMSE",
            "Current Train MAE",
            "Current Train R2",
            "Current Train Corr",
            "Current Test MSE",
            "Current Test RMSE",
            "Current Test MAE",
            "Current Test R2",
            "Current Test Corr",
            "Original CV Mean MSE",
            "Original CV Mean RMSE",
            "Original CV Mean MAE",
            "Original CV Mean R2",
            "Original CV Mean Corr",
            "Original CV Std MSE",
            "Original CV Std RMSE",
            "Original CV Std MAE",
            "Original CV Std R2",
            "Original CV Std Corr",
            "Original Train MSE",
            "Original Train RMSE",
            "Original Train MAE",
            "Original Train R2",
            "Original Train Corr",
            "Original Test MSE",
            "Original Test RMSE",
            "Original Test MAE",
            "Original Test R2",
            "Original Test Corr",
            "best hyperparameters",
        ])

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values(["dataset", "model"]).reset_index(drop=True)
    return summary_df


def build_prob_tables(results_base: Path, models, datasets):
    tables = []

    for dataset in datasets:
        for model in models:
            prob_path = results_base / model / dataset / "prob_and_gt.csv"
            if not prob_path.exists():
                continue

            df = pd.read_csv(prob_path)
            df.insert(0, "dataset", dataset.upper())
            df.insert(1, "model", model)
            sheet_name = f"{dataset.upper()}_{model}"
            tables.append((sheet_name[:31], df))

    return tables


def write_summary_excel(summary_df: pd.DataFrame, prob_tables, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path) as writer:
        summary_df.to_excel(writer, index=False, sheet_name="summary")
        for sheet_name, df in prob_tables:
            df.to_excel(writer, index=False, sheet_name=sheet_name)


def run_embedding_mode(args, embedding_mode: str):
    results_base = Path(args.results_root) / f"results_{embedding_mode}"
    summary_xlsx = results_base / "ML_results_summary.xlsx"
    prepared_root = Path(args.prepared_root) / embedding_mode

    print(f"\n=== Running embedding mode: {embedding_mode} ===")
    print(f"Results folder: {results_base}")

    prepared_map = {
        "rcf": prepare_dataset_files("rcf", args.rcf_train_csv, args.rcf_test_csv, prepared_root / "rcf", embedding_mode),
        "tscf": prepare_dataset_files("tscf", args.tscf_train_csv, args.tscf_test_csv, prepared_root / "tscf", embedding_mode),
    }

    job_list = [(m, d) for m in args.models for d in args.datasets]
    if args.skip_if_done:
        job_list = [jd for jd in job_list if not is_done(str(results_base / jd[0] / jd[1]))]

    total = len(job_list)
    if total == 0:
        print("Nothing to run. (All jobs already have results.txt)")
        summary_df = build_summary_df(results_base, args.models, args.datasets)
        prob_tables = build_prob_tables(results_base, args.models, args.datasets)
        write_summary_excel(summary_df, prob_tables, summary_xlsx)
        print(f"Summary saved to: {summary_xlsx}")
        return

    start_time = time.time()
    pbar = tqdm(total=total, desc=f"{embedding_mode}")

    pending = set()
    idx = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        while idx < total and len(pending) < args.max_workers:
            wait_until_load_ok(args.load_limit, args.poll_sec)
            model, dataset = job_list[idx]
            pending.add(executor.submit(run_job, model, dataset, args, prepared_map[dataset], results_base))
            idx += 1

        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)

            for fut in done:
                res = fut.result()
                pbar.update(1)
                if res["returncode"] != 0:
                    print(f"\nERROR: {res['model']} / {res['dataset']}")
                    print(f"logs: {res['results_dir']}/stderr.log")

            while idx < total and len(pending) < args.max_workers:
                wait_until_load_ok(args.load_limit, args.poll_sec)
                model, dataset = job_list[idx]
                pending.add(executor.submit(run_job, model, dataset, args, prepared_map[dataset], results_base))
                idx += 1

    pbar.close()
    print(f"{embedding_mode} total time:", timedelta(seconds=time.time() - start_time))

    summary_df = build_summary_df(results_base, args.models, args.datasets)
    prob_tables = build_prob_tables(results_base, args.models, args.datasets)
    write_summary_excel(summary_df, prob_tables, summary_xlsx)
    print(f"Summary saved to: {summary_xlsx}")


def main():
    args = parse_args()
    print(
        f"cpu_count={cpu_count()} | max_workers={args.max_workers} | "
        f"load_limit={args.load_limit} | skip_if_done={args.skip_if_done}"
    )
    print(f"Chemical/SMILES are excluded from model features.")

    for embedding_mode in args.embedding_modes:
        run_embedding_mode(args, embedding_mode)


if __name__ == "__main__":
    main()
