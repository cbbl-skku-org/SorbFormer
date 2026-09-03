import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import gc
import glob
import torch
import optuna
import pandas as pd
import numpy as np
from src.data import SorbFormerDataset, SorbFormer_collate_fn

import json
import copy
import yaml
from pathlib import Path
from argparse import Namespace
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint

# Import user's model components
from src.model import PlantUptakeModel
from src.molecules.feature_extractor import MolecularFeatureExtractor


# =========================================================
# 1. Ablation Dataset (No SMILES / No Molecular Features)
# Mirrors AblationDataset in optimize_ablation_no_embedding.py
# =========================================================
class AblationDataset(Dataset):
    def __init__(self, X_df, y_array, metadata, split="train", scaler_stats=None):
        self.X_df = X_df.reset_index(drop=True)
        self.y = y_array
        self.metadata = metadata
        self.split = split
        self.scaler_stats = scaler_stats
        self.cat_cols = [c for c in metadata["categorical_columns"] if c in X_df.columns]
        self.num_cols = [c for c in metadata["numerical_columns"] if c in X_df.columns]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        row = self.X_df.iloc[idx]
        cat_feats = {}
        for col in self.cat_cols:
            cat_val = row[col]
            cat_idx = self.metadata["categorical_maps"][col].get(cat_val, 0)
            cat_feats[f"cat_{col}"] = torch.tensor(cat_idx, dtype=torch.long)
        num_feats = {}
        for col in self.num_cols:
            val = float(row[col])
            if self.scaler_stats and col in self.scaler_stats:
                stats = self.scaler_stats[col]
                val = (val - stats["mean"]) / (stats["std"] + 1e-8)
            num_feats[f"num_{col}"] = torch.tensor(val, dtype=torch.float)
        target_val = float(self.y[idx])
        if self.metadata["target_column"] == "RCF":
            target_val = np.log10(target_val)
        return {
            "smiles": "",
            "cat_feats": cat_feats,
            "num_feats": num_feats,
            "target": torch.tensor(target_val, dtype=torch.float),
        }


def ablation_collate_fn(batch):
    smiles = [""] * len(batch)
    targets = torch.stack([item["target"] for item in batch])
    categorical_feats = {}
    for key in batch[0]["cat_feats"]:
        categorical_feats[key] = torch.stack([item["cat_feats"][key] for item in batch])
    numerical_feats = {}
    for key in batch[0]["num_feats"]:
        numerical_feats[key] = torch.stack([item["num_feats"][key] for item in batch])
    return {
        "smiles": smiles,
        "numerical_feats": numerical_feats,
        "categorical_feats": categorical_feats,
        "targets": targets,
        "molecular_feats": None,
    }


# =========================================================
# 2. Helper Functions
# =========================================================

def _extract_metrics(test_results, prefix):
    """Pull the 6 standard metrics from a trainer.test() result dict."""
    return {
        "r2":       test_results[f"{prefix}r2"],
        "mse":      test_results[f"{prefix}mse"],
        "rmse":     test_results[f"{prefix}rmse"],
        "mae":      test_results[f"{prefix}mae"],
        "pearson":  test_results[f"{prefix}pearson"],
        "spearman": test_results[f"{prefix}spearman"],
    }


def _eval_checkpoint(ckpt_path, trial_args, extractor, trainer, test_loader, prefix):
    """Load a checkpoint and return its test metrics (without holding the model in memory)."""
    model = PlantUptakeModel.load_from_checkpoint(
        ckpt_path,
        args=trial_args,
        molecular_extractor=extractor,  # matches PlantUptakeModel.__init__ signature
        strict=False,
    )
    results = trainer.test(model, test_loader, verbose=False)[0]
    metrics = _extract_metrics(results, prefix)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def _find_best_cv_epoch(study_dir, trial_number, num_folds, cv_metric):
    """
    Determine the epoch (0-indexed) that achieved the best average validation
    performance across folds during the HPO phase.

    Reads the per-epoch metrics.csv saved by the HPO CSVLogger for each fold.
    Returns the best epoch number, or None if logs are not found.
    """
    fold_best_epochs = []
    for fold in range(num_folds):
        csv_path = os.path.join(
            study_dir, f"trial_{trial_number}", f"fold_{fold}", "metrics.csv"
        )
        if not os.path.exists(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path)
            # Keep only rows where the validation metric is present
            val_df = df[df[cv_metric].notna()].copy()
            if val_df.empty:
                continue
            best_epoch = int(val_df.loc[val_df[cv_metric].idxmin(), "epoch"])
            fold_best_epochs.append(best_epoch)
        except Exception as e:
            print(f"   Warning: could not read fold metrics from {csv_path}: {e}")

    if not fold_best_epochs:
        return None
    # Round the average to the nearest integer
    return int(round(np.mean(fold_best_epochs)))


# =========================================================
# 3. Model Refitting
# =========================================================
def refit_single_trial(trial, args, train_data, test_data, metadata, save_path, study_dir):
    print(f"\n🚀 Refitting Trial {trial.number}...")

    # ── Load hparams from the HPO fold 0 checkpoint ──────────────────────────
    hparams = {}
    trial_folder = os.path.join(study_dir, f"trial_{trial.number}")
    for fold in range(5):
        h_path = os.path.join(trial_folder, f"fold_{fold}", "hparams.yaml")
        if os.path.exists(h_path):
            with open(h_path, "r") as f:
                hparams = yaml.safe_load(f)
            print(f"   Loaded full hyperparameters from {h_path}")
            break

    # ── Build trial_args: hparams.yaml → trial.params → CLI overrides ────────
    PATH_OVERRIDE_KEYS = {
        "train_csv", "test_csv", "metadata_file",
        "accelerator", "num_workers", "seed", "target", "no_enum",
    }
    trial_args = Namespace(**hparams)
    for k, v in trial.params.items():
        setattr(trial_args, k, v)
    for k in PATH_OVERRIDE_KEYS:
        v = getattr(args, k, None)
        if v is not None:
            setattr(trial_args, k, v)

    # Fix hidden_dim divisibility for cross-attention
    if (
        getattr(trial_args, "fusion_type", None) == "cross_attention"
        and getattr(trial_args, "hidden_dim", None) is not None
    ):
        num_heads = getattr(trial_args, "num_heads", 4)
        if trial_args.hidden_dim % num_heads != 0:
            trial_args.hidden_dim = (trial_args.hidden_dim // num_heads + 1) * num_heads

    seed_everything(args.seed)

    train_X_raw, train_y = train_data
    test_X_raw, test_y = test_data

    # ── Scaler stats (from full train set) ───────────────────────────────────
    scaler_stats = {}
    for col in metadata["numerical_columns"]:
        if col in train_X_raw.columns:
            scaler_stats[col] = {
                "mean": float(train_X_raw[col].mean()),
                "std": float(train_X_raw[col].std()),
            }

    is_ablation = not getattr(trial_args, "use_molecule", True)
    extractor = MolecularFeatureExtractor(trial_args.mol_extractor_type)
    trial_args.mol_dim = extractor.config["feature_dim"]
    if not is_ablation:
        extractor.to("cuda" if args.accelerator == "gpu" else "cpu")
    dl_num_workers = 0 if getattr(trial_args, "finetune_llm", False) else args.num_workers

    # ── Datasets & loaders ───────────────────────────────────────────────────
    if is_ablation:
        refit_train_ds = AblationDataset(
            train_X_raw, train_y, metadata, split="train", scaler_stats=scaler_stats
        )
        test_ds = AblationDataset(
            test_X_raw, test_y, metadata, split="test", scaler_stats=scaler_stats
        )
        collate_fn = ablation_collate_fn
    else:
        refit_train_ds = SorbFormerDataset(
            train_X_raw, train_y, metadata,
            split="train", scaler_stats=scaler_stats,
        )
        test_ds = SorbFormerDataset(
            test_X_raw, test_y, metadata,
            split="test", scaler_stats=scaler_stats,
        )
        collate_fn = SorbFormer_collate_fn

    refit_tr_loader = DataLoader(
        refit_train_ds,
        batch_size=trial_args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=dl_num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=trial_args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=dl_num_workers,
    )

    # ── Checkpointing: save every epoch ──────────────────────────────────────
    ckpt_dir = os.path.join(save_path, "checkpoints")
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="epoch={epoch:03d}",
        auto_insert_metric_name=False, # Avoid epoch=epoch=...
        save_top_k=-1,       # save ALL epochs
        every_n_epochs=1,
        save_last=True,      # also write 'last.ckpt' symlink
        enable_version_counter=False,
    )

    model = PlantUptakeModel(trial_args, extractor)
    logger = CSVLogger(save_dir=save_path, name="refit")

    trainer = Trainer(
        max_epochs=trial_args.max_epochs,
        accelerator=args.accelerator,
        devices=1,
        logger=logger,
        enable_progress_bar=True,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],
        gradient_clip_val=1.0,
        precision="16-mixed" if torch.cuda.is_available() else 32,
    )

    trainer.fit(model, refit_tr_loader)

    # ── Metric prefix depends on target ──────────────────────────────────────
    # For RCF the model logs 'test/revert_*' (original scale); TSCF uses 'test/*'
    prefix = "test/revert_" if args.target == "rcf" else "test/"
    # The CV metric minimised by Optuna (used to pick best CV epoch)
    cv_metric = "val/revert_rmse" if args.target == "rcf" else "val/rmse"

    # ── 1. LAST EPOCH (model currently in memory after fit) ──────────────────
    print("   \U0001f4ca Evaluating LAST epoch...")
    last_results = trainer.test(model, test_loader, verbose=False)[0]
    last_metrics = _extract_metrics(last_results, prefix)
    last_epoch = trial_args.max_epochs - 1
    print(f"      Epoch {last_epoch}: RMSE={last_metrics['rmse']:.4f}  R2={last_metrics['r2']:.4f}")

    # ── 2. BEST TEST EPOCH (scan all saved checkpoints) ──────────────────────
    print("   \U0001f4ca Scanning all checkpoints for best TEST epoch...")
    ckpt_files = sorted(
        glob.glob(os.path.join(ckpt_dir, "epoch=*.ckpt")),
        key=lambda p: int(Path(p).stem.split("=")[-1]),
    )

    best_test_rmse = float("inf")
    best_test_metrics = None
    best_test_epoch = None
    all_epoch_test_metrics = {}   # epoch → metrics dict (for saving)

    for ckpt_path in ckpt_files:
        ep = int(Path(ckpt_path).stem.split("=")[-1])
        ep_metrics = _eval_checkpoint(ckpt_path, trial_args, extractor, trainer, test_loader, prefix)
        all_epoch_test_metrics[ep] = ep_metrics
        if ep_metrics["rmse"] < best_test_rmse:
            best_test_rmse = ep_metrics["rmse"]
            best_test_metrics = ep_metrics
            best_test_epoch = ep

    print(f"      Best test epoch: {best_test_epoch}  RMSE={best_test_rmse:.4f}  "
          f"R2={best_test_metrics['r2']:.4f}")

    # ── 3. CROSS-VAL EPOCH (best avg validation epoch from HPO logs) ─────────
    print("   \U0001f4ca Determining CROSS-VAL best epoch from HPO logs...")
    num_folds = getattr(trial_args, "num_folds", 5)
    cv_best_epoch = _find_best_cv_epoch(
        study_dir, trial.number, num_folds, cv_metric
    )

    cv_epoch_metrics = None
    if cv_best_epoch is not None:
        # Find the closest saved checkpoint
        available_epochs = sorted(all_epoch_test_metrics.keys())
        closest_epoch = min(available_epochs, key=lambda e: abs(e - cv_best_epoch))
        cv_epoch_metrics = all_epoch_test_metrics[closest_epoch]
        print(f"      Cross-val best epoch (avg across folds): {cv_best_epoch} "
              f"→ using checkpoint epoch {closest_epoch}  "
              f"RMSE={cv_epoch_metrics['rmse']:.4f}  R2={cv_epoch_metrics['r2']:.4f}")
    else:
        print("      Could not determine cross-val best epoch (HPO logs not found).")
        cv_best_epoch = None

    # ── Load original CV metrics from HPO trial_summary.json ─────────────────
    original_summary_path = os.path.join(
        study_dir, f"trial_{trial.number}", "trial_summary.json"
    )
    original_cv_metrics = {}
    if os.path.exists(original_summary_path):
        try:
            with open(original_summary_path, "r") as f:
                orig_data = json.load(f)
                original_cv_metrics = orig_data.get("avg_metrics", {})
        except Exception as e:
            print(f"   Warning: Could not read HPO summary at {original_summary_path}: {e}")

    # ── Save consolidated trial_summary.json ─────────────────────────────────
    results = {
        "trial_number":        trial.number,
        "params":              trial.params,
        "cv_performance":      trial.value,        # Optuna objective (val revert_rmse)
        "original_cv_metrics": original_cv_metrics,

        # 1. Last epoch
        "last_epoch": last_epoch,
        "test_metrics_last": last_metrics,

        # 2. Best test epoch
        "best_test_epoch": best_test_epoch,
        "test_metrics_best_test": best_test_metrics,

        # 3. Cross-val best epoch
        "cv_best_epoch": cv_best_epoch,
        "test_metrics_cv_epoch": cv_epoch_metrics,

        # Per-epoch test metrics for all checkpoints
        "all_epoch_test_metrics": {str(k): v for k, v in all_epoch_test_metrics.items()},
    }

    # Keep backward-compatible 'test_metrics' key (= best test epoch)
    results["test_metrics"] = best_test_metrics

    summary_path = os.path.join(save_path, "trial_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=4)

    cv_rmse_str = f"{cv_epoch_metrics['rmse']:.4f}" if cv_epoch_metrics else "n/a"
    print(
        f"\u2705 Trial {trial.number} done. "
        f"Last RMSE={last_metrics['rmse']:.4f}  "
        f"Best-test RMSE={best_test_metrics['rmse']:.4f} (ep {best_test_epoch})  "
        f"CV-epoch RMSE={cv_rmse_str} (ep {cv_best_epoch})"
    )

    # ── Cleanup ───────────────────────────────────────────────────────────────
    del model, trainer, refit_train_ds, test_ds, refit_tr_loader, test_loader, extractor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =========================================================
# 4. Main
# =========================================================
def main():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study_dir", type=str, required=True, help="Folder containing optuna_study.db"
    )
    parser.add_argument(
        "--k", type=int, default=10, help="Number of top trials to refit"
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default="SorbFormer_results_final",
        help="Root folder for final results (fallback if study_dir pattern is not recognized)",
    )

    # Data paths — combined CSV format (target col extracted from metadata)
    parser.add_argument("--train_csv", type=str, required=True, help="Combined train+val CSV")
    parser.add_argument("--test_csv", type=str, required=True, help="Combined test CSV")
    parser.add_argument("--metadata_file", type=str, required=True)
    parser.add_argument("--target", type=str, choices=["rcf", "tscf"], required=True)

    # Runtime/device args (override hparams.yaml values)
    parser.add_argument(
        "--accelerator", type=str, default="gpu" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_enum", action="store_true", default=False,
                        help="Disable SMILES enumeration during training")

    args = parser.parse_args()

    db_path = os.path.join(args.study_dir, "optuna_study.db")
    if not os.path.exists(db_path):
        print(f"❌ Error: {db_path} not found.")
        return

    db_url = f"sqlite:///{os.path.abspath(db_path)}"
    study_summaries = optuna.get_all_study_summaries(storage=db_url)
    if not study_summaries:
        print("❌ Error: No studies found in DB.")
        return

    study_name = study_summaries[0].study_name
    study = optuna.load_study(study_name=study_name, storage=db_url)
    args.opt_direction = (
        "maximize"
        if study.direction == optuna.study.StudyDirection.MAXIMIZE
        else "minimize"
    )

    # Get top K completed trials
    completed_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    if args.opt_direction == "maximize":
        sorted_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)
    else:
        sorted_trials = sorted(completed_trials, key=lambda t: t.value, reverse=False)

    top_k = sorted_trials[: args.k]
    print(f"🔎 Found {len(top_k)} trials to refit in {study_name}")

    # Load Data — always using combined CSV format
    with open(args.metadata_file, "r") as f:
        metadata = json.load(f)
    target_col = metadata["target_column"]

    full_train_df = pd.read_csv(args.train_csv)
    train_y = full_train_df[target_col].values.ravel()
    train_X_df = full_train_df.drop(columns=[target_col])

    full_test_df = pd.read_csv(args.test_csv)
    test_y = full_test_df[target_col].values.ravel()
    test_X_raw = full_test_df.drop(columns=[target_col])

    # Determine relative save path
    # args.study_dir might be results_folder/tscf/biot5/gated
    # We want to create final_results_folder/tscf/biot5/gated

    study_path = Path(args.study_dir).resolve()
    parts = list(study_path.parts)
    
    # Look for the results root folder in the path
    results_root_idx = -1
    for i, part in enumerate(parts):
        if part.startswith("finetune_SorbFormer_results") or part.startswith("ablation_no_embedding"):
            results_root_idx = i
            break
    
    if results_root_idx != -1:
        save_root = f"final_{parts[results_root_idx]}"
        sub_path = os.path.join(*parts[results_root_idx+1:])
        final_save_dir = os.path.join(save_root, sub_path)
    else:
        final_save_dir = args.save_root

    os.makedirs(final_save_dir, exist_ok=True)
    print(f"📁 Saving final refit results to: {final_save_dir}")

    for trial in top_k:
        trial_save_path = os.path.join(final_save_dir, f"trial_{trial.number}")
        os.makedirs(trial_save_path, exist_ok=True)

        # Check if already done
        if os.path.exists(os.path.join(trial_save_path, "trial_summary.json")):
            print(f"⏩ Trial {trial.number} already refitted. Skipping.")
            continue

        refit_single_trial(
            trial,
            args,
            (train_X_df, train_y),
            (test_X_raw, test_y),
            metadata,
            trial_save_path,
            args.study_dir,
        )


if __name__ == "__main__":
    main()
