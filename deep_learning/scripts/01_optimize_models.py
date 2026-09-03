import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import gc
import torch
import optuna
import pandas as pd
import numpy as np
from src.data import SorbFormerDataset, SorbFormer_collate_fn

import json
import time
import copy
from argparse import Namespace
from functools import partial
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback

from src.model import PlantUptakeModel
from src.molecules.feature_extractor import MolecularFeatureExtractor


# =========================================================
# 1. Utilities
# =========================================================
def make_stratified_bins(y_list, n_bins=10, strategy="uniform"):
    y = np.array(y_list).reshape(-1, 1)
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy=strategy)
    y_binned = discretizer.fit_transform(y).astype(int).ravel()
    return y_binned


# =========================================================
# 2. Optuna Objective
# =========================================================
def objective(trial, base_args):

    args = Namespace(**vars(base_args))

    if args.loss_type == "huber":
        args.huber_delta = trial.suggest_float("huber_delta", 0.25, 1.5, step=0.25)

    args.num_heads = 4
    args.gate_l1_lambda = 0.0
    if args.fusion_type == "cross_attention":
        args.num_heads = trial.suggest_categorical("num_heads", [4, 8, 16])
    elif args.fusion_type == "gated":
        args.gate_l1_lambda = trial.suggest_categorical(
            "gate_l1_lambda", [0.0, 0.01, 0.1]
        )

    if not base_args.mol_extractor_type:
        args.mol_extractor_type = trial.suggest_categorical(
            "mol_extractor_type", ["biot5-plus-base", "chemberta-77m-mlm", "molt5-base"]
        )
    else:
        args.mol_extractor_type = base_args.mol_extractor_type

    if not base_args.fusion_type:
        args.fusion_type = trial.suggest_categorical(
            "fusion_type", ["gated", "cross_attention", "bilinear", "concat"]
        )
    else:
        args.fusion_type = base_args.fusion_type

    if args.target == "rcf":
        args.categorical_dim = trial.suggest_categorical(
             "categorical_dim", [16, 32, 64]
        )
        args.hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 384, 512])
        args.dropout = trial.suggest_float("dropout", 0.0, 0.2, step=0.05)
        args.lr = trial.suggest_float("lr", 5e-5, 5e-4, log=True)
        args.max_epochs = trial.suggest_categorical("max_epochs", [30, 50, 75]) if getattr(args, "finetune_llm", False) else trial.suggest_categorical("max_epochs", [100, 150])
        args.batch_size = trial.suggest_categorical("batch_size", [2, 4, 8]) if getattr(args, "finetune_llm", False) else trial.suggest_categorical("batch_size", [4, 8, 16])
        args.num_layers = trial.suggest_int("num_layers", 2, 4)
    else:
        args.categorical_dim = trial.suggest_categorical(
            "categorical_dim", [8, 16, 32]
        )
        args.hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 384, 512])
        args.dropout = trial.suggest_float("dropout", 0.0, 0.2, step=0.05)
        args.lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
        args.max_epochs = trial.suggest_categorical("max_epochs", [30, 50, 75]) if getattr(args, "finetune_llm", False) else trial.suggest_categorical("max_epochs", [50, 100, 150])
        args.batch_size = trial.suggest_categorical("batch_size", [2, 4, 8]) if getattr(args, "finetune_llm", False) else trial.suggest_categorical("batch_size", [8, 16, 32])
        args.num_layers = trial.suggest_int("num_layers", 2, 4)

    args.weight_decay = trial.suggest_float("weight_decay", 1e-2, 1e-1, log=True)
    args.mixup_alpha = trial.suggest_float("mixup_alpha", 0.0, 0.2, step=0.1)

    if getattr(args, "finetune_llm", False):
        args.llm_lr = trial.suggest_float("llm_lr", 5e-6, 1e-4, log=True)
        if not getattr(args, "unfreeze_all_llm", False):
            args.num_layers_to_unfreeze = trial.suggest_int("num_layers_to_unfreeze", 1, 4)

    if args.fusion_type == "cross_attention" and args.hidden_dim % args.num_heads != 0:
        args.hidden_dim = (args.hidden_dim // args.num_heads + 1) * args.num_heads

    seed_everything(args.seed + trial.number)

    with open(args.metadata_file, "r") as f:
        metadata = json.load(f)

    # Load Data
    full_train_df = pd.read_csv(args.train_csv)
    train_y = full_train_df[metadata["target_column"]].values.ravel()
    train_X_df = full_train_df.drop(columns=[metadata["target_column"]])

    # Setup Extractor and Enumerator
    extractor = MolecularFeatureExtractor(args.mol_extractor_type)
    args.mol_dim = extractor.config["feature_dim"]

    # 5-Fold Cross-Validation
    y_binned = make_stratified_bins(train_y, n_bins=10)
    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    
    full_test_df = pd.read_csv(args.test_csv)
    test_y = full_test_df[metadata["target_column"]].values.ravel()
    test_X_raw = full_test_df.drop(columns=[metadata["target_column"]])

    fold_metrics_scores = []
    all_test_metrics = []

    # When fine-tuning, the LLM tokenizer cannot be shared across forked workers
    dl_num_workers = 0 if getattr(args, "finetune_llm", False) else args.num_workers

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(train_X_df, y_binned)):
        X_tr_raw = train_X_df.iloc[tr_idx]
        X_va_raw = train_X_df.iloc[va_idx]
        y_tr = train_y[tr_idx]
        y_va = train_y[va_idx]

        scaler_stats = {}
        for col in metadata["numerical_columns"]:
            if col in X_tr_raw.columns:
                scaler_stats[col] = {
                    "mean": float(X_tr_raw[col].mean()),
                    "std": float(X_tr_raw[col].std()),
                }

        # Datasets
        train_ds = SorbFormerDataset(
            X_tr_raw,
            y_tr,
            metadata,
            split="train",
            scaler_stats=scaler_stats,
        )
        val_ds = SorbFormerDataset(
            X_va_raw,
            y_va,
            metadata,
            split="val",
            scaler_stats=scaler_stats,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=SorbFormer_collate_fn,
            num_workers=dl_num_workers,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=SorbFormer_collate_fn,
            num_workers=dl_num_workers,
        )

        fold_extractor = copy.deepcopy(extractor)
        fold_extractor.to("cuda" if args.accelerator == "gpu" else "cpu")

        model = PlantUptakeModel(args, fold_extractor)

        logger = CSVLogger(
            save_dir=args.save_folder,
            name=f"trial_{trial.number}",
            version=f"fold_{fold_idx}",
        )
        monitoring_metric = "val/revert_r2" if args.target == "rcf" else "val/r2"

        pruning_callback = PyTorchLightningPruningCallback(trial, monitor=monitoring_metric)

        trainer = Trainer(
            max_epochs=args.max_epochs,
            accelerator=args.accelerator,
            devices=1,
            logger=logger,
            callbacks=[pruning_callback],
            enable_progress_bar=False,
            enable_checkpointing=False,
            gradient_clip_val=1.0,
            precision="16-mixed" if torch.cuda.is_available() else 32,
        )

        try:
            trainer.fit(model, train_loader, val_loader)

            metrics_log = os.path.join(
                args.save_folder, f"trial_{trial.number}", f"fold_{fold_idx}", "metrics.csv"
            )
            if os.path.exists(metrics_log):
                m_df = pd.read_csv(metrics_log)
                monitoring_metric = "val/revert_r2" if args.target == "rcf" else "val/r2"


                if monitoring_metric in m_df.columns:
                    m_df_val = m_df.dropna(subset=[monitoring_metric])
                    if not m_df_val.empty:
                        best_row = m_df_val.loc[m_df_val[monitoring_metric].idxmax()]
                        fold_scores_dict = {
                            k: v
                            for k, v in best_row.to_dict().items()
                            if isinstance(v, (int, float)) and not np.isnan(v) and (k.startswith("val/") or k.startswith("train/"))
                        }
                        fold_metrics_scores.append(best_row[monitoring_metric])

                        if "all_metrics" not in locals():
                            all_metrics = []
                        all_metrics.append(fold_scores_dict)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                print(f"OOM triggered in trial {trial.number}. Pruning trial to free memory.")
                del model
                del trainer
                del train_ds
                del val_ds
                del train_loader
                del val_loader
                del fold_extractor
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise optuna.exceptions.TrialPruned()
            else:
                raise e

        # Cleanup
        del (
            model,
            trainer,
            train_ds,
            val_ds,
            train_loader,
            val_loader,
            fold_extractor,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not fold_metrics_scores:
        return -1.0

    # Average across folds for the summary
    trial_summary = {
        "trial_number": trial.number,
        "params": trial.params,
        "mean_val_score": float(np.mean(fold_metrics_scores)),
        "fold_metrics": all_metrics if "all_metrics" in locals() else [],
    }

    if "all_metrics" in locals() and all_metrics:
        summary_avg = {}
        all_keys = set().union(*(m.keys() for m in all_metrics))
        for k in all_keys:
            vals = [m[k] for m in all_metrics if k in m and not pd.isna(m[k])]
            if vals:
                summary_avg[k] = float(np.mean(vals))
        trial_summary["avg_metrics"] = summary_avg

    # Aggregate Test Metrics (Averaged from Folds for additional info)
    if all_test_metrics:
        test_avg = {}
        all_test_keys = set().union(*(m.keys() for m in all_test_metrics))
        for k in all_test_keys:
            vals = [m[k] for m in all_test_metrics if k in m and not pd.isna(m[k])]
            if vals:
                test_avg[k] = float(np.mean(vals))
        trial_summary["avg_fold_test_metrics"] = test_avg

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    trial_summary_path = os.path.join(
        args.save_folder, f"trial_{trial.number}", "trial_summary.json"
    )
    os.makedirs(os.path.dirname(trial_summary_path), exist_ok=True)
    with open(trial_summary_path, "w") as f:
        json.dump(trial_summary, f, indent=4)

    return trial_summary["mean_val_score"]


# =========================================================
# 3. Final Evaluation (Refit on full train & Test)
# =========================================================

def evaluate_final_model(best_params, best_cv_score, base_args, trial_number=None):
    print("\n" + "=" * 50)
    if trial_number is not None:
        print(f"🏆 Refitting Trial {trial_number}")
    else:
        print("🏆 Running Final Evaluation with Best Hyperparameters")
    print("=" * 50)

    args = Namespace(**vars(base_args))
    for k, v in best_params.items():
        setattr(args, k, v)

    if args.fusion_type == "cross_attention" and args.hidden_dim % args.num_heads != 0:
        args.hidden_dim = (args.hidden_dim // args.num_heads + 1) * args.num_heads

    seed_everything(args.seed + (trial_number if trial_number is not None else 0))

    with open(args.metadata_file, "r") as f:
        metadata = json.load(f)

    # Load All Data
    full_train_df = pd.read_csv(args.train_csv)
    train_y = full_train_df[metadata["target_column"]].values.ravel()
    train_X_raw = full_train_df.drop(columns=[metadata["target_column"]])

    full_test_df = pd.read_csv(args.test_csv)
    test_y = full_test_df[metadata["target_column"]].values.ravel()
    test_X_raw = full_test_df.drop(columns=[metadata["target_column"]])

    y_binned = make_stratified_bins(train_y, n_bins=10)
    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    final_tr_idx, final_va_idx = next(iter(skf.split(train_X_raw, y_binned)))
    final_tr_X_raw = train_X_raw.iloc[final_tr_idx].reset_index(drop=True)
    final_va_X_raw = train_X_raw.iloc[final_va_idx].reset_index(drop=True)
    final_tr_y = train_y[final_tr_idx]
    final_va_y = train_y[final_va_idx]

    # When fine-tuning, the LLM tokenizer cannot be shared across forked workers
    dl_num_workers = 0 if getattr(args, "finetune_llm", False) else args.num_workers

    scaler_stats = {}
    for col in metadata["numerical_columns"]:
        if col in train_X_raw.columns:
            scaler_stats[col] = {
                "mean": float(train_X_raw[col].mean()),
                "std": float(train_X_raw[col].std()),
            }

    model_args = Namespace(**best_params)
    for k in ["target", "loss_type", "huber_delta", "use_molecule", "use_conventional"]:
        setattr(model_args, k, getattr(args, k, None))

    if not is_ablation:
        extractor = MolecularFeatureExtractor(
            extractor_type=model_args.mol_extractor_type,
            device=device,
        )
        model_args.mol_dim = extractor.config["feature_dim"]
        extractor.to(device)
    else:
        extractor = None
        model_args.mol_dim = 0

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
            train_X_raw, train_y, metadata, split="train", scaler_stats=scaler_stats
        )
        test_ds = SorbFormerDataset(
            test_X_raw, test_y, metadata, split="test", scaler_stats=scaler_stats
        )
        collate_fn = SorbFormer_collate_fn

    refit_tr_loader = DataLoader(
        refit_train_ds, batch_size=model_args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4
    )
    tr_loader = DataLoader(
        refit_train_ds, batch_size=model_args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4
    )
    test_loader = DataLoader(
        test_ds, batch_size=model_args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4
    )

    model = PlantUptakeModel(model_args, metadata=metadata, extractor=extractor)

    save_dir = (
        os.path.join(args.save_folder, f"trial_{trial_number}")
        if trial_number is not None
        else args.save_folder
    )
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "hparams.yaml"), "w") as f:
        yaml_dict = {
            k: (v.item() if isinstance(v, (np.integer, np.floating)) else v)
            for k, v in vars(model_args).items()
            if not k.startswith("_")
        }
        yaml.dump(yaml_dict, f, default_flow_style=False)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(save_dir, "checkpoints"),
        filename="best-{epoch:02d}",
        save_top_k=1,
        monitor="val/r2" if args.target == "rcf" else "val/rmse",
        mode="max" if args.target == "rcf" else "min",
        save_last=True,
    )

    logger = CSVLogger(save_dir, name="logs")

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=1,
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=5,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],
    )

    print("🏋️ Refitting model on combined Train+Val set...")
    trainer.fit(model, refit_tr_loader, test_loader)

    ckpt_path = checkpoint_callback.best_model_path if checkpoint_callback.best_model_path else None

    tr_results = trainer.test(model, tr_loader, ckpt_path=ckpt_path, verbose=False)[0]
    test_results = trainer.test(model, test_loader, ckpt_path=ckpt_path, verbose=False)[0]

    results = {
        "best_params": best_params,
        "cv_score": best_cv_score,
        "train_metrics": tr_results,
        "test_metrics": test_results,
    }

    results_file = os.path.join(save_dir, "final_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

    print("\n🏁 Metrics Summary:")
    print(f"   CV Score: {best_cv_score:.4f}")
    print(f"   Test RMSE: {test_results['test/rmse']:.4f}")
    if "test/r2" in test_results:
        print(f"   Test R2:   {test_results['test/r2']:.4f}")

    return results


# =========================================================
# 4. Main
# =========================================================
def main():
    import argparse
    
    torch.set_float32_matmul_precision("medium")

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True, help="Combined train_val CSV file")
    parser.add_argument("--test_csv", type=str, required=True, help="Combined test CSV file")
    parser.add_argument("--target", type=str, choices=["rcf", "tscf"], required=True)
    parser.add_argument("--save_folder", type=str, default="SorbFormer_results")
    parser.add_argument("--metadata_file", type=str, default="metadata_SorbFormer.json")
    parser.add_argument("--study_name", type=str, default="SorbFormer_v2_opt_r2")
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--max_epochs", type=int, default=100)

    # Default model params
    parser.add_argument(
        "--mol_extractor_type",
        type=str,
        default=None,
        choices=["biot5-plus-base", "chemberta-77m-mlm"],
    )
    parser.add_argument(
        "--fusion_type",
        type=str,
        default=None,
        choices=["gated", "cross_attention", "bilinear", "concat"],
    )
    parser.add_argument("--loss_type", type=str, default="huber")
    parser.add_argument("--use_molecule", action="store_true", default=True)
    parser.add_argument("--use_conventional", action="store_true", default=True)


    parser.add_argument(
        "--accelerator", type=str, default="gpu" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr_eta_min", type=float, default=1e-6)
    parser.add_argument("--num_layers", type=int, default=3)
    
    parser.add_argument("--finetune_llm", action="store_true", help="Finetune language model alongside prediction head")
    parser.add_argument("--unfreeze_all_llm", action="store_true", help="Unfreeze all parameters instead of top layers")

    args = parser.parse_args()
    os.makedirs(args.save_folder, exist_ok=True)

    db_path = os.path.join(args.save_folder, "optuna_study.db")
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10),
    )


    # Load Data (Pre-loading for statistics logging)
    with open(args.metadata_file, "r") as f:
        metadata = json.load(f)
    full_train_df = pd.read_csv(args.train_csv)
    train_y = full_train_df[metadata["target_column"]].values.ravel()

