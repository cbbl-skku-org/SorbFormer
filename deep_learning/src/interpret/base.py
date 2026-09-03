import os
import json
import yaml
import torch
import numpy as np
from src.data import SorbFormerDataset, SorbFormer_collate_fn
import pandas as pd
from argparse import Namespace
from torch.utils.data import Dataset, DataLoader
from src.model import PlantUptakeModel
from src.molecules.feature_extractor import MolecularFeatureExtractor

# Inline Dataset to avoid import issues

def load_model_and_data(args):
    """Load trained model from checkpoint + hparams, and prepare test data."""
    with open(args.hparams, "r") as f:
        hparams = yaml.safe_load(f)

    model_args = Namespace(**hparams)
    model_args.test_X_csv = args.test_X
    model_args.test_y_csv = args.test_y
    if hasattr(args, "metadata_file") and args.metadata_file:
        model_args.metadata_file = args.metadata_file

    with open(model_args.metadata_file, "r") as f:
        metadata = json.load(f)

    test_X = pd.read_csv(args.test_X)
    test_y = pd.read_csv(args.test_y).values.ravel()
    
    # Pre-process inputs

    # Use train_X_csv if present, else fallback to train_csv
    train_path = getattr(model_args, "train_X_csv", getattr(model_args, "train_csv", None))
    if not train_path:
        raise AttributeError("Could not find 'train_X_csv' or 'train_csv' in model hyperparameters.")
    
    train_X = pd.read_csv(train_path)
    scaler_stats = {}
    for col in metadata["numerical_columns"]:
        if col in train_X.columns:
            scaler_stats[col] = {
                "mean": float(train_X[col].mean()),
                "std": float(train_X[col].std()),
            }

    extractor = MolecularFeatureExtractor(model_args.mol_extractor_type)
    model_args.mol_dim = extractor.config["feature_dim"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor.to(device)

    model = PlantUptakeModel.load_from_checkpoint(
        args.checkpoint,
        args=model_args,
        molecular_extractor=extractor,
    )
    model.eval()
    model.to(device)

    test_ds = SorbFormerDataset(
        test_X, test_y, metadata,
        split="test", scaler_stats=scaler_stats,
    )
    test_loader = DataLoader(
        test_ds, batch_size=32, shuffle=False,
        collate_fn=SorbFormer_collate_fn, num_workers=4,
    )

    return model, model_args, metadata, test_X, test_y, test_ds, test_loader, scaler_stats

def predict_batch(model, batch, device):
    """Run a forward pass and return predictions as numpy."""
    with torch.no_grad():
        preds, gates = model(
            batch["smiles"],
            batch["categorical_feats"],
            batch["numerical_feats"],
            molecular_features=batch.get("molecular_feats"),
        )
    return preds.cpu().numpy(), gates.cpu().numpy() if gates is not None else None

def predict_all(model, loader, device):
    """Run predictions on entire dataloader."""
    all_preds, all_gates = [], []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        preds, gates = predict_batch(model, batch, device)
        all_preds.append(preds)
        if gates is not None:
            all_gates.append(gates)
    preds = np.concatenate(all_preds)
    gates = np.concatenate(all_gates) if all_gates else None
    return preds, gates

def move_batch_to_device(batch, device):
    """Move batch tensors to device."""
    new_batch = {"smiles": batch["smiles"]}
    new_batch["categorical_feats"] = {
        k: v.to(device) for k, v in batch["categorical_feats"].items()
    }
    new_batch["numerical_feats"] = {
        k: v.to(device) for k, v in batch["numerical_feats"].items()
    }
    new_batch["targets"] = batch["targets"].to(device)
    new_batch["molecular_feats"] = batch.get("molecular_feats")
    return new_batch

def calculate_metrics(y_true, y_pred, target):
    """Calculates R2 and RMSE, reverting log-transform if target is rcf."""
    from sklearn.metrics import r2_score, mean_squared_error
    y_t, y_p = np.array(y_true), np.array(y_pred)
    if target.lower() == "rcf":
        y_t, y_p = np.power(10.0, y_t), np.power(10.0, y_p)
    return {"r2": r2_score(y_t, y_p), "rmse": np.sqrt(mean_squared_error(y_t, y_p)), "mae": np.mean(np.abs(y_t - y_p))}
