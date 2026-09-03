import pandas as pd
import numpy as np
import torch
import json
from torch.utils.data import Dataset

class SorbFormerDataset(Dataset):
    def __init__(
        self, X_df, y_array, metadata, split="train", scaler_stats=None
    ):
        self.X_df = X_df.reset_index(drop=True)
        self.y = y_array
        self.metadata = metadata
        self.split = split
        self.scaler_stats = scaler_stats

        self.cat_cols = [
            c for c in metadata["categorical_columns"] if c in X_df.columns
        ]
        self.num_cols = [c for c in metadata["numerical_columns"] if c in X_df.columns]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        row = self.X_df.iloc[idx]
        smiles = row["SMILES"]

        cat_feats = {}
        for col in self.cat_cols:
            cat_val = row[col]
            cat_idx = self.metadata["categorical_maps"][col].get(cat_val, 0)  # 0 = <UNK>/padding
            cat_feats[f"cat_{col}"] = torch.tensor(cat_idx, dtype=torch.long)

        num_feats = {}
        for col in self.num_cols:
            val = float(row[col])
            if self.scaler_stats and col in self.scaler_stats:
                stats = self.scaler_stats[col]
                mean = stats["mean"]
                std = stats["std"]
                val = (val - mean) / (std + 1e-8)
            num_feats[f"num_{col}"] = torch.tensor(val, dtype=torch.float)

        target_val = float(self.y[idx])
        if self.metadata["target_column"] == "RCF":
            target_val = np.log10(target_val)
        target_tensor = torch.tensor(target_val, dtype=torch.float)

        return {
            "smiles": smiles,
            "categorical_feats": cat_feats,
            "numerical_feats": num_feats,
            "targets": target_tensor,
        }


def SorbFormer_collate_fn(batch):
    smiles = [item["smiles"] for item in batch]
    targets = torch.stack([item["targets"] for item in batch])

    categorical_feats = {}
    if batch[0]["categorical_feats"]:
        for key in batch[0]["categorical_feats"].keys():
            categorical_feats[key] = torch.stack(
                [item["categorical_feats"][key] for item in batch]
            )

    numerical_feats = {}
    if batch[0]["numerical_feats"]:
        for key in batch[0]["numerical_feats"].keys():
            numerical_feats[key] = torch.stack(
                [item["numerical_feats"][key] for item in batch]
            )

    return {
        "smiles": smiles,
        "numerical_feats": numerical_feats,
        "categorical_feats": categorical_feats,
        "targets": targets,
        "molecular_feats": None,
    }
