"""
SorbFormer Model Interpretation Suite
Refactored into src.interpret package (Permutation Importance, Token Attribution IG, and Phylogenetic Analysis).
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Disable tokenizers parallelism to avoid deadlocks and warnings with DataLoader workers
os.environ["TOKENIZERS_PARALLELISM"] = "false"


import json
import pandas as pd
import numpy as np

import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.interpret import (
    load_model_and_data,
    permutation_importance,
    token_attribution,
    phylogenetic_signal,
    phylogenetic_tree_heatmap,
    clade_performance,
    taxonomic_feature_importance,
)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="SorbFormer Interpretation Suite")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--hparams", type=str, required=True)
    parser.add_argument("--test_X", type=str, required=True)
    parser.add_argument("--test_y", type=str, required=True)
    parser.add_argument("--train_X", type=str, default=None)
    parser.add_argument("--metadata_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--mechanisms", type=str, default="all")
    parser.add_argument(
        "--n_baselines",
        type=int,
        default=5,
        help="Number of IG baselines for uncertainty.",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- Section 1: Load model and data ---
    print("\n🚀 Loading model and data...")
    from src.interpret import load_model_and_data

    model, model_args, metadata, test_X, test_y, test_ds, test_loader, scaler_stats = (
        load_model_and_data(args)
    )

    if str(metadata.get("target_column")).upper() == "RCF" or (
        args.target and args.target.lower() == "rcf"
    ):
        print(
            "📊 RCF Target detected: Applying log10 scaling to test labels for model evaluation..."
        )
        test_y = np.log10(test_y.astype(float))

        # Re-initialize dataset and loader with logged test_y
        from src.interpret import SorbFormerDataset, SorbFormer_collate_fn

        test_ds = SorbFormerDataset(
            test_X,
            test_y,
            metadata,
            enumerator=None,
            split="test",
            scaler_stats=scaler_stats,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=32,
            shuffle=False,
            collate_fn=SorbFormer_collate_fn,
            num_workers=4,
        )

    # Save scaler stats for downstream unit conversion
    with open(os.path.join(args.output_dir, "scaler_stats.json"), "w") as f:
        json.dump(scaler_stats, f, indent=2)

    # --- Section 2: Performance Preamble ---
    if args.target is None:
        args.target = model_args.target

    print("\n📈 Model Performance Preamble")
    from src.interpret import predict_all, calculate_metrics

    device = next(model.parameters()).device
    preds, _ = predict_all(model, test_loader, device)
    metrics = calculate_metrics(test_y, preds, target=args.target)
    print(f"   Test R²   : {metrics['r2']:.4f}")
    print(f"   Test RMSE : {metrics['rmse']:.4f}")
    print(f"   Test MAE  : {metrics['mae']:.4f}")

    if metrics["r2"] < 0.5:
        print("   ⚠️ WARNING: Model R² is low (< 0.5). Interpret results with caution.")

    with open(os.path.join(args.output_dir, "performance_preamble.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    if args.mechanisms == "all":
        mechanisms = {
            "perm",
            "ig",
            "phylo",
        }
    else:
        mechanisms = set(args.mechanisms.split(","))

    # --- Section 3.1: Permutation Feature Importance (Section 3.4.1) ---
    if "perm" in mechanisms:
        try:
            permutation_importance(
                model,
                test_X,
                test_y,
                metadata,
                scaler_stats,
                model_args,
                args.output_dir,
            )
        except Exception as e:
            print(f"❌ Permutation Importance failed: {e}")

    # --- Section 3.2: Token Attribution & Integrated Gradients (Section 3.4.2) ---
    if "ig" in mechanisms:
        try:
            token_attribution(
                model,
                test_X,
                test_ds,
                args.output_dir,
                model_args.mol_extractor_type,
                n_baselines=args.n_baselines,
            )
        except Exception as e:
            print(f"❌ Token Attribution (IG) failed: {e}")

    # --- Section 3.3: Phylogenetic Analysis (Section 3.4.3) ---
    if "phylo" in mechanisms:
        phylo_dir = os.path.join(args.output_dir, "phylogenetic")
        os.makedirs(phylo_dir, exist_ok=True)
        try:
            phylogenetic_signal(test_X, test_y, metadata, args, phylo_dir)
        except Exception as e:
            print(f"❌ Phylogenetic Signal failed: {e}")
        try:
            phylogenetic_tree_heatmap(
                model, test_X, test_y, metadata, scaler_stats, args, phylo_dir
            )
        except Exception as e:
            print(f"❌ Tree Heatmap failed: {e}")
        try:
            clade_performance(
                model, test_X, test_y, metadata, scaler_stats, args, phylo_dir
            )
        except Exception as e:
            print(f"❌ Clade Performance failed: {e}")
        try:
            taxonomic_feature_importance(
                model, test_X, test_y, metadata, scaler_stats, args, phylo_dir
            )
        except Exception as e:
            print(f"❌ Taxonomic Importance failed: {e}")

    # --- Section 4: Summary & Output Organization ---

    power_summary = {
        "n_test_samples": int(len(test_y)),
        "statistical_power": (
            "Moderate"
            if 150 <= len(test_y) < 500
            else ("Good" if len(test_y) >= 500 else "Low")
        ),
        "permutation_repeats": 50,
        "target": args.target,
        "model": f"{model_args.mol_extractor_type}_{model_args.fusion_type}",
    }
    with open(os.path.join(args.output_dir, "statistical_summary.json"), "w") as f:
        json.dump(power_summary, f, indent=2)

    organize_output_directory(args.output_dir)

    print("\n" + "=" * 70)
    print("🎉 Interpretation suite complete!")
    print(f"   Outputs saved to: {args.output_dir}")
    print("=" * 70)


import shutil
import glob


def organize_output_directory(output_dir):
    """Group interpretation result files into categorized subfolders."""
    groups = {
        "permutation_importance": [
            "feature_importance.*",
        ],
        "attributions": [
            "functional_group_attribution*",
            "token_attribution*",
        ],
    }

    print("\n📁 Organizing output results into subfolders...")
    for group_name, patterns in groups.items():
        group_dir = os.path.join(output_dir, group_name)
        if isinstance(patterns, str):
            patterns = [patterns]

        files_to_move = []
        for pattern in patterns:
            files_to_move.extend(glob.glob(os.path.join(output_dir, pattern)))

        if files_to_move:
            os.makedirs(group_dir, exist_ok=True)
            for f in set(files_to_move):
                # Don't move a folder into itself
                if os.path.abspath(f) == os.path.abspath(group_dir):
                    continue
                shutil.move(f, os.path.join(group_dir, os.path.basename(f)))


if __name__ == "__main__":
    main()
