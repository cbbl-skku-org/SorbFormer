import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests
from torch.utils.data import DataLoader
from matplotlib.patches import Patch
from .base import SorbFormerDataset, SorbFormer_collate_fn, predict_all, calculate_metrics
from .chem_utils import annotate_smiles


def permutation_importance(
    model, test_X, test_y, metadata, scaler_stats, args, output_dir, n_repeats=50
):
    """Rank features by their impact on test-set R², with significance testing."""
    print("\n" + "=" * 70)
    print("📊 Permutation Feature Importance")
    print("=" * 70)

    device = next(model.parameters()).device
    all_features = metadata["numerical_columns"] + metadata["categorical_columns"]

    base_ds = SorbFormerDataset(
        test_X,
        test_y,
        metadata,
        enumerator=None,
        split="test",
        scaler_stats=scaler_stats,
    )
    base_loader = DataLoader(
        base_ds,
        batch_size=32,
        shuffle=False,
        collate_fn=SorbFormer_collate_fn,
        num_workers=4,
    )
    base_preds, _ = predict_all(model, base_loader, device)

    base_metrics = calculate_metrics(test_y, base_preds, args.target)
    base_r2, base_rmse = base_metrics["r2"], base_metrics["rmse"]
    print(f"   Baseline Test R²: {base_r2:.4f}, RMSE: {base_rmse:.4f}")

    importances = []
    for feat in tqdm(all_features, desc="   Permuting features"):
        r2_drops = []
        for _ in range(n_repeats):
            shuffled_X = test_X.copy()
            shuffled_X[feat] = np.random.permutation(shuffled_X[feat].values)
            shuf_ds = SorbFormerDataset(
                shuffled_X,
                test_y,
                metadata,
                enumerator=None,
                split="test",
                scaler_stats=scaler_stats,
            )
            shuf_loader = DataLoader(
                shuf_ds,
                batch_size=32,
                shuffle=False,
                collate_fn=SorbFormer_collate_fn,
                num_workers=4,
            )
            shuf_preds, _ = predict_all(model, shuf_loader, device)
            shuf_metrics = calculate_metrics(test_y, shuf_preds, args.target)
            r2_drops.append(base_r2 - shuf_metrics["r2"])

        t_stat, p_val = ttest_1samp(r2_drops, popmean=0.0, alternative="greater")
        perm_p_val = float(np.mean(np.array(r2_drops) <= 0))
        importances.append(
            {
                "feature": feat,
                "importance_mean": np.mean(r2_drops),
                "importance_std": np.std(r2_drops),
                "type": (
                    "numerical"
                    if feat in metadata["numerical_columns"]
                    else "categorical"
                ),
                "t_stat": float(t_stat),
                "p_value_ttest": float(p_val),
                "p_value_perm": perm_p_val,
            }
        )

    imp_df = pd.DataFrame(importances).sort_values("importance_mean", ascending=False)
    imp_df["p_value"] = imp_df[["p_value_ttest", "p_value_perm"]].max(axis=1)
    reject, p_adj, _, _ = multipletests(
        imp_df["p_value"].values, method="fdr_bh", alpha=0.05
    )
    imp_df["p_value_adj"], imp_df["significant_fdr05"] = p_adj, reject
    def get_feature_domain(feat):
        if feat in ["t_dur (days)", "exposure_medium", "pH"]:
            return "Experimental/Environmental"
        elif any(term in feat.lower() for term in ["plant", "trait", "genus", "species", "order", "class", "family"]):
            return "Biological Trait"
        else:
            return "Chemical Descriptor"

    imp_df["domain"] = imp_df["feature"].apply(get_feature_domain)
    imp_df.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)

    fig, ax = plt.subplots(figsize=(16, 4.5))
    
    domain_colors = {
        "Chemical Descriptor": "#FF5722", 
        "Biological Trait": "#4CAF50", 
        "Experimental/Environmental": "#9C27B0"
    }
    colors = [domain_colors.get(d, '#333333') for d in imp_df["domain"]]
    
    x_pos = np.arange(len(imp_df))
    means = imp_df["importance_mean"].values
    stds = imp_df["importance_std"].values
    
    ax.bar(
        x_pos, 
        means, 
        color=colors, 
        yerr=stds, 
        width=0.7, 
        error_kw={'linewidth': 1.2, 'capsize': 0}
    )
    
    for i, (mean, std, p) in enumerate(zip(means, stds, imp_df['p_value_adj'])):
        if p < 0.05:
            marker = "***" if p < 0.001 else "**" if p < 0.01 else "*"
            ax.text(
                i, 
                mean + std + 0.005, 
                marker, 
                ha="center", 
                va="bottom", 
                fontsize=12, 
                fontweight="bold"
            )
            
    ax.set_xticks(x_pos)
    ax.set_xticklabels(imp_df["feature"], fontsize=12, rotation=45, ha='right', rotation_mode='anchor')
    ax.set_ylabel('$\Delta R^2$ (importance)\n* p<0.05, ** p<0.01, *** p<0.001 (FDR)', fontsize=11)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_color('gray')
    ax.tick_params(axis='x', length=0)
    ax.tick_params(axis='y', labelsize=10)
    
    ax.legend(
        handles=[
            Patch(facecolor="#FF5722", label="Chemical"),
            Patch(facecolor="#4CAF50", label="Biological"),
            Patch(facecolor="#9C27B0", label="Experimental"),
        ],
        loc="upper right",
        fontsize=11
    )
    
    plt.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.35)
    fig.savefig(
        os.path.join(output_dir, "feature_importance.png"), dpi=600, facecolor="white"
    )
    plt.close(fig)
    return imp_df
