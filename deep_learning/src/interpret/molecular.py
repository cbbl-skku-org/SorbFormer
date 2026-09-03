import os
import re
import io
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageDraw as PILDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, MolFromSmarts, rdMolDescriptors, Crippen
from rdkit.Chem.Draw import SimilarityMaps, rdMolDraw2D
from scipy.stats import ttest_1samp, wilcoxon, spearmanr
from statsmodels.stats.multitest import multipletests


class MoleculeVisualizer:
    def __init__(self, model_type):
        self.model_type = model_type

    def get_atom_weights(self, smiles, tokens, scores):
        """Map token/SMILES/SELFIES scores to atom-level weights."""
        if isinstance(scores[0], list):
            # Handle multi-baseline scores: return mean and std
            weights_list = []
            for s_set in zip(
                *[scores[i] for i in range(len(scores))]
            ):  # This is wrong, correct below
                pass
            # Correct way to transpose if scores is list of lists
            scores_arr = np.array(scores)  # (len(tokens), n_baselines)
            mean_scores = scores_arr.mean(axis=1)
            std_scores = scores_arr.std(axis=1)

            mol, mean_w = self._map_to_mol(smiles, tokens, mean_scores.tolist())
            _, std_w = self._map_to_mol(smiles, tokens, std_scores.tolist())
            return mol, mean_w, std_w
        else:
            mol, weights = self._map_to_mol(smiles, tokens, scores)
            return mol, weights, None

    def _map_to_mol(self, smiles, tokens, scores):
        return (
            self._map_selfies(smiles, tokens, scores)
            if "biot5" in self.model_type.lower()
            else self._map_smiles(smiles, tokens, scores)
        )

    def _map_selfies(self, smiles, tokens, scores):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None

        atom_weights = [0.0] * mol.GetNumAtoms()
        atom_idx = 0

        for tok, score in zip(tokens, scores):
            if atom_idx >= len(atom_weights):
                break

            # SELFIES atoms are always in brackets like [C], [F], [Branch1_1]
            # Rule: Only map tokens that represent actual atoms.
            # Indicators like Branch, Ring, Bond, etc. are NOT atoms.
            is_atom = (
                "[" in tok
                and "]" in tok
                and not any(
                    p in tok for p in ["Branch", "Ring", "Bond", "Expl", "Phase"]
                )
            )

            if is_atom:
                atom_weights[atom_idx] = float(score)
                atom_idx += 1

        return mol, atom_weights

    def _map_smiles(self, smiles, tokens, scores):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None
        atom_weights, atom_idx, symbols = (
            [0.0] * mol.GetNumAtoms(),
            0,
            set(
                ["C", "O", "N", "S", "P", "F", "Cl", "Br", "I", "c", "o", "n", "s", "p"]
            ),
        )
        for tok, score in zip(tokens, scores):
            if atom_idx >= len(atom_weights):
                break
            if any(s in re.sub(r"[()=#0-9\[\]]", "", tok) for s in symbols):
                atom_weights[atom_idx] = float(score)
                atom_idx += 1
        return mol, atom_weights

    def save_heatmap(
        self,
        mol,
        weights,
        out_path,
        std_weights=None,
        title="",
        pred_val=None,
        target_val=None,
    ):
        try:
            mol = Chem.Mol(mol)
            AllChem.Compute2DCoords(mol)
            d2d = rdMolDraw2D.MolDraw2DCairo(1200, 1200)
            d2d.drawOptions().padding = 0.15
            d2d.drawOptions().setBackgroundColour(
                (1, 1, 1, 1)
            )  # Force pure white background
            d2d.drawOptions().clearBackground = True

            # Draw base heatmap
            SimilarityMaps.GetSimilarityMapFromWeights(
                mol, weights, draw2d=d2d, colorMap="bwr", contourLines=10, alpha=0.2
            )
            d2d.FinishDrawing()

            img = Image.open(io.BytesIO(d2d.GetDrawingText())).convert("RGBA")

            # We no longer draw pred/target text since they are in the JSON metadata.

            # Crop white background using ImageChops
            from PIL import ImageChops

            bg = Image.new(img.mode, img.size, (255, 255, 255, 255))
            diff = ImageChops.difference(img, bg)
            # Find the bounding box of the non-white pixels
            bbox = diff.getbbox()
            if bbox:
                # Add a 150-pixel padding around the molecule
                padding = 150
                left = max(0, bbox[0] - padding)
                top = max(0, bbox[1] - padding)
                right = min(img.width, bbox[2] + padding)
                bottom = min(img.height, bbox[3] + padding)
                img = img.crop((left, top, right, bottom))

            img.convert("RGB").save(out_path, dpi=(600, 600))
        except Exception as e:
            print(f"      ⚠️ Visualization failed: {e}")


def token_attribution(
    model, test_X, test_ds, output_dir, model_type, max_samples=None, n_baselines=5
):
    print("\n" + "=" * 70)
    print("🧬 IG & Molecular Visualization (Enhanced)")
    print("=" * 70)
    visualizer = MoleculeVisualizer(model_type)
    smiles_list = (
        test_X["SMILES"].tolist()[:max_samples]
        if max_samples
        else test_X["SMILES"].tolist()
    )
    all_attributions = []
    m_dir = os.path.join(output_dir, "mol_visualization")
    os.makedirs(m_dir, exist_ok=True)

    device = next(model.parameters()).device

    for i, smi in enumerate(tqdm(smiles_list, desc="   IG attribution")):
        try:
            batch = test_ds[i]
            # Extract categorical and numerical features from batch
            cat_feats = {
                k: v.unsqueeze(0).to(device)
                for k, v in batch["categorical_feats"].items()
                if k.startswith("cat_")
            }
            num_feats = {
                k: v.unsqueeze(0).to(device)
                for k, v in batch["numerical_feats"].items()
                if k.startswith("num_")
            }

            explanations = model.explain_molecule(
                [smi],
                cat_feats,
                num_feats,
                n_baselines=n_baselines,
            )

            if explanations and explanations[0]:
                tokens, scores = zip(*explanations[0])

                # Check for RCF target to revert log transform
                target_val = float(batch["targets"])
                with torch.no_grad():
                    pred_tensor, _ = model(
                        [smi],
                        cat_feats,
                        num_feats,
                    )
                    pred_val = float(pred_tensor.squeeze())

                is_rcf = (
                    hasattr(model.args, "target") and model.args.target.lower() == "rcf"
                )
                if is_rcf:
                    target_val = np.power(10.0, target_val)
                    pred_val = np.power(10.0, pred_val)

                # Collapse multi-baseline scores to mean for global stats, keep all for stability
                if n_baselines > 1:
                    mean_scores = [
                        np.mean(s) if isinstance(s, list) else s for s in scores
                    ]
                else:
                    mean_scores = scores

                all_attributions.append(
                    {
                        "sample_idx": i,
                        "smiles": smi,
                        "tokens": list(tokens),
                        "scores": scores,
                        "mean_scores": [float(s) for s in mean_scores],
                        "target_val": target_val,
                        "pred_val": pred_val,
                    }
                )



                # Molecular heatmap with uncertainty
                mol, weights, std_weights = visualizer.get_atom_weights(
                    smi, tokens, scores
                )
                if mol and weights:
                    visualizer.save_heatmap(
                        mol,
                        weights,
                        os.path.join(m_dir, f"sample_{i}_molecule.png"),
                        std_weights=std_weights,
                        pred_val=pred_val,
                        target_val=target_val,
                    )

                    # Save metadata corresponding to this sample
                    # We convert numpy types to native Python types for JSON serialization
                    raw_features = test_X.iloc[i].to_dict()
                    safe_features = {
                        k: (float(v) if pd.api.types.is_numeric_dtype(type(v)) else v)
                        for k, v in raw_features.items()
                    }

                    sample_metadata = {
                        "sample_idx": i,
                        "smiles": smi,
                        "target_val": float(target_val),
                        "pred_val": float(pred_val),
                        "features": safe_features,
                        "tokens": list(tokens),
                        "token_attribution_scores": [float(s) for s in mean_scores],
                        "atom_weights": [float(w) for w in weights] if weights else [],
                    }
                    with open(
                        os.path.join(m_dir, f"sample_{i}_metadata.json"), "w"
                    ) as mf:
                        json.dump(sample_metadata, mf, indent=4)

        except Exception as e:
            print(f"   ⚠️ Error on sample {i}: {e}")
            continue

    with open(os.path.join(output_dir, "token_attributions.json"), "w") as f:
        json.dump(all_attributions, f, indent=2)

    # Rank samples by prediction quality (lowest error first) to help select high-quality visualizations
    if all_attributions:
        ranking_data = []
        for attr in all_attributions:
            err = abs(attr["target_val"] - attr["pred_val"])
            ranking_data.append({
                "sample_idx": attr["sample_idx"],
                "smiles": attr["smiles"],
                "target_val": attr["target_val"],
                "pred_val": attr["pred_val"],
                "abs_error": err
            })
        ranking_df = pd.DataFrame(ranking_data).sort_values("abs_error", ascending=True)
        ranking_df.to_csv(os.path.join(m_dir, "sample_quality_ranking.csv"), index=False)

    # Global Importance
    token_importance = {}
    for attr in all_attributions:
        for tok, score in zip(attr["tokens"], attr["mean_scores"]):
            if tok not in token_importance:
                token_importance[tok] = []
            token_importance[tok].append(abs(score))
    agg_df = pd.DataFrame(
        [
            {"token": tok, "mean_abs_attribution": np.mean(s), "count": len(s)}
            for tok, s in token_importance.items()
        ]
    ).sort_values("mean_abs_attribution", ascending=False)
    agg_df.to_csv(os.path.join(output_dir, "token_importance_global.csv"), index=False)

    if all_attributions:
        attribution_significance(all_attributions, output_dir)
        attribution_stability(
            model, test_X, test_ds, output_dir, n_baselines=n_baselines
        )
        functional_group_attribution(all_attributions, output_dir)



    return all_attributions





def attribution_significance(all_attributions, output_dir, n_bootstrap=500):
    print("   Running significance tests...")
    token_scores = {}
    for attr in all_attributions:
        for tok, score in zip(attr["tokens"], attr["mean_scores"]):
            if tok not in token_scores:
                token_scores[tok] = []
            token_scores[tok].append(score)
    results = []
    for tok, scores in token_scores.items():
        scores = np.array(scores)
        if len(scores) < 5:
            continue
        _, p_t = ttest_1samp(scores, 0.0)
        try:
            _, p_w = wilcoxon(scores, alternative="two-sided")
        except:
            p_w = 1.0
        boot_means = [
            np.random.choice(scores, len(scores)).mean() for _ in range(n_bootstrap)
        ]
        ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
        results.append(
            {
                "token": tok,
                "n_molecules": len(scores),
                "mean_attribution": float(scores.mean()),
                "mean_abs_attribution": float(np.abs(scores).mean()),
                "ci_95_lower": float(ci_lo),
                "ci_95_upper": float(ci_hi),
                "direction": "positive" if scores.mean() > 0 else "negative",
                "p_ttest": float(p_t),
                "p_wilcoxon": float(p_w),
            }
        )
    sig_df = pd.DataFrame(results).sort_values("mean_abs_attribution", ascending=False)
    if not sig_df.empty:
        _, sig_df["p_ttest_adj"], _, _ = multipletests(
            sig_df["p_ttest"], method="fdr_bh"
        )
        _, sig_df["p_wilcox_adj"], _, _ = multipletests(
            sig_df["p_wilcoxon"], method="fdr_bh"
        )
        sig_df["significant"], sig_df["ci_excludes_zero"] = (
            sig_df["p_ttest_adj"] < 0.05
        ) & (sig_df["p_wilcox_adj"] < 0.05), (sig_df["ci_95_lower"] > 0) | (
            sig_df["ci_95_upper"] < 0
        )
    sig_df.to_csv(
        os.path.join(output_dir, "token_attribution_significance.csv"), index=False
    )
    sig_only = (
        sig_df[sig_df["significant"] & sig_df["ci_excludes_zero"]]
        if not sig_df.empty
        else pd.DataFrame()
    )
    if not sig_only.empty:
        exclude = ['<bom>', '<eom>', '</s>']
        plot_df = sig_only[~sig_only['token'].isin(exclude)]
        plot_df = plot_df.sort_values('mean_abs_attribution', ascending=False).head(15).iloc[::-1]
        
        if not plot_df.empty:
            fig, ax = plt.subplots(figsize=(5.5, 8))
            ax.grid(axis='x', linestyle='-', alpha=0.3, color='gray')
            ax.set_axisbelow(True)
            ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
            
            colors = ['#E63970' if d == 'positive' else '#2096F3' for d in plot_df['direction']]
            means = plot_df['mean_attribution'].values
            y_pos = np.arange(len(plot_df))
            
            lower = plot_df['ci_95_lower'].values
            upper = plot_df['ci_95_upper'].values
            xerr = np.abs([means - lower, upper - means])
            
            ax.barh(y_pos, means, color=colors, xerr=xerr, height=0.6, error_kw={'linewidth': 1.5, 'capsize': 0})
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(plot_df['token'], fontsize=16)
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_color('gray')
            ax.tick_params(axis='y', length=0)
            ax.tick_params(axis='x', labelsize=12)
            ax.set_xlabel('Mean Attribution', fontsize=14, fontweight='bold')
            
            plt.subplots_adjust(left=0.40, right=0.95, top=0.95, bottom=0.10)
            fig.savefig(
                os.path.join(output_dir, "token_attribution_significant.png"),
                dpi=600,
                facecolor="white",
            )
            plt.close(fig)
    return sig_df


def attribution_stability(model, test_X, test_ds, output_dir, n_baselines=10):
    print("   Running stability check...")
    results = []
    device = next(model.parameters()).device
    # Use first 5 samples for deeper stability analysis if n_baselines > 1
    for i, smi in enumerate(test_X["SMILES"].tolist()[:5]):
        try:
            batch = test_ds[i]
            # Extract categorical and numerical features from batch
            cat_feats_stability = {
                k: v.unsqueeze(0).to(device)
                for k, v in batch["categorical_feats"].items()
                if k.startswith("cat_")
            }
            num_feats_stability = {
                k: v.unsqueeze(0).to(device)
                for k, v in batch["numerical_feats"].items()
                if k.startswith("num_")
            }

            ex = model.explain_molecule(
                [smi],
                cat_feats_stability,
                num_feats_stability,
                n_baselines=n_baselines,
            )
            if ex and ex[0] and isinstance(ex[0][0][1], list):
                # Calculate mean standard deviation across atoms as stability metric
                all_scores = np.array([e[1] for e in ex[0]])  # (n_tokens, n_baselines)
                avg_std = np.mean(np.std(all_scores, axis=1))
                success = True
            else:
                avg_std = 0.0
                success = ex is not None and len(ex) > 0
        except:
            success = False
            avg_std = 0.0
        results.append(
            {
                "sample_idx": i,
                "smiles": smi,
                "ig_succeeded": success,
                "avg_token_std": avg_std,
                "n_baselines": n_baselines,
            }
        )
    pd.DataFrame(results).to_csv(
        os.path.join(output_dir, "attribution_stability.csv"), index=False
    )
    return results


def functional_group_attribution(all_attributions, output_dir):
    print("   Mapping functional groups...")
    fgs = {
        "carboxylic_acid": "[CX3](=O)[OX2H1]",
        "hydroxyl": "[OX2H]",
        "amine": "[NX3;H2,H1;!$(NC=O)]",
        "amide": "[NX3][CX3](=[OX1])",
        "ester": "[#6][CX3](=O)[OX2H0][#6]",
        "ketone": "[#6][CX3](=O)[#6]",
        "halogen_Cl": "[Cl]",
        "halogen_F": "[F]",
        "halogen_Br": "[Br]",
        "aromatic_ring": "c1ccccc1",
        "sulfonate": "[SX4](=O)(=O)[OX2H,OX1-]",
        "phosphate": "[PX4](=O)([OX2H])[OX2H]",
        "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
    }
    group_scores = {g: [] for g in fgs}
    for attr in all_attributions:
        mol = Chem.MolFromSmiles(attr["smiles"])
        if mol is None:
            continue
        n_at = mol.GetNumAtoms()
        weights = (
            attr["mean_scores"][:n_at]
            if len(attr["mean_scores"]) >= n_at
            else attr["mean_scores"] + [0.0] * (n_at - len(attr["mean_scores"]))
        )
        for g, smarts in fgs.items():
            pat = MolFromSmarts(smarts)
            if pat:
                for match in mol.GetSubstructMatches(pat):
                    group_scores[g].append(
                        np.mean([weights[idx] for idx in match if idx < len(weights)])
                    )
    rows = []
    for g, s_l in group_scores.items():
        if len(s_l) >= 3:
            s_a = np.array(s_l)
            _, p = ttest_1samp(s_a, 0.0)
            rows.append(
                {
                    "functional_group": g,
                    "n_occurrences": len(s_l),
                    "mean_attribution": float(s_a.mean()),
                    "std_attribution": float(s_a.std()),
                    "p_value": float(p),
                    "effect": (
                        "increases uptake" if s_a.mean() > 0 else "decreases uptake"
                    ),
                }
            )
    df = pd.DataFrame(rows).sort_values("mean_attribution", key=abs, ascending=False)
    if not df.empty:
        _, df["p_adj"], _, _ = multipletests(df["p_value"], method="fdr_bh")
        df["significant"] = df["p_adj"] < 0.05
        df.to_csv(
            os.path.join(output_dir, "functional_group_attribution.csv"), index=False
        )
        plot_df = df.iloc[::-1]
        fig, ax = plt.subplots(figsize=(5.5, 8))
        ax.grid(axis='x', linestyle='-', alpha=0.3, color='gray')
        ax.set_axisbelow(True)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
        
        colors = ['#E63970' if v > 0 else '#2096F3' for v in plot_df['mean_attribution']]
        means = plot_df['mean_attribution'].values
        y_pos = np.arange(len(plot_df))
        
        ax.barh(y_pos, means, color=colors, height=0.6)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_df['functional_group'], fontsize=16)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('gray')
        ax.tick_params(axis='y', length=0)
        ax.tick_params(axis='x', labelsize=12)
        ax.set_xlabel('Mean Attribution', fontsize=14, fontweight='bold')
        
        plt.subplots_adjust(left=0.40, right=0.95, top=0.95, bottom=0.10)
        fig.savefig(
            os.path.join(output_dir, "functional_group_attribution.png"),
            dpi=600,
            facecolor="white",
        )
        plt.close(fig)
    return df
