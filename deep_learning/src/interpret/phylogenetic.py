"""
Phylogenetic Interpretation Module
===================================
Phylogenetic signal, tree visualization, clade performance,
and taxonomic feature importance.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from .base import SorbFormerDataset, SorbFormer_collate_fn, predict_all, calculate_metrics


# =========================================================
#  Phylogenetic Signal Analysis
# =========================================================
def phylogenetic_signal(test_X, test_y, metadata, args, output_dir):
    """Test whether uptake traits show phylogenetic signal using Blomberg's K analog."""
    print("\n" + "=" * 70)
    print("🌿 Phylogenetic Signal Analysis")
    print("=" * 70)

    taxonomic_levels = ["Class", "Order", "Family", "Genus", "Species"]
    available_levels = [t for t in taxonomic_levels if t in test_X.columns]

    if not available_levels:
        print("   ⚠️ No taxonomic columns found in test data. Skipping.")
        return None

    y_vals = np.array(test_y)
    if args.target == "rcf":
        y_vals = np.power(10.0, y_vals)

    results = []
    total_var = np.var(y_vals)

    for level in available_levels:
        groups = test_X[level].values
        unique_groups = np.unique(groups)

        if len(unique_groups) < 2:
            continue

        # Between-group variance / total variance (eta-squared analog)
        group_means = {g: y_vals[groups == g].mean() for g in unique_groups}
        group_sizes = {g: np.sum(groups == g) for g in unique_groups}
        grand_mean = y_vals.mean()

        ss_between = sum(
            group_sizes[g] * (group_means[g] - grand_mean) ** 2
            for g in unique_groups
        )
        ss_total = total_var * len(y_vals)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0

        # Permutation test for significance
        n_perm = 999
        perm_eta = []
        for _ in range(n_perm):
            perm_y = np.random.permutation(y_vals)
            perm_means = {g: perm_y[groups == g].mean() for g in unique_groups}
            perm_ss = sum(
                group_sizes[g] * (perm_means[g] - grand_mean) ** 2
                for g in unique_groups
            )
            perm_eta.append(perm_ss / ss_total if ss_total > 0 else 0)

        p_value = (np.sum(np.array(perm_eta) >= eta_squared) + 1) / (n_perm + 1)

        results.append({
            "taxonomic_level": level,
            "n_groups": len(unique_groups),
            "eta_squared": float(eta_squared),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
        })
        print(f"   {level}: η² = {eta_squared:.4f}, p = {p_value:.4f} "
              f"({'*' if p_value < 0.05 else 'ns'})")

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        os.path.join(output_dir, "phylogenetic_signal.csv"), index=False
    )

    # Plot
    if not results_df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#4CAF50" if s else "#9E9E9E" for s in results_df["significant"]]
        bars = ax.bar(
            results_df["taxonomic_level"], results_df["eta_squared"],
            color=colors, edgecolor="white",
        )
        for bar, row in zip(bars, results_df.itertuples()):
            marker = "***" if row.p_value < 0.001 else (
                "**" if row.p_value < 0.01 else ("*" if row.p_value < 0.05 else "ns")
            )
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                marker, ha="center", fontsize=10, fontweight="bold",
            )
        ax.set_ylabel("η² (Variance Explained)")
        ax.set_title(f"Phylogenetic Signal in {args.target.upper()} Uptake")
        ax.set_ylim(0, min(1.0, results_df["eta_squared"].max() * 1.5 + 0.05))
        plt.tight_layout()
        fig.savefig(
            os.path.join(output_dir, "phylogenetic_signal.png"),
            dpi=300, bbox_inches="tight",
        )
        plt.close(fig)

    return results_df


# =========================================================
#  Phylogenetic Tree + Heatmap
# =========================================================
def phylogenetic_tree_heatmap(
    model, test_X, test_y, metadata, scaler_stats, args, output_dir
):
    """Construct a taxonomy-based tree annotated with uptake values per species."""
    print("\n" + "=" * 70)
    print("🌳 Phylogenetic Tree + Uptake Heatmap")
    print("=" * 70)

    if "Species" not in test_X.columns:
        print("   ⚠️ 'Species' column not found. Skipping.")
        return None

    device = next(model.parameters()).device
    test_loader = DataLoader(
        SorbFormerDataset(test_X, test_y, metadata, None, "test", scaler_stats),
        batch_size=32, shuffle=False, collate_fn=SorbFormer_collate_fn,
    )
    preds, _ = predict_all(model, test_loader, device)

    y_true = np.power(10.0, test_y) if args.target == "rcf" else np.array(test_y)
    y_pred = np.power(10.0, preds) if args.target == "rcf" else preds

    # Aggregate per species
    species_data = []
    for sp in test_X["Species"].unique():
        mask = test_X["Species"].values == sp
        n = mask.sum()
        sp_true = y_true[mask]
        sp_pred = y_pred[mask]
        sp_err = np.abs(sp_true - sp_pred)

        row = {
            "Species": sp,
            "n_samples": int(n),
            "mean_true": float(sp_true.mean()),
            "std_true": float(sp_true.std()) if n > 1 else 0.0,
            "mean_pred": float(sp_pred.mean()),
            "mean_abs_error": float(sp_err.mean()),
        }

        # Add taxonomy
        sp_idx = np.where(mask)[0][0]
        for level in ["Class", "Order", "Family", "Genus"]:
            if level in test_X.columns:
                row[level] = test_X.iloc[sp_idx][level]

        species_data.append(row)

    sp_df = pd.DataFrame(species_data).sort_values(
        ["Class", "Order", "Family", "Genus", "Species"]
    )
    sp_df.to_csv(
        os.path.join(output_dir, "species_uptake_summary.csv"), index=False
    )

    # Heatmap-style bar chart (species on y-axis, bars for true/pred/error)
    fig, axes = plt.subplots(1, 3, figsize=(16, max(6, len(sp_df) * 0.4)))

    species_labels = [
        f"{row['Species']} ({row['Family']})" if "Family" in row else row["Species"]
        for _, row in sp_df.iterrows()
    ]
    y_pos = range(len(sp_df))

    # True uptake
    axes[0].barh(y_pos, sp_df["mean_true"], color="#4CAF50", edgecolor="white")
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(species_labels, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel(f"Mean True {args.target.upper()}")
    axes[0].set_title("Actual Uptake")

    # Predicted uptake
    axes[1].barh(y_pos, sp_df["mean_pred"], color="#2196F3", edgecolor="white")
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels([])
    axes[1].invert_yaxis()
    axes[1].set_xlabel(f"Mean Predicted {args.target.upper()}")
    axes[1].set_title("Predicted Uptake")

    # Error
    axes[2].barh(y_pos, sp_df["mean_abs_error"], color="#FF5722", edgecolor="white")
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels([])
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Mean |Error|")
    axes[2].set_title("Prediction Error")

    # Color-code by Class (monocot vs dicot)
    if "Class" in sp_df.columns:
        for i, (_, row) in enumerate(sp_df.iterrows()):
            bg_color = "#E8F5E9" if row["Class"] == "Liliopsida" else "#E3F2FD"
            for ax in axes:
                ax.axhspan(i - 0.5, i + 0.5, color=bg_color, alpha=0.3, zorder=0)

    plt.suptitle(
        f"Species-Level {args.target.upper()} Uptake Summary",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "phylogenetic_tree_heatmap.png"),
        dpi=300, bbox_inches="tight",
    )
    plt.close(fig)
    
    # GBIF Taxonomic Tree Visualization
    try:
        print("   [ETE3] Rendering GBIF taxonomic tree...")
        render_gbif_taxonomic_tree(sp_df, args.target, output_dir)
        print("   ✅ GBIF taxonomic tree rendered successfully.")
    except Exception as e:
        print(f"   ⚠️ ETE3 tree rendering failed: {e}")

    # Radial Bubble Clade Plot (Friend's suggestion)
    try:
        print("   [Matplotlib] Rendering radial bubble clade plot...")
        render_radial_bubble_clade_plot(sp_df, args.target, output_dir)
        print("   ✅ Radial bubble plot rendered successfully.")
    except Exception as e:
        print(f"   ⚠️ Radial bubble plot rendering failed: {e}")

    print(f"   ✅ {len(sp_df)} species summarized.")
    return sp_df


# =========================================================
# Radial Bubble Clade Plot (pathway-style aesthetic)
# =========================================================
def render_radial_bubble_clade_plot(
    sp_df, target_name, output_dir, group_level="Order"
):
    """
    Pathway-enrichment-style radial bubble plot for phylogenetic clades.

    - Outer ring: species labels (italic, colored by clade)
    - Inner ring: clade 'hub' bubbles, sized by # species in clade
    - Spokes:    color-coded lines hub -> species
    - Leaf dot:  optional encoding of prediction error magnitude

    Parameters
    ----------
    sp_df : pd.DataFrame
        Must contain 'Species' and `group_level`. Optionally 'mean_abs_error'.
    target_name : str
        e.g. 'rcf' or 'tscf' for the title.
    group_level : str
        Taxonomic rank used for hub grouping ('Order', 'Family', 'Class', ...).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Circle

    # ---- 1. Validate grouping column -------------------------------------
    if group_level not in sp_df.columns:
        for fallback in ["Family", "Order", "Class"]:
            if fallback in sp_df.columns:
                print(f"   ⚠️ '{group_level}' missing, falling back to '{fallback}'.")
                group_level = fallback
                break
        else:
            print("   ⚠️ No taxonomic grouping column found. Skipping radial plot.")
            return

    df = sp_df.sort_values([group_level, "Species"]).reset_index(drop=True)
    groups = df[group_level].unique().tolist()
    n_species, n_groups = len(df), len(groups)
    if n_species == 0 or n_groups == 0:
        print("   ⚠️ Empty species table. Skipping.")
        return

    # ---- 2. Pastel-vibrant palette (matches pathway-figure aesthetic) ----
    palette = [
        "#E57373", "#F06292", "#BA68C8", "#9575CD", "#7986CB",
        "#64B5F6", "#4FC3F7", "#4DD0E1", "#4DB6AC", "#81C784",
        "#AED581", "#DCE775", "#FFD54F", "#FFB74D", "#FF8A65",
        "#A1887F", "#90A4AE", "#F48FB1", "#CE93D8", "#80CBC4",
    ]
    group_colors = {g: palette[i % len(palette)] for i, g in enumerate(groups)}

    # ---- 3. Angular layout (equal gaps between clades) -------------------
    gap = 0.012 * 2 * np.pi
    usable = 2 * np.pi - gap * n_groups
    per_species = usable / n_species

    leaf_angles = np.zeros(n_species)
    group_spans = {}
    cur = np.pi / 2   # start at top, sweep clockwise
    idx = 0
    for g in groups:
        n_g = int((df[group_level] == g).sum())
        gs_start = cur
        for _ in range(n_g):
            leaf_angles[idx] = cur
            cur -= per_species
            idx += 1
        group_spans[g] = (gs_start, cur + per_species)
        cur -= gap

    # ---- 4. Geometry: leaf and hub positions -----------------------------
    R_leaf, R_label = 1.00, 1.04
    R_hub_inner, R_hub_outer = 0.32, 0.55

    sizes = {g: int((df[group_level] == g).sum()) for g in groups}
    max_g, min_g = max(sizes.values()), min(sizes.values())

    hub_pos = {}
    for i, g in enumerate(groups):
        gs, ge = group_spans[g]
        ang = (gs + ge) / 2
        # Larger clades nearer the center; small zigzag breaks symmetry.
        sf = 0.5 if max_g == min_g else (sizes[g] - min_g) / (max_g - min_g)
        R_hub = R_hub_outer - sf * (R_hub_outer - R_hub_inner) + 0.035 * (i % 2)
        hub_pos[g] = (R_hub * np.cos(ang), R_hub * np.sin(ang), ang, R_hub)

    # ---- 5. Plot ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(16, 16), facecolor="white")

    # 5a. Spokes (background)
    for i, row in df.iterrows():
        g = row[group_level]
        hx, hy, _, _ = hub_pos[g]
        a = leaf_angles[i]
        lx, ly = R_leaf * np.cos(a), R_leaf * np.sin(a)
        ax.plot([hx, lx], [hy, ly],
                color=group_colors[g], alpha=0.45, linewidth=0.7, zorder=1)

    # 5b. Leaf dots (size encodes prediction error if available)
    use_error = "mean_abs_error" in df.columns
    max_err = float(df["mean_abs_error"].max()) if use_error else 1.0
    max_err = max_err if max_err > 0 else 1.0

    for i, row in df.iterrows():
        a = leaf_angles[i]
        lx, ly = R_leaf * np.cos(a), R_leaf * np.sin(a)
        s = 25 + 70 * (row["mean_abs_error"] / max_err) if use_error else 35
        ax.scatter(lx, ly, s=s, color=group_colors[row[group_level]],
                   edgecolor="white", linewidth=0.8, zorder=2)

    # 5c. Hub bubbles (size ∝ √(# species)) + clade label
    for g in groups:
        hx, hy, _, _ = hub_pos[g]
        r = 0.035 + 0.024 * np.sqrt(sizes[g])
        ax.add_patch(Circle((hx, hy), r, color=group_colors[g],
                            alpha=0.78, ec="white", lw=1.5, zorder=3))
        ax.text(hx, hy, g, ha="center", va="center",
                fontsize=10, fontweight="bold", color="#222", zorder=4)

    # 5d. Tangential species labels
    for i, row in df.iterrows():
        a = leaf_angles[i]
        lx, ly = R_label * np.cos(a), R_label * np.sin(a)
        deg = np.degrees(a)
        if -90 < deg <= 90:
            ha, rot = "left", deg
        else:
            ha, rot = "right", deg + 180
        ax.text(lx, ly, str(row["Species"]).replace("_", " "), rotation=rot,
                ha=ha, va="center", fontsize=8.5,
                color=group_colors[row[group_level]],
                fontweight="bold", fontstyle="italic")

    # 5e. Titles
    ax.text(0, 1.30,
            f"Phylogenetic Performance Landscape ({target_name.upper()})",
            ha="center", va="center", fontsize=18, fontweight="bold")
    ax.text(0, -1.15,
            f"{n_species} species identified across {n_groups} {group_level.lower()}s",
            ha="center", va="center", fontsize=14, fontweight="bold", color="#222")

    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "radial_bubble_phylogenetic_plot.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"   ✅ Radial bubble plot saved: {out_path}")


# =========================================================
# ETE3 GBIF Taxonomic Tree Helper
# =========================================================
def render_gbif_taxonomic_tree(sp_df, target_name, output_dir):
    """
    Constructs a taxonomic tree from dataframe columns (GBIF) and
    overlays model error metrics on the leaves.
    """
    try:
        from ete3 import Tree, TreeStyle, NodeStyle, TextFace, CircleFace, RectFace
    except ImportError:
        raise ImportError("ETE3 is not installed. Please install 'ete3' and 'PyQt5'.")

    # Force PyQt5 to run headlessly
    import os
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    # 1. Build topology programmatically
    tree = Tree()
    tree.name = "Root"
    
    # Store reference to leaf nodes for fast mapping later
    valid_species_map = {}  # tree_node -> species row

    levels = ["Class", "Order", "Family", "Genus", "Species"]
    valid_levels = [l for l in levels if l in sp_df.columns]

    # Map node path to Tree node object
    node_map = {"Root": tree}

    for _, row in sp_df.iterrows():
        parent_path = "Root"
        parent_node = tree
        
        for level in valid_levels:
            val = row[level]
            if pd.isna(val) or val == "Unknown":
                continue
                
            current_path = f"{parent_path}::{val}"
            
            if current_path not in node_map:
                new_node = parent_node.add_child(name=str(val))
                node_map[current_path] = new_node
                
            parent_node = node_map[current_path]
            parent_path = current_path
            
            # If we just added/found the species node, store its data row
            if level == "Species":
                valid_species_map[parent_node] = row

    if not tree.children:
        raise ValueError("No valid taxonomy found in dataframe.")

    # Calculate global max error to normalize circle sizes
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    
    max_err = sp_df["mean_abs_error"].max()
    if max_err == 0:
        max_err = 1.0
        
    max_true = sp_df["mean_true"].max()
    if max_true == 0:
        max_true = 1.0

    cmap_true = cm.get_cmap("Blues")
    cmap_err = cm.get_cmap("Greens")
    norm_true = mcolors.Normalize(vmin=0, vmax=max_true)
    norm_err = mcolors.Normalize(vmin=0, vmax=max_err)

    # 3. Layout function for custom faces
    def layout_fn(node):
        if node.is_leaf():
            if node in valid_species_map:
                row = valid_species_map[node]
                err_val = row["mean_abs_error"]
                true_val = row["mean_true"]
                
                name_face = TextFace(f" {node.name.replace('_', ' ')} ", fsize=12, fgcolor="black", fstyle="italic")
                node.add_face(name_face, column=0, position="branch-right")

                true_text = f"True: {true_val:.3f}   |"
                true_face = TextFace(true_text, fsize=10, fgcolor="#666666")
                true_face.margin_left = 15
                node.add_face(true_face, column=1, position="aligned")

                err_norm_val = norm_err(err_val)
                color_val = max(0.2, err_norm_val) 
                err_color = mcolors.to_hex(cmap_err(color_val))
                
                radius = 5 + (err_norm_val * 6)
                err_circle = CircleFace(radius=radius, color=err_color, style="circle")
                err_circle.margin_left = 10
                node.add_face(err_circle, column=2, position="aligned")

                err_text = f"Err: {err_val:.3f}"
                err_face = TextFace(err_text, fsize=10, fgcolor="#2E7D32", fstyle="bold")
                err_face.margin_left = 6
                node.add_face(err_face, column=3, position="aligned")
        else:
            if node.name != "Root":
                name_face = TextFace(node.name, fsize=11, fgcolor="#666666", fstyle="italic")
                node.add_face(name_face, column=0, position="branch-top")

    # 4. Apply styles
    from ete3 import NodeStyle
    base_nstyle = NodeStyle()
    base_nstyle["hz_line_width"] = 2
    base_nstyle["vt_line_width"] = 2
    for n in tree.traverse():
        n.set_style(base_nstyle)

    ts = TreeStyle()
    ts.layout_fn = layout_fn
    ts.show_leaf_name = False
    ts.show_scale = False
    ts.scale = 120 # branch length scale
    ts.margin_left = 40
    ts.margin_right = 40
    ts.margin_top = 30
    ts.margin_bottom = 30
    
    # 5. Render
    out_pdf = os.path.join(output_dir, "gbif_taxonomic_tree.pdf")
    out_png = os.path.join(output_dir, "gbif_taxonomic_tree.png")
    
    tree.render(out_png, w=200, units="mm", dpi=600, tree_style=ts)
    tree.render(out_pdf, w=200, units="mm", tree_style=ts)
    
    # 6. Circular Clade Tree (Aesthetic version matching user request)
    try:
        render_circular_clade_tree(tree, sp_df, target_name, output_dir, valid_species_map)
    except Exception as e:
        print(f"      [Warning] Circular clade tree failed: {e}")


def render_circular_clade_tree(tree, sp_df, target_name, output_dir, valid_species_map):
    """
    Renders a circular version of the tree with distinctly colored clades
    and large group labels, directly from GBIF groups.
    """
    from ete3 import TreeStyle, NodeStyle, TextFace, CircleFace, RectFace
    
    palette = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEEAD", 
        "#D4A5A5", "#9B59B6", "#3498DB", "#E67E22", "#2ECC71"
    ]
    
    orders = [o for o in sp_df["Order"].unique() if not pd.isna(o)]
    order_colors = {order: palette[i % len(palette)] for i, order in enumerate(orders)}
    
    for node in tree.traverse():
        nstyle = NodeStyle()
        nstyle["size"] = 0
        nstyle["hz_line_width"] = 2
        nstyle["vt_line_width"] = 2
        
        if node.is_leaf() and node in valid_species_map:
            order = valid_species_map[node]["Order"]
            if order in order_colors:
                nstyle["fgcolor"] = order_colors[order]
                nstyle["hz_line_color"] = order_colors[order]
                nstyle["vt_line_color"] = order_colors[order]
        
        node.set_style(nstyle)

    # 2. Add Clade Labels
    classes = [c for c in sp_df["Class"].unique() if not pd.isna(c)]
    for cls in classes:
        # Find the node corresponding to the Class
        cls_node = None
        for n in tree.traverse():
            if n.name == cls:
                cls_node = n
                break
                
        if cls_node:
            label_text = "Monocots" if cls == "Liliopsida" else "Dicots"
            color = "#2E7D32" if cls == "Liliopsida" else "#1565C0"
            face = TextFace(label_text, fsize=24, fgcolor=color, fstyle="bold")
            face.margin_top = 10
            face.margin_bottom = 10
            cls_node.add_face(face, column=0, position="branch-top")

    ts = TreeStyle()
    ts.mode = "c"
    ts.arc_span = 340 
    ts.arc_start = 10
    ts.show_leaf_name = False
    ts.show_branch_length = False
    ts.show_branch_support = False
    ts.force_topology = True 
    
    def circular_layout(node):
        if node.is_leaf() and node in valid_species_map:
            name_face = TextFace(f" {node.name.replace('_', ' ')} ", fsize=16, fstyle="italic", fgcolor="#333333")
            node.add_face(name_face, column=10, position="aligned")
            
            row = valid_species_map[node]
            order = row["Order"]
            if order in order_colors:
                color = order_colors[order]
                rect = RectFace(width=20, height=40, fgcolor=color, bgcolor=color)
                node.add_face(rect, column=1, position="aligned")
                
            err = row["mean_abs_error"]
            err_max = sp_df["mean_abs_error"].max() or 1.0
            err_norm = min(1.0, err / err_max)
            r, g = int(255 * err_norm), int(255 * (1 - err_norm))
            b_color = f"#{r:02x}{g:02x}44"
            
            radius = 8 + (err_norm * 12)
            bubble = CircleFace(radius=radius, color=b_color, style="sphere")
            node.add_face(bubble, column=2, position="aligned")

    ts.layout_fn = circular_layout
    
    title_face = TextFace(f"Phylogenetic Performance Landscape ({target_name.upper()})", fsize=28, fstyle="bold")
    title_face.margin_bottom = 40
    ts.title.add_face(title_face, column=0)
    
    legend_title = TextFace("Taxonomic Orders", fsize=18, fstyle="bold")
    legend_title.margin_top = 20
    ts.legend.add_face(legend_title, column=0)
    for order, color in order_colors.items():
        color_box = RectFace(width=20, height=20, fgcolor=color, bgcolor=color)
        order_text = TextFace(f" {order}", fsize=14)
        ts.legend.add_face(color_box, column=0)
        ts.legend.add_face(order_text, column=1)

    ts.margin_left = 100
    ts.margin_right = 100
    ts.margin_top = 100
    ts.margin_bottom = 100

    out_png = os.path.join(output_dir, "gbif_circular_taxonomic_tree.png")
    tree.render(out_png, w=200, units="mm", dpi=600, tree_style=ts)


# =========================================================
#  Clade-Level Performance
# =========================================================
def clade_performance(
    model, test_X, test_y, metadata, scaler_stats, args, output_dir,
    n_bootstrap=1000
):
    """Analyze model performance grouped by Family and Order."""
    print("\n" + "=" * 70)
    print("📊 Clade-Level Performance Breakdown")
    print("=" * 70)

    device = next(model.parameters()).device
    test_loader = DataLoader(
        SorbFormerDataset(test_X, test_y, metadata, None, "test", scaler_stats),
        batch_size=32, shuffle=False, collate_fn=SorbFormer_collate_fn,
    )
    preds, _ = predict_all(model, test_loader, device)

    y_true = np.power(10.0, test_y) if args.target == "rcf" else np.array(test_y)
    y_pred = np.power(10.0, preds) if args.target == "rcf" else preds

    all_results = []

    for level in ["Family", "Order", "Class"]:
        if level not in test_X.columns:
            continue

        print(f"   Analyzing by {level}...")
        groups = test_X[level].values

        for group_name in np.unique(groups):
            mask = groups == group_name
            n = mask.sum()
            if n < 3:
                continue

            gt = y_true[mask]
            gp = y_pred[mask]

            r2 = r2_score(gt, gp) if n >= 2 else float("nan")
            rmse = np.sqrt(mean_squared_error(gt, gp))
            mae = mean_absolute_error(gt, gp)

            # Bootstrap CI for R²
            boot_r2 = []
            for _ in range(n_bootstrap):
                idx = np.random.choice(n, n, replace=True)
                if len(np.unique(gt[idx])) < 2:
                    continue
                boot_r2.append(r2_score(gt[idx], gp[idx]))

            ci_lo = np.percentile(boot_r2, 5) if boot_r2 else float("nan")
            ci_hi = np.percentile(boot_r2, 95) if boot_r2 else float("nan")

            all_results.append({
                "taxonomic_level": level,
                "group": group_name,
                "n_samples": int(n),
                "r2": float(r2),
                "r2_ci_lo": float(ci_lo),
                "r2_ci_hi": float(ci_hi),
                "rmse": float(rmse),
                "mae": float(mae),
            })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(
        os.path.join(output_dir, "clade_performance.csv"), index=False
    )

    # Plot per taxonomic level
    for level in results_df["taxonomic_level"].unique():
        level_df = results_df[results_df["taxonomic_level"] == level].copy()
        level_df = level_df.sort_values("r2", ascending=True)

        fig, ax = plt.subplots(figsize=(10, max(4, len(level_df) * 0.5)))
        y_pos = range(len(level_df))

        bars = ax.barh(
            y_pos, level_df["r2"],
            xerr=[
                level_df["r2"] - level_df["r2_ci_lo"],
                level_df["r2_ci_hi"] - level_df["r2"],
            ],
            color="#2196F3", edgecolor="white", capsize=4,
        )

        labels = [
            f"{row['group']} (n={row['n_samples']})"
            for _, row in level_df.iterrows()
        ]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("R² ± 90% Bootstrap CI")
        ax.set_title(f"Model Performance by {level}")
        ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
        plt.tight_layout()
        fig.savefig(
            os.path.join(output_dir, f"clade_performance_{level.lower()}.png"),
            dpi=300, bbox_inches="tight",
        )
        plt.close(fig)

    print(f"   ✅ Performance computed for {len(results_df)} clades.")
    return results_df



# =========================================================
#  Taxonomic Feature Importance
# =========================================================
def taxonomic_feature_importance(
    model, test_X, test_y, metadata, scaler_stats, args, output_dir, n_repeats=50
):
    """Permutation importance restricted to taxonomic categorical features."""
    print("\n" + "=" * 70)
    print("🏷️ Taxonomic Feature Importance")
    print("=" * 70)

    device = next(model.parameters()).device
    taxonomic_cols = ["Class", "Order", "Family", "Genus", "Species"]
    available_tax = [c for c in taxonomic_cols if c in test_X.columns]

    if not available_tax:
        print("   ⚠️ No taxonomic columns found. Skipping.")
        return None

    # Baseline predictions
    base_ds = SorbFormerDataset(
        test_X, test_y, metadata,
        enumerator=None, split="test", scaler_stats=scaler_stats,
    )
    base_loader = DataLoader(
        base_ds, batch_size=32, shuffle=False,
        collate_fn=SorbFormer_collate_fn, num_workers=4,
    )
    base_preds, _ = predict_all(model, base_loader, device)
    base_metrics = calculate_metrics(test_y, base_preds, args.target)
    base_r2 = base_metrics["r2"]

    print(f"   Baseline R²: {base_r2:.4f}")

    importances = []
    for feat in available_tax:
        r2_drops = []
        for _ in range(n_repeats):
            shuffled_X = test_X.copy()
            shuffled_X[feat] = np.random.permutation(shuffled_X[feat].values)
            shuf_ds = SorbFormerDataset(
                shuffled_X, test_y, metadata,
                enumerator=None, split="test", scaler_stats=scaler_stats,
            )
            shuf_loader = DataLoader(
                shuf_ds, batch_size=32, shuffle=False,
                collate_fn=SorbFormer_collate_fn, num_workers=4,
            )
            shuf_preds, _ = predict_all(model, shuf_loader, device)
            shuf_metrics = calculate_metrics(test_y, shuf_preds, args.target)
            r2_drops.append(base_r2 - shuf_metrics["r2"])

        mean_drop = np.mean(r2_drops)
        std_drop = np.std(r2_drops)
        from scipy.stats import ttest_1samp
        _, p_val = ttest_1samp(r2_drops, popmean=0.0, alternative="greater")

        importances.append({
            "taxonomic_level": feat,
            "importance_mean": float(mean_drop),
            "importance_std": float(std_drop),
            "p_value": float(p_val),
            "significant": p_val < 0.05,
        })
        sig = "*" if p_val < 0.05 else "ns"
        print(f"   {feat}: ΔR² = {mean_drop:.4f} ± {std_drop:.4f} ({sig})")

    imp_df = pd.DataFrame(importances).sort_values(
        "importance_mean", ascending=False
    )
    imp_df.to_csv(
        os.path.join(output_dir, "taxonomic_importance.csv"), index=False
    )

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4CAF50" if s else "#9E9E9E" for s in imp_df["significant"]]
    ax.barh(
        range(len(imp_df)), imp_df["importance_mean"],
        xerr=imp_df["importance_std"],
        color=colors, edgecolor="white", capsize=4,
    )
    for j, (_, row) in enumerate(imp_df.iterrows()):
        if row["significant"]:
            marker = (
                "***" if row["p_value"] < 0.001
                else "**" if row["p_value"] < 0.01
                else "*"
            )
            ax.text(
                row["importance_mean"] + row["importance_std"] + 0.002,
                j, marker, va="center", fontsize=10, fontweight="bold",
            )

    ax.set_yticks(range(len(imp_df)))
    ax.set_yticklabels(imp_df["taxonomic_level"], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Δ R² (Importance)")
    ax.set_title("Taxonomic Feature Importance (Permutation)")
    plt.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "taxonomic_importance.png"),
        dpi=300, bbox_inches="tight",
    )
    plt.close(fig)

    return imp_df
