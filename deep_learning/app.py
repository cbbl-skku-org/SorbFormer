import os
import sys
import json
import yaml
import time
import tempfile
import torch
import pandas as pd
import numpy as np
import gradio as gr
from pathlib import Path
import threading
import gc

# Add current directory to path
sys.path.append(str(Path(__file__).resolve().parent))

from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, rdMolDescriptors, Crippen
import selfies as sf

from src.model import PlantUptakeModel
from src.molecules.feature_extractor import MolecularFeatureExtractor

from huggingface_hub import snapshot_download

# =========================================================
# 1. Global Model & Data Loader
# =========================================================
APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IDLE_TIMEOUT = 1 * 60
CHECK_INTERVAL = 30
MODEL_LOCK = threading.RLock()
LAST_USED = time.time()
MODELS_ON_GPU = DEVICE == "cuda"

print(f"🚀 Initializing SorbFormer Gradio App on device: {DEVICE}")

if not (MODELS_DIR / "rcf" / "model.ckpt").exists() or not (MODELS_DIR / "tscf" / "model.ckpt").exists():
    print("📥 Models not found locally. Downloading from Hugging Face Hub (cbbl-skku-org/SorbFormer)...")
    snapshot_download(repo_id="cbbl-skku-org/SorbFormer", local_dir=str(MODELS_DIR))

with open(MODELS_DIR / "rcf" / "metadata_rcf.json", "r") as f:
    RCF_META = json.load(f)
with open(MODELS_DIR / "rcf" / "scaler_stats.json", "r") as f:
    RCF_STATS = json.load(f)
with open(MODELS_DIR / "rcf" / "hparams.yaml", "r") as f:
    RCF_HPARAMS = yaml.safe_load(f)

# Load TSCF metadata and stats
with open(MODELS_DIR / "tscf" / "metadata_tscf.json", "r") as f:
    TSCF_META = json.load(f)
with open(MODELS_DIR / "tscf" / "scaler_stats.json", "r") as f:
    TSCF_STATS = json.load(f)
with open(MODELS_DIR / "tscf" / "hparams.yaml", "r") as f:
    TSCF_HPARAMS = yaml.safe_load(f)

# Shared Molecular Feature Extractor (BioT5+ base)
print("📥 Loading shared BioT5+ feature extractor...")
EXTRACTOR = MolecularFeatureExtractor(model_name="biot5-plus-base", device=DEVICE)

# Load RCF Model (BioT5+ Cross-Attention)
print("📥 Loading RCF Model checkpoint...")
rcf_args = sys.modules["argparse"].Namespace(**RCF_HPARAMS)
rcf_args.metadata_file = str(MODELS_DIR / "rcf" / "metadata_rcf.json")
RCF_MODEL = PlantUptakeModel.load_from_checkpoint(
    str(MODELS_DIR / "rcf" / "model.ckpt"),
    args=rcf_args,
    molecular_extractor=EXTRACTOR,
)
RCF_MODEL.to(DEVICE)
RCF_MODEL.eval()

# Load TSCF Model (BioT5+ Concat)
print("📥 Loading TSCF Model checkpoint...")
tscf_args = sys.modules["argparse"].Namespace(**TSCF_HPARAMS)
tscf_args.metadata_file = str(MODELS_DIR / "tscf" / "metadata_tscf.json")
TSCF_MODEL = PlantUptakeModel.load_from_checkpoint(
    str(MODELS_DIR / "tscf" / "model.ckpt"),
    args=tscf_args,
    molecular_extractor=EXTRACTOR,
)
TSCF_MODEL.to(DEVICE)
TSCF_MODEL.eval()

print("✅ Models loaded successfully!")

def move_models_to_gpu():
    global MODELS_ON_GPU, LAST_USED

    if DEVICE != "cuda":
        return

    with MODEL_LOCK:
        if not MODELS_ON_GPU:
            print("🔥 Moving SorbFormer models back to GPU...")

            EXTRACTOR.model.to("cuda")
            RCF_MODEL.to("cuda")
            TSCF_MODEL.to("cuda")

            RCF_MODEL.eval()
            TSCF_MODEL.eval()

            MODELS_ON_GPU = True

        LAST_USED = time.time()


def mark_model_used():
    global LAST_USED

    with MODEL_LOCK:
        LAST_USED = time.time()


def offload_models_to_cpu():
    global MODELS_ON_GPU

    if DEVICE != "cuda":
        return

    with MODEL_LOCK:
        if MODELS_ON_GPU:
            print("💤 SorbFormer idle. Offloading models to CPU...")

            # Main prediction models
            RCF_MODEL.to("cpu")
            TSCF_MODEL.to("cpu")

            # Shared BioT5 extractor
            EXTRACTOR.model.to("cpu")

            # Clean remaining temporary tensors
            gc.collect()

            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

            MODELS_ON_GPU = False

            print(
                f"✅ CUDA allocated: "
                f"{torch.cuda.memory_allocated() / 1024**2:.1f} MB"
            )
            print(
                f"✅ CUDA reserved: "
                f"{torch.cuda.memory_reserved() / 1024**2:.1f} MB"
            )

def idle_monitor():
    while True:
        time.sleep(CHECK_INTERVAL)

        with MODEL_LOCK:
            idle_time = time.time() - LAST_USED

        if MODELS_ON_GPU and idle_time >= IDLE_TIMEOUT:
            offload_models_to_cpu()


if DEVICE == "cuda":
    threading.Thread(
        target=idle_monitor,
        daemon=True,
    ).start()

# Taxonomy helper dictionary (Complete Species -> Taxonomy Defaults)
TAXONOMY_MAP = {
    "Allium cepa": {"Class": "Liliopsida", "Order": "Asparagales", "Family": "Amaryllidaceae", "Genus": "Allium", "group": "grass", "root": "fibrous"},
    "Amaranthus caudatus": {"Class": "Magnoliopsida", "Order": "Caryophyllales", "Family": "Amaranthaceae", "Genus": "Amaranthus", "group": "forb", "root": "taproot"},
    "Beta vulgaris subsp. vulgaris": {"Class": "Magnoliopsida", "Order": "Caryophyllales", "Family": "Amaranthaceae", "Genus": "Beta", "group": "forb", "root": "taproot"},
    "Brassica oleracea": {"Class": "Magnoliopsida", "Order": "Brassicales", "Family": "Brassicaceae", "Genus": "Brassica", "group": "forb", "root": "taproot"},
    "Brassica rapa": {"Class": "Magnoliopsida", "Order": "Brassicales", "Family": "Brassicaceae", "Genus": "Brassica", "group": "forb", "root": "taproot"},
    "Capsicum annuum": {"Class": "Magnoliopsida", "Order": "Solanales", "Family": "Solanaceae", "Genus": "Capsicum", "group": "forb", "root": "taproot"},
    "Capsicum annuum L.": {"Class": "Magnoliopsida", "Order": "Solanales", "Family": "Solanaceae", "Genus": "Capsicum", "group": "forb", "root": "taproot"},
    "Cucumis sativus": {"Class": "Magnoliopsida", "Order": "Cucurbitales", "Family": "Cucurbitaceae", "Genus": "Cucumis", "group": "forb", "root": "taproot"},
    "Cucurbita maxima": {"Class": "Magnoliopsida", "Order": "Cucurbitales", "Family": "Cucurbitaceae", "Genus": "Cucurbita", "group": "forb", "root": "taproot"},
    "Cucurbita pepo": {"Class": "Magnoliopsida", "Order": "Cucurbitales", "Family": "Cucurbitaceae", "Genus": "Cucurbita", "group": "forb", "root": "taproot"},
    "Daucus carota": {"Class": "Magnoliopsida", "Order": "Apiales", "Family": "Apiaceae", "Genus": "Daucus", "group": "forb", "root": "taproot"},
    "Elaeagnus angustifolia": {"Class": "Magnoliopsida", "Order": "Rosales", "Family": "Elaeagnaceae", "Genus": "Elaeagnus", "group": "forb", "root": "taproot"},
    "Elaeagnus umbellata": {"Class": "Magnoliopsida", "Order": "Rosales", "Family": "Elaeagnaceae", "Genus": "Elaeagnus", "group": "forb", "root": "taproot"},
    "Festuca baffinensis": {"Class": "Liliopsida", "Order": "Poales", "Family": "Poaceae", "Genus": "Festuca", "group": "grass", "root": "fibrous"},
    "Glycine max": {"Class": "Magnoliopsida", "Order": "Fabales", "Family": "Fabaceae", "Genus": "Glycine", "group": "legume", "root": "taproot"},
    "Hordeum vulgare": {"Class": "Liliopsida", "Order": "Poales", "Family": "Poaceae", "Genus": "Hordeum", "group": "grass", "root": "fibrous"},
    "Lactuca sativa": {"Class": "Magnoliopsida", "Order": "Asterales", "Family": "Asteraceae", "Genus": "Lactuca", "group": "forb", "root": "taproot"},
    "Lolium multiflorum": {"Class": "Liliopsida", "Order": "Poales", "Family": "Poaceae", "Genus": "Lolium", "group": "grass", "root": "fibrous"},
    "Lonicera xylosteum": {"Class": "Magnoliopsida", "Order": "Dipsacales", "Family": "Caprifoliaceae", "Genus": "Lonicera", "group": "shrub", "root": "taproot"},
    "Phaseolus vulgaris": {"Class": "Magnoliopsida", "Order": "Fabales", "Family": "Fabaceae", "Genus": "Phaseolus", "group": "legume", "root": "taproot"},
    "Phragmites australis": {"Class": "Liliopsida", "Order": "Poales", "Family": "Poaceae", "Genus": "Phragmites", "group": "grass", "root": "fibrous"},
    "Populus X tomentosa": {"Class": "Magnoliopsida", "Order": "Malpighiales", "Family": "Salicaceae", "Genus": "Populus", "group": "tree", "root": "taproot"},
    "Populus tomentosa": {"Class": "Magnoliopsida", "Order": "Malpighiales", "Family": "Salicaceae", "Genus": "Populus", "group": "tree", "root": "taproot"},
    "Raphanus sativus": {"Class": "Magnoliopsida", "Order": "Brassicales", "Family": "Brassicaceae", "Genus": "Raphanus", "group": "forb", "root": "taproot"},
    "Solanum lycopersicum": {"Class": "Magnoliopsida", "Order": "Solanales", "Family": "Solanaceae", "Genus": "Solanum", "group": "forb", "root": "taproot"},
    "Solanum melongena": {"Class": "Magnoliopsida", "Order": "Solanales", "Family": "Solanaceae", "Genus": "Solanum", "group": "forb", "root": "taproot"},
    "Spinacia oleracea": {"Class": "Magnoliopsida", "Order": "Caryophyllales", "Family": "Amaranthaceae", "Genus": "Spinacia", "group": "forb", "root": "taproot"},
    "Sporobolus alterniflorus": {"Class": "Liliopsida", "Order": "Poales", "Family": "Poaceae", "Genus": "Sporobolus", "group": "grass", "root": "fibrous"},
    "Tamarix ramosissima": {"Class": "Magnoliopsida", "Order": "Caryophyllales", "Family": "Tamaricaceae", "Genus": "Tamarix", "group": "shrub", "root": "taproot"},
    "Trifolium repens": {"Class": "Magnoliopsida", "Order": "Fabales", "Family": "Fabaceae", "Genus": "Trifolium", "group": "legume", "root": "taproot"},
    "Triticum aestivum": {"Class": "Liliopsida", "Order": "Poales", "Family": "Poaceae", "Genus": "Triticum", "group": "grass", "root": "fibrous"},
    "Zea mays": {"Class": "Liliopsida", "Order": "Poales", "Family": "Poaceae", "Genus": "Zea", "group": "grass", "root": "fibrous"},
}

ALL_SPECIES = sorted(list(TAXONOMY_MAP.keys()))

# Combined Exposure Medium Options & Plant Condition Options
EXPOSURE_OPTIONS = sorted(list(set(RCF_META["categorical_maps"]["exposure_medium"].keys()).union(set(TSCF_META["categorical_maps"]["exposure_medium"].keys()))))
CONDITION_OPTIONS = sorted(list(set(RCF_META["categorical_maps"]["plant_condition"].keys()).union(set(TSCF_META["categorical_maps"]["plant_condition"].keys()))))


# =========================================================
# 2. Descriptor & Image Utilities
# =========================================================
def compute_descriptors_from_smiles(smiles_str: str):
    """Compute RDKit chemical descriptors from a SMILES string."""
    if not smiles_str or smiles_str.strip() == "":
        return None
    mol = Chem.MolFromSmiles(smiles_str.strip())
    if mol is None:
        return None
    return {
        "logkow": round(float(Descriptors.MolLogP(mol)), 3),
        "molecular_weight": round(float(Descriptors.MolWt(mol)), 2),
        "hbd": int(Descriptors.NumHDonors(mol)),
        "hba": int(Descriptors.NumHAcceptors(mol)),
        "rotatable_bonds": int(Descriptors.NumRotatableBonds(mol)),
        "psa": round(float(Descriptors.TPSA(mol)), 2),
        "vp": 0.001,
        "hl": 1e-6,
        "ws": 100.0,
        "aromatic_rings": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "fraction_csp3": round(float(Descriptors.FractionCSP3(mol)), 3),
        "num_rings": int(rdMolDescriptors.CalcNumRings(mol)),
        "num_heavy_atoms": int(mol.GetNumHeavyAtoms()),
        "num_heteroatoms": int(rdMolDescriptors.CalcNumHeteroatoms(mol)),
        "num_saturated_rings": int(rdMolDescriptors.CalcNumSaturatedRings(mol)),
        "num_aliphatic_rings": int(rdMolDescriptors.CalcNumAliphaticRings(mol)),
        "labute_asa": round(float(rdMolDescriptors.CalcLabuteASA(mol)), 2),
        "molar_refractivity": round(float(Crippen.MolMR(mol)), 2),
    }


def draw_molecule_image(smiles_str: str):
    """Generate PIL Image for 2D molecular structure."""
    if not smiles_str or smiles_str.strip() == "":
        return None
    mol = Chem.MolFromSmiles(smiles_str.strip())
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=(320, 320))


def update_structure_and_descriptors(smiles_str: str):
    """Generate both 2D structure image and calculated RDKit descriptors table."""
    if not smiles_str or smiles_str.strip() == "":
        return None, pd.DataFrame()
    mol_img = draw_molecule_image(smiles_str)
    desc = compute_descriptors_from_smiles(smiles_str)
    if desc is None:
        return None, pd.DataFrame()
    desc_df = pd.DataFrame([desc]).T.reset_index()
    desc_df.columns = ["Physicochemical Property", "Calculated Value"]
    return mol_img, desc_df


def update_structure_and_descriptors_delayed(smiles_str: str):
    """Wait 1 second before generating structure preview & RDKit descriptors table to debounce user typing."""
    time.sleep(1.0)
    return update_structure_and_descriptors(smiles_str)


# =========================================================
# 3. Model Preprocessing & Prediction Pipeline
# =========================================================
def prepare_model_batch(row_dict: dict, metadata: dict, scaler_stats: dict):
    """Format single sample dictionary into PyTorch tensor inputs matching target model metadata."""
    smiles_list = [row_dict.get("SMILES", "C")]

    # Categorical features
    cat_feats = {}
    for col in metadata["categorical_columns"]:
        val = row_dict.get(col, None)
        cat_idx = metadata["categorical_maps"][col].get(val, 0)  # 0 = <UNK>
        cat_feats[f"cat_{col}"] = torch.tensor([cat_idx], dtype=torch.long, device=DEVICE)

    # Numerical features
    num_feats = {}
    for col in metadata["numerical_columns"]:
        val = float(row_dict.get(col, 0.0))
        if scaler_stats and col in scaler_stats:
            stats = scaler_stats[col]
            val = (val - stats["mean"]) / (stats["std"] + 1e-8)
        num_feats[f"num_{col}"] = torch.tensor([val], dtype=torch.float, device=DEVICE)

    return smiles_list, cat_feats, num_feats


@torch.no_grad()
def predict_single(
    smiles: str,
    species: str,
    class_val: str,
    order_val: str,
    family_val: str,
    genus_val: str,
    plant_group: str,
    root_system: str,
    exposure_medium: str,
    plant_condition: str,
    t_dur: float,
    ph_val: float,
    co_val: float,
):
    """Run SorbFormer single sample prediction for both RCF and TSCF."""
    if not smiles or smiles.strip() == "":
        return (
            "<div class='error-card'>⚠️ Please provide a valid SMILES string.</div>",
            "<div class='error-card'>-</div>",
        )

    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return (
            f"<div class='error-card'>⚠️ Invalid SMILES string: <code>{smiles}</code></div>",
            "<div class='error-card'>-</div>",
        )

    # Calculate descriptors
    desc = compute_descriptors_from_smiles(smiles)

    # Build input dictionary
    input_dict = {
        "SMILES": smiles,
        "Class": class_val,
        "Order": order_val,
        "Family": family_val,
        "Genus": genus_val,
        "Species": species,
        "plant_functional_group": plant_group,
        "plant_root_system": root_system,
        "exposure_medium": exposure_medium,
        "plant_condition": plant_condition,
        "t_dur (days)": t_dur,
        "pH": ph_val,
        "Co (mg/L)": co_val,
        # Plant trait averages
        "plant_monocot": 1 if class_val == "Liliopsida" else 0,
        "plant_root_tissue_density": 0.1,
        "plant_root_lipid_content": 0.015,
        "plant_root_protein_content": 0.02,
        "plant_transpiration_rate": 1.5,
        "plant_stomatal_density": 100.0,
        "plant_specific_root_length": 50.0,
        "trait_LDMC": 0.15,
        "trait_rooting_depth": 0.5,
        "trait_SLA": 25.0,
        "trait_leaf_area": 10.0,
    }
    input_dict.update(desc)

    # Predict RCF (Root Concentration Factor)
    smiles_rcf, cat_rcf, num_rcf = prepare_model_batch(input_dict, RCF_META, RCF_STATS)
    pred_rcf_log10, _ = RCF_MODEL(smiles_rcf, cat_rcf, num_rcf)
    val_rcf_log10 = float(pred_rcf_log10.detach().cpu().numpy()[0])
    val_rcf_linear = 10 ** val_rcf_log10

    # Predict TSCF (Transpiration Stream Concentration Factor)
    smiles_tscf, cat_tscf, num_tscf = prepare_model_batch(input_dict, TSCF_META, TSCF_STATS)
    pred_tscf, _ = TSCF_MODEL(smiles_tscf, cat_tscf, num_tscf)
    val_tscf = float(pred_tscf.detach().cpu().numpy()[0])

    # HTML Card outputs
    rcf_card = f"""
    <div class="result-card rcf-card">
        <div class="card-title">🌱 Root Concentration Factor (RCF)</div>
        <div class="primary-metric">{val_rcf_linear:.3f} <span class="unit">L/kg (linear)</span></div>
        <div class="sub-metric">log₁₀(RCF): <strong>{val_rcf_log10:.3f}</strong></div>
    </div>
    """

    tscf_card = f"""
    <div class="result-card tscf-card">
        <div class="card-title">💧 Transpiration Stream Concentration Factor (TSCF)</div>
        <div class="primary-metric">{val_tscf:.3f} <span class="unit">(dimensionless)</span></div>
        <div class="sub-metric">Relative Sap Translocation Efficiency</div>
    </div>
    """

    return rcf_card, tscf_card


@torch.no_grad()
def predict_batch_csv(file_obj):
    """Run batch prediction on uploaded CSV file."""
    if file_obj is None:
        return "⚠️ Please upload a CSV file.", None

    df = pd.read_csv(file_obj.name)
    if "SMILES" not in df.columns:
        return "❌ Error: Input CSV must contain a 'SMILES' column.", None

    rcf_log10_list = []
    rcf_linear_list = []
    tscf_list = []

    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        smi = str(row_dict.get("SMILES", "C"))
        desc = compute_descriptors_from_smiles(smi)
        if desc:
            for k, v in desc.items():
                if k not in row_dict or pd.isna(row_dict[k]):
                    row_dict[k] = v

        # RCF
        try:
            s_rcf, c_rcf, n_rcf = prepare_model_batch(row_dict, RCF_META, RCF_STATS)
            p_rcf, _ = RCF_MODEL(s_rcf, c_rcf, n_rcf)
            val_log10 = float(p_rcf.detach().cpu().numpy()[0])
            rcf_log10_list.append(round(val_log10, 4))
            rcf_linear_list.append(round(10 ** val_log10, 4))
        except Exception:
            rcf_log10_list.append(np.nan)
            rcf_linear_list.append(np.nan)

        # TSCF
        try:
            s_tscf, c_tscf, n_tscf = prepare_model_batch(row_dict, TSCF_META, TSCF_STATS)
            p_tscf, _ = TSCF_MODEL(s_tscf, c_tscf, n_tscf)
            val_tscf = float(p_tscf.detach().cpu().numpy()[0])
            tscf_list.append(round(val_tscf, 4))
        except Exception:
            tscf_list.append(np.nan)

    res_df = df.copy()
    res_df["Predicted_RCF_log10"] = rcf_log10_list
    res_df["Predicted_RCF_linear"] = rcf_linear_list
    res_df["Predicted_TSCF"] = tscf_list

    # Save to temp file
    temp_dir = tempfile.mkdtemp()
    out_path = os.path.join(temp_dir, "sorbformer_predictions.csv")
    res_df.to_csv(out_path, index=False)

    return res_df, out_path


def update_species_taxonomy(species_name: str):
    """Instantly populate read-only Taxonomic Hierarchy fields when a user selects a plant species."""
    if species_name in TAXONOMY_MAP:
        info = TAXONOMY_MAP[species_name]
        return info["Class"], info["Order"], info["Family"], info["Genus"], info["group"], info["root"]
    return "Magnoliopsida", "Solanales", "Solanaceae", "Solanum", "forb", "taproot"


# =========================================================
# 4. Gradio Interface Construction
# =========================================================
CUSTOM_CSS = """
.container { max-width: 1200px; margin: 0 auto; }
.header-box { text-align: center; background: linear-gradient(135deg, #059669 0%, #0d9488 100%); color: white; padding: 25px; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
.header-box h1 { margin: 0; font-size: 2.2rem; font-weight: 700; color: #ffffff; }
.header-box p { margin-top: 8px; font-size: 1.05rem; opacity: 0.92; }
.result-card { padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); border: 1px solid #e5e7eb; }
.rcf-card { background-color: #f0fdf4; border-left: 5px solid #16a34a; }
.tscf-card { background-color: #f0f9ff; border-left: 5px solid #0284c7; }
.card-title { font-size: 1.1rem; font-weight: 600; color: #374151; margin-bottom: 8px; }
.primary-metric { font-size: 2.2rem; font-weight: 800; color: #111827; }
.primary-metric .unit { font-size: 1rem; font-weight: 400; color: #6b7280; }
.sub-metric { font-size: 0.95rem; color: #4b5563; margin-top: 4px; }
.error-card { padding: 15px; background-color: #fef2f2; color: #991b1b; border-radius: 8px; border: 1px solid #fecaca; font-weight: 500; }
"""

with gr.Blocks(title="SorbFormer", css=CUSTOM_CSS, theme=gr.themes.Soft(primary_hue="emerald", secondary_hue="indigo")) as demo:
    gr.HTML(
        """
        <div class="header-box">
            <h1>🌿 SorbFormer: Deep Multimodal Transformer for Plant Uptake</h1>
            <p>Predicting Root Concentration Factor (RCF) & Transpiration Stream Concentration Factor (TSCF) of Organic Contaminants</p>
            <p style="margin-top: 10px; font-size: 0.95rem;">Developed by <a href="https://balalab-skku.org/" target="_blank" style="color: #e0e7ff; text-decoration: underline; font-weight: 600;">Computational Biology and Bioinformatics Laboratory</a></p>
        </div>
        """
    )


    with gr.Tabs():
        # --- TAB 1: Single Prediction ---
        with gr.TabItem("🎯 Single Molecule Prediction"):
            with gr.Row():
                # Left Column: Chemical Input, 2D Structure Preview & RDKit Descriptors Panel
                with gr.Column(scale=1):
                    gr.Markdown("### 🧪 1. Chemical Structure & Properties")
                    smiles_input = gr.Textbox(
                        label="SMILES String",
                        placeholder="e.g., CC1=C(C(=O)N(N1C)C2=CC=CC=C2)N(C)CC3=CC=CC=C3",
                        value="CC1=C(C(=O)N(N1C)C2=CC=CC=C2)N(C)CC3=CC=CC=C3",
                    )
                    mol_image_output = gr.Image(label="2D Structure Preview (Auto-generated after 1s)", height=260)
                    desc_table_output = gr.DataFrame(label="Calculated RDKit Descriptors (Auto-generated after 1s)")

                    gr.Markdown("#### 💡 Preset Example Chemicals")
                    gr.Examples(
                        examples=[
                            ["CC1=C(C(=O)N(N1C)C2=CC=CC=C2)N(C)CC3=CC=CC=C3"],  # Aminopyrine
                            ["CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"],  # Ibuprofen
                            ["CC(=O)NC1=CC=C(C=C1)O"],  # Paracetamol
                            ["ClC1=CC=C(C=C1)C(C2=CC=C(Cl)C=C2)C(Cl)(Cl)Cl"],  # DDT
                        ],
                        inputs=[smiles_input],
                    )

                # Middle Column: Plant Taxonomy & Trait Parameters
                with gr.Column(scale=1):
                    gr.Markdown("### 🌱 2. Plant Taxonomy & Environmental Parameters")

                    species_dropdown = gr.Dropdown(
                        choices=ALL_SPECIES,
                        value="Solanum lycopersicum",
                        label="Plant Species (Selecting auto-populates read-only Taxonomy)",
                    )

                    with gr.Accordion("Taxonomic Hierarchy (Auto-Generated & Read-Only)", open=False):
                        class_input = gr.Textbox(value="Magnoliopsida", label="Class", interactive=False)
                        order_input = gr.Textbox(value="Solanales", label="Order", interactive=False)
                        family_input = gr.Textbox(value="Solanaceae", label="Family", interactive=False)
                        genus_input = gr.Textbox(value="Solanum", label="Genus", interactive=False)


                    with gr.Accordion("Biological & Exposure Conditions", open=True):
                        plant_group_input = gr.Textbox(value="forb", label="Plant Functional Group", interactive=False)
                        root_system_input = gr.Textbox(value="taproot", label="Root System", interactive=False)
                        exposure_dropdown = gr.Dropdown(
                            choices=EXPOSURE_OPTIONS,
                            value="hydroponic",
                            label="Exposure Medium",
                        )
                        condition_dropdown = gr.Dropdown(
                            choices=CONDITION_OPTIONS,
                            value="intact",
                            label="Plant Condition",
                        )

                    with gr.Row():
                        dur_input = gr.Number(label="Exposure Duration (days)", value=7.0)
                        ph_input = gr.Number(label="pH", value=6.5)
                        co_input = gr.Number(label="Initial Co (mg/L, for TSCF)", value=1.0)

                    predict_btn = gr.Button("🚀 Predict RCF & TSCF", variant="primary")

                # Right Column: Output Results
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 3. Predicted Uptake Results")
                    rcf_card_output = gr.HTML("<div class='result-card rcf-card'><div class='card-title'>🌱 Root Concentration Factor (RCF)</div><div class='primary-metric'>-</div></div>")
                    tscf_card_output = gr.HTML("<div class='result-card tscf-card'><div class='card-title'>💧 Transpiration Stream Concentration Factor (TSCF)</div><div class='primary-metric'>-</div></div>")

            # 1. SMILES Input Change: Auto-generate 2D structure review AND RDKit Descriptors table after 1 second (debounced)
            smiles_input.change(
                fn=update_structure_and_descriptors_delayed,
                inputs=[smiles_input],
                outputs=[mol_image_output, desc_table_output],
                trigger_mode="always_last",
                show_progress="hidden",
            )

            # Initial Page Load: Render structure and descriptors immediately
            demo.load(
                fn=update_structure_and_descriptors,
                inputs=[smiles_input],
                outputs=[mol_image_output, desc_table_output],
            )

            # 2. Species Selection Change: Instantly populate read-only Taxonomic Hierarchy
            species_dropdown.change(
                fn=update_species_taxonomy,
                inputs=[species_dropdown],
                outputs=[class_input, order_input, family_input, genus_input, plant_group_input, root_system_input],
            )

            # 3. Predict Button Click
            predict_btn.click(
                fn=predict_single,
                inputs=[
                    smiles_input,
                    species_dropdown,
                    class_input,
                    order_input,
                    family_input,
                    genus_input,
                    plant_group_input,
                    root_system_input,
                    exposure_dropdown,
                    condition_dropdown,
                    dur_input,
                    ph_input,
                    co_input,
                ],
                outputs=[rcf_card_output, tscf_card_output],
            )

        # --- TAB 2: Batch Prediction ---
        with gr.TabItem("📂 Batch Prediction"):
            gr.Markdown("### 📂 Upload CSV File for High-Throughput Batch Predictions")
            gr.Markdown("Upload a CSV file containing `SMILES` and optional plant/environmental feature columns. Missing numerical/categorical values will be automatically filled with default values.")

            with gr.Row():
                file_input = gr.File(label="Upload Input CSV", file_types=[".csv"])
                template_file = gr.File(value=str(APP_DIR / "sample_batch_input.csv"), label="Download Sample Template CSV", interactive=False)

            batch_btn = gr.Button("⚡ Run Batch Prediction", variant="primary")

            batch_output_df = gr.DataFrame(label="Batch Predictions Output Preview")
            download_file_output = gr.File(label="Download Full Predictions CSV")

            batch_btn.click(
                fn=predict_batch_csv,
                inputs=[file_input],
                outputs=[batch_output_df, download_file_output],
            )

        # --- TAB 3: Model Information ---
        with gr.TabItem("ℹ️ Model Information & Architecture"):
            gr.Markdown("(To be updated)")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, root_path="/SorbFormer", share=False)
