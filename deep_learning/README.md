---
title: SorbFormer - Deep Multimodal Transformer for Plant Uptake Prediction
emoji: 🌿
colorFrom: green
colorTo: green
sdk: gradio
sdk_version: 4.41.0
app_file: app.py
pinned: false
short_description: Predict plant uptake (RCF/TSCF) using SorbFormer.
---

# 🧠 SorbFormer: Deep Multimodal Transformer Pipeline

This module implements the **SorbFormer** deep learning architecture for predicting **Root Concentration Factor (RCF)** and **Transpiration Stream Concentration Factor (TSCF)** of organic contaminants in plant systems.

Developed by the **[Computational Biology and Bioinformatics Laboratory (CBBL)](https://balalab-skku.org/)**.

- 🌐 **Interactive Web App**: [https://balalab-skku.org/SorbFormer/](https://balalab-skku.org/SorbFormer/)

---

## 🚀 Key Features

- **Multimodal Architecture**: Fuses 768-dimensional SELFIES chemical language embeddings (BioT5+) with plant taxonomy, environmental conditions, and physicochemical traits via Cross-Attention & Gated Fusion.
- **Single & Batch Predictions**: Gradio web interface supporting interactive 2D structure rendering, automatic RDKit descriptor calculation, read-only taxonomic hierarchy auto-population, and high-throughput CSV batch predictions.
- **Interpretation Suite**: Active paper interpretation mechanisms:
  1. **Permutation Feature Importance (PFI)** (`importance.py`)
  2. **Token Attribution via Integrated Gradients** (`molecular.py`)
  3. **Phylogenetic Error Mapping & Taxonomic Signal** (`phylogenetic.py`)

---

## 📥 Input Features

1. **Chemical Representation**: SMILES string automatically converted to SELFIES for BioT5+ feature extraction.
2. **Plant Taxonomy**: Class, Order, Family, Genus, Species (32 unique species auto-populated).
3. **Exposure & Trait Conditions**: Functional group, root system, exposure medium, plant condition, exposure duration (days), pH, initial concentration $C_0$ (mg/L).
4. **Physicochemical Descriptors**: 18 continuous RDKit descriptors (`logKow`, `MW`, `HBD`, `HBA`, `TPSA`, `Rotatable Bonds`, `Labute ASA`, etc.).

---

## 📦 Pre-trained Model Weights

Due to large file sizes (~860MB total), the pre-trained PyTorch checkpoint files (`model.ckpt`) are hosted externally on Hugging Face at `cbbl-skku-org/SorbFormer`. 

**You do not need to download the models manually!** 
The application uses `huggingface_hub` to automatically download the necessary RCF and TSCF checkpoints into the `models/` directory the very first time you start the app. 

---

## 🛠️ How to Deploy Locally

You can easily deploy the SorbFormer web interface on your own machine using either Docker or standard Python.

### Option A: Using Docker (Recommended)
1. Ensure you have Docker and Docker Compose installed.
2. In the `deep_learning` directory, run:
   ```bash
   docker compose up -d --build
   ```
3. The app will be available in your browser at `http://127.0.0.1:8181`.
*(Note: If you have an NVIDIA GPU, you can uncomment the `deploy` block in `docker-compose.yml` to enable GPU acceleration).*

### Option B: Using Python
1. Create and activate a Python 3.10 virtual environment.
2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Gradio app:
   ```bash
   python app.py
   ```
4. Access the app at `http://127.0.0.1:7860`.

---

## 🔒 For Maintainers: Pushing Models to Hugging Face

To host the models so the app can auto-download them, you must push the model files to your Hugging Face Organization (`cbbl-skku-org`):

1. Go to [huggingface.co/new](https://huggingface.co/new) and create a new **Model** repository under the owner `cbbl-skku-org` named `SorbFormer`.
2. Install the huggingface CLI and login:
   ```bash
   pip install -U "huggingface_hub[cli]"
   hf auth login
   ```
3. Upload the model folders directly via the CLI:
   ```bash
   hf upload cbbl-skku-org/SorbFormer ./models/rcf /rcf --repo-type model
   hf upload cbbl-skku-org/SorbFormer ./models/tscf /tscf --repo-type model
   ```
This will place the checkpoints in the exact structure the `app.py` script expects to download.

---

## 🏛️ Laboratory

Developed by the **[Computational Biology and Bioinformatics Laboratory (CBBL)](https://balalab-skku.org/)** at Sungkyunkwan University.

---

## 📖 Citation

(To be updated)
