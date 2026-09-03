# 🌱 SorbFormer: Multimodal Deep Learning for Plant-Uptake Prediction 🌿

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Web App](https://img.shields.io/badge/%F0%9F%8C%90%20Web%20App-Online-brightgreen.svg)](https://balalab-skku.org/SorbFormer/)
[![Lab](https://img.shields.io/badge/CBBL-SKKU-00529B.svg)](https://balalab-skku.org/)

**SorbFormer** is a hybrid multimodal deep learning framework designed to predict plant-uptake characteristics of emerging organic contaminants (including PFAS, pharmaceuticals, and pesticides):
1. **Root Concentration Factor (RCF)**: Sorption capacity in root tissues.
2. **Transpiration Stream Concentration Factor (TSCF)**: Translocation efficiency into xylem sap.

Developed by the **[Computational Biology and Bioinformatics Laboratory (CBBL)](https://balalab-skku.org/)**.

---

### 📝 Project & Manuscript Status
- [x] SorbFormer Model & Web App Released ([Web App](https://balalab-skku.org/SorbFormer/))
- [ ] Paper Submitted
- [ ] Under Review
- [ ] Under Revision
- [ ] Accepted / Published

---

## 🏛️ Organization & Laboratory

- **Laboratory**: [Computational Biology and Bioinformatics Laboratory (CBBL)](https://balalab-skku.org/)
- **Institution**: Sungkyunkwan University (SKKU)
- **Website**: [https://balalab-skku.org/](https://balalab-skku.org/)
- **Interactive Web App**: [https://balalab-skku.org/SorbFormer/](https://balalab-skku.org/SorbFormer/)

---

## 📁 Repository Structure

```text
SorbFormer/
├── 🧠 deep_learning/           # Deep Learning framework (SorbFormer & BioT5+)
│   ├── app.py                 # Gradio Web Deployment Application
│   ├── bash/                  # Shell scripts for optimization, training & evaluation
│   ├── data/                  # Experimental datasets (RCF & TSCF)
│   ├── scripts/               # Entrypoint Python execution scripts
│   │   ├── 01_optimize_models.py       # Hyperparameter optimization (Optuna)
│   │   ├── 02_train_best_models.py     # Final model refitting and training
│   │   ├── 03_evaluate_performance.py  # Test set evaluation
│   │   └── 04_interpret_models.py      # Interpretation suite (PFI, IG, Phylogenetic)
│   ├── src/                   # Model architectures, feature extractors & interpretation modules
│   │   ├── interpret/         # Active paper mechanisms (importance, molecular, phylogenetic)
│   │   └── molecules/         # BioT5+ feature extractors and chemical helpers
│   ├── Dockerfile             # Docker container definition
│   ├── docker-compose.yml     # Docker Compose for deployment
│   ├── environment.yml        # Conda environment definition
│   └── requirements.txt       # Dependencies for Deep Learning
│
└── 🤖 machine_learning/        # Traditional Machine Learning baselines
    ├── ML_run.py              # Unified ML benchmarking execution script
    ├── *.py                   # Individual ML regression models (RF, XGBoost, LightGBM, CatBoost, etc.)
    └── requirements.txt       # Dependencies for Machine Learning
```

---

## ⚙️ Installation

Clone the repository and set up your Python environment:

```bash
git clone https://github.com/ndhieunguyen/SorbFormer.git
cd SorbFormer/deep_learning
pip install -r requirements.txt
```

---

## 📦 Pre-trained Model Weights

Due to large file sizes (~860MB total), the pre-trained PyTorch checkpoint files (`model.ckpt`) are hosted externally on Hugging Face at `cbbl-skku-org/SorbFormer`. 

**You do not need to download the models manually!** 
The application uses `huggingface_hub` to automatically download the necessary RCF and TSCF checkpoints into the `models/` directory the very first time you start the app.

- 🌐 **Online Interactive Web UI**: [Web App](https://balalab-skku.org/SorbFormer/)

---

## 🔬 Running Deep Learning Pipeline

Navigate to the `deep_learning/` directory to run training, evaluation, or web deployment:

### 1. 🌐 Launch Gradio Web UI
```bash
python app.py
```

### 2. 🎯 Hyperparameter Optimization
```bash
bash bash/run_optimization.sh rcf r2 biot5-plus-base
```

### 3. 🏋️ Model Refitting & Training
```bash
bash bash/run_training.sh rcf biot5-plus-base cross_attention
```

### 4. 📊 Test Evaluation
```bash
bash bash/run_evaluation.sh final_models_rcf
```

### 5. 🔍 Model Interpretation
```bash
bash bash/run_interpretation.sh rcf final_models_rcf/trial_0
```

---

## 🤖 Running Machine Learning Baselines

Navigate to the `machine_learning/` directory to evaluate traditional ML baselines (Random Forest, XGBoost, LightGBM, CatBoost, Extra Trees, AdaBoost, KNN, MLP, etc.):

```bash
cd machine_learning
pip install -r requirements.txt
python ML_run.py
```

---

## 📖 Citation

(To be updated)
