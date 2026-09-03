import pandas as pd
import torch
import sys
import os
from pathlib import Path

# Add project root to sys.path to import src
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.molecules.feature_extractor import MolecularFeatureExtractor

def extract_and_save(input_csv, output_csv):
    """
    Extract features from SMILES in input_csv and save to output_csv.
    """
    print(f"Processing {input_csv}...")
    df = pd.read_csv(input_csv)
    
    if "SMILES" not in df.columns:
        print(f"Error: 'SMILES' column not found in {input_csv}")
        return

    smiles_list = df["SMILES"].tolist()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. BioT5+ Features
    print("Extracting BioT5+ features (dim=768)...")
    biot5_extractor = MolecularFeatureExtractor(model_name="biot5-plus-base", device=device)
    biot5_features = biot5_extractor.extract_features(smiles_list, batch_size=32, show_progress=True)
    
    # Add BioT5 columns
    biot5_df = pd.DataFrame(
        biot5_features.numpy(), 
        columns=[f"biot5_{i}" for i in range(biot5_features.shape[1])]
    )
    df = pd.concat([df, biot5_df], axis=1)
    
    # Clean up to save memory
    del biot5_extractor, biot5_features, biot5_df
    torch.cuda.empty_cache()
    
    # 2. ChemBERTa Features
    print("Extracting ChemBERTa features (dim=384)...")
    chemberta_extractor = MolecularFeatureExtractor(model_name="chemberta-77m-mlm", device=device)
    chemberta_features = chemberta_extractor.extract_features(smiles_list, batch_size=32, show_progress=True)
    
    # Add ChemBERTa columns
    chemberta_df = pd.DataFrame(
        chemberta_features.numpy(), 
        columns=[f"chemberta_{i}" for i in range(chemberta_features.shape[1])]
    )
    df = pd.concat([df, chemberta_df], axis=1)
    
    # Clean up
    del chemberta_extractor, chemberta_features, chemberta_df
    torch.cuda.empty_cache()
    
    # 3. Save
    print(f"Saving to {output_csv}...")
    df.to_csv(output_csv, index=False)
    print(f"Saved {df.shape[0]} rows and {df.shape[1]} columns.")

if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent
    
    # Process RCF mapping if it exists
    rcf_path = data_dir / "rcf.csv"
    if rcf_path.exists():
        extract_and_save(rcf_path, data_dir / "rcf_features.csv")
    
    # Process TSCF mapping if it exists
    tscf_path = data_dir / "tscf.csv"
    if tscf_path.exists():
        extract_and_save(tscf_path, data_dir / "tscf_features.csv")
