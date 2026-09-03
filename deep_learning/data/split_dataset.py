import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

def split_and_save(file_path, target_col, test_size=0.2, random_state=42):
    """
    Load dataset, perform stratified split on the target column, and save files.
    """
    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path)
    
    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found in {file_path}")
        return

    # For continuous targets, create bins for stratification
    print(f"Creating bins for stratification on {target_col}...")
    # Use 5 bins (quintiles). Handle cases with very few unique values if necessary.
    try:
        # We use qcut to create equal-sized bins based on sample quantiles
        # duplicates='drop' helps if there are many identical values (e.g., zeros)
        stratify_cols = pd.qcut(df[target_col], q=5, labels=False, duplicates='drop')
    except Exception as e:
        print(f"Warning: Could not create bins for stratification: {e}. Falling back to random split.")
        stratify_cols = None

    print(f"Splitting dataset (test_size={test_size})...")
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=stratify_cols
    )
    
    # Generate output paths
    base_name = file_path.stem.replace("_features", "")
    train_val_path = file_path.parent / f"{base_name}_train_val.csv"
    test_path = file_path.parent / f"{base_name}_test.csv"
    
    print(f"Saving {train_val_path}...")
    train_val_df.to_csv(train_val_path, index=False)
    
    print(f"Saving {test_path}...")
    test_df.to_csv(test_path, index=False)
    
    print(f"Split completed: Train/Val={len(train_val_df)}, Test={len(test_df)}")

if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent
    
    # 1. Split RCF features
    rcf_features = data_dir / "rcf_features.csv"
    if rcf_features.exists():
        split_and_save(rcf_features, target_col="RCF")
    else:
        print(f"Skipping RCF: {rcf_features} not found.")

    # 2. Split TSCF features
    tscf_features = data_dir / "tscf_features.csv"
    if tscf_features.exists():
        split_and_save(tscf_features, target_col="TSCF")
    else:
        print(f"Skipping TSCF: {tscf_features} not found.")
