import pubchempy as pcp
from tqdm import tqdm

tqdm.pandas()

def get_smiles_from_name(molecule_name):
    """Retrieve canonical SMILES string for a given chemical name via PubChemPy."""
    try:
        results = pcp.get_compounds(molecule_name, 'name')
        if results:
            compound = results[0]
            return compound.smiles
        else:
            return "404"
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path

    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    tscf_file = data_dir / "tscf.csv"
    rcf_file = data_dir / "rcf.csv"

    if tscf_file.exists():
        df_tscf = pd.read_csv(tscf_file)
        if 'Chemical' in df_tscf.columns:
            df_tscf['SMILES'] = df_tscf['Chemical'].progress_apply(get_smiles_from_name)
            df_tscf.to_csv(data_dir / "tscf_with_smiles.csv", index=False)

    if rcf_file.exists():
        df_rcf = pd.read_csv(rcf_file)
        if 'Chemical' in df_rcf.columns:
            df_rcf['SMILES'] = df_rcf['Chemical'].progress_apply(get_smiles_from_name)
            df_rcf.to_csv(data_dir / "rcf_with_smiles.csv", index=False)