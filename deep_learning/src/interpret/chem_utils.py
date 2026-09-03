import re
import inspect
from rdkit import Chem
from rdkit.Chem import Fragments

# --- Constants & Overrides ---
METAL_ATOMIC_NUMBERS = {
    3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
    72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
    87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103,
    104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
}
PERIODIC_TABLE = Chem.GetPeriodicTable()

DISPLAY_NAME_OVERRIDES = {
    "fr_Al_COO": "aliphatic carboxylic acid",
    "fr_Al_OH": "aliphatic hydroxyl",
    "fr_Al_OH_noTert": "aliphatic hydroxyl (non-tertiary)",
    "fr_ArN": "nitrogen functional group attached to aromatic ring",
    "fr_Ar_COO": "aromatic carboxylic acid",
    "fr_Ar_N": "aromatic nitrogen",
    "fr_Ar_NH": "aromatic amine",
    "fr_Ar_OH": "aromatic hydroxyl",
    "fr_COO": "carboxylic acid",
    "fr_COO2": "carboxylic acid",
    "fr_C_O": "carbonyl oxygen",
    "fr_C_O_noCOO": "carbonyl oxygen (excluding carboxylic acid)",
    "fr_C_S": "thiocarbonyl / carbon-sulfur motif",
    "fr_HOCCN": "hydroxyethylamine-like motif",
    "fr_Imine": "imine",
    "fr_NH0": "tertiary amine nitrogen",
    "fr_NH1": "secondary amine nitrogen",
    "fr_NH2": "primary amine nitrogen",
    "fr_N_O": "N-oxide / N-O motif",
    "fr_Ndealkylation1": "N-dealkylation motif",
    "fr_Ndealkylation2": "tertiary alicyclic amine motif",
    "fr_Nhpyrrole": "pyrrolic NH",
    "fr_SH": "thiol",
    "fr_allylic_oxid": "allylic oxidation motif",
    "fr_aryl_methyl": "aryl methyl",
    "fr_bicyclic": "bicyclic ring system",
    "fr_hdrzine": "hydrazine",
    "fr_hdrzone": "hydrazone",
    "fr_priamide": "primary amide",
    "fr_quatN": "quaternary nitrogen",
    "fr_term_acetylene": "terminal acetylene",
    "fr_unbrch_alkane": "unbranched alkane (>=4 carbons)",
    "fr_phos_acid": "phosphoric acid",
    "fr_phos_ester": "phosphoric ester",
}

# --- Internal Helpers ---

def singularize_label(text: str) -> str:
    replacements = [
        (r"\bcarboxylic acids\b", "carboxylic acid"),
        (r"\bhydroxyl groups\b", "hydroxyl"),
        (r"\bfunctional groups\b", "functional group"),
        (r"\bquaternary nitrogens\b", "quaternary nitrogen"),
        (r"\baromatic nitrogens\b", "aromatic nitrogen"),
        (r"\baromatic amines\b", "aromatic amine"),
        (r"\bprimary sulfonamides\b", "primary sulfonamide"),
        (r"\bsulfonamides\b", "sulfonamide"),
        (r"\bphosphoric acid groups\b", "phosphoric acid"),
        (r"\bphosphoric ester groups\b", "phosphoric ester"),
        (r"\bketones\b", "ketone"), (r"\besters\b", "ester"), (r"\besters\b", "ester"),
        (r"\bethers\b", "ether"), (r"\baldehydes\b", "aldehyde"), (r"\bamides\b", "amide"),
        (r"\banilines\b", "aniline"), (r"\bnitriles\b", "nitrile"), (r"\bnitro groups\b", "nitro"),
        (r"\bring systems\b", "ring system"), (r"\brings\b", "ring"), (r"\bgroups\b", "group"),
        (r"\bacetylenes\b", "acetylene"), (r"\balkanes\b", "alkane"), (r"\bamines\b", "amine"),
        (r"\bacids\b", "acid"), (r"\bnitrogens\b", "nitrogen"), (r"\boxygens\b", "oxygen"),
        (r"\bhalides\b", "halide"), (r"\bhalogens\b", "halogen"), (r"\bsites\b", "site"),
    ]
    label = text
    for pattern, target in replacements:
        label = re.sub(pattern, target, label)
    return label

def make_display_name(fragment_id: str, description: str) -> str:
    if fragment_id in DISPLAY_NAME_OVERRIDES:
        return DISPLAY_NAME_OVERRIDES[fragment_id]
    label = description.strip().replace("Aromatic carboxylic acide", "aromatic carboxylic acid")
    if label.startswith("Number of "):
        label = label[len("Number of ") :]
    label = singularize_label(label)
    label = label.strip().rstrip(".")
    if not label:
        label = fragment_id.replace("fr_", "").replace("_", " ")
    return label[0].lower() + label[1:] if label else label

def load_fragment_rules():
    rules = []
    for name in sorted(attr for attr in dir(Fragments) if attr.startswith("fr_")):
        fn = getattr(Fragments, name)
        doc = inspect.getdoc(fn) or ""
        rules.append({
            "fragment_id": name, "description": doc,
            "display_name": make_display_name(name, doc),
            "callable": fn,
        })
    return rules

FRAGMENT_RULES = load_fragment_rules()

def detect_manual_motifs(mol: Chem.Mol):
    if mol.GetNumAtoms() != 1: return []
    atom = mol.GetAtomWithIdx(0)
    atomic_num = atom.GetAtomicNum()
    if atomic_num not in METAL_ATOMIC_NUMBERS: return []
    element_name = PERIODIC_TABLE.GetElementName(atomic_num).strip().lower()
    return ["metal", element_name]

# --- Public API ---

def annotate_smiles(smiles: str):
    """
    Annotate a SMILES string with chemical motifs.
    Returns: (motifs_string, status, matched_counts_dict, manual_motifs_list)
    """
    smiles = (smiles or "").strip()
    if not smiles: return "", "missing_smiles", {}, []
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return "", "invalid_smiles", {}, []

    matched_counts = {}
    for rule in FRAGMENT_RULES:
        count = rule["callable"](mol)
        if count > 0:
            matched_counts[rule["fragment_id"]] = {
                "count": int(count),
                "display_name": rule["display_name"],
            }

    manual_motifs = detect_manual_motifs(mol)
    if not matched_counts and not manual_motifs:
        return "", "unresolved_after_manual", {}, []

    seen_labels = set()
    ordered_labels = []
    for rule in FRAGMENT_RULES:
        if rule["fragment_id"] not in matched_counts: continue
        label = matched_counts[rule["fragment_id"]]["display_name"]
        if label not in seen_labels:
            seen_labels.add(label)
            ordered_labels.append(label)
    for label in manual_motifs:
        if label not in seen_labels:
            seen_labels.add(label)
            ordered_labels.append(label)

    motifs = "; ".join(ordered_labels)
    status = "matched_rdkit" if matched_counts else "matched_manual_only"
    if matched_counts and manual_motifs: status = "matched_rdkit_and_manual"
    
    return motifs, status, matched_counts, manual_motifs
