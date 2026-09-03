chemical_to_smi = {
    'Nonylphenol tetraethoxylate': "CCCCCCCCCC1=CC=C(C=C1)OCCOCCOCCOCCO", # https://direct.hpc-j.co.jp/sds/eng/V2-09_EN.pdf
    'Nonylphenol nonylethoxylate': "C(COCCOC=1(C=CC(=CC=1)CCCCCCCCC))O", # https://www.chemicalbook.com/ChemicalProductProperty_EN_CB1159958.htm
    'Acetone O-methylcarbamoyloxime': "CC1=C(C=C(C=C1)NC(=O)ON=C(C)C)C", # https://pubchem.ncbi.nlm.nih.gov/compound/O-_N-_3_4-Xylyl_carbamoyl_acetone-oxime
    'Benzaldehyde O-methylcarbamoyloxime': 'C(C=1(C=CC=CC=1))=NOC(N)=OC', # https://www.chemicalbook.com/ProductChemicalPropertiesCB11446350_EN.htm
    '4-chlorobenzaldehyde O-methylcarbamoyloxime': "C(C=1(C=CC(=CC=1)Cl))=NOC(=O)N", # https://m.chemicalbook.com/ProdSupplierGWCB32300154_EN.htm
    '3,4-dichlorobenzaldehyde O-methylcarbamoyloxime': "C(C=1(C=C(C(=CC=1)Cl)Cl))=NOC(N)=OC", # https://www.chemicalbook.com/ProdSupplierGNCB71441069.htm
    '3-phenoxybenzaldehyde O-methylcarbamoyloxime': "C(C=1(C=C(C=CC=1)OC=2(C=CC=CC=2)))=NOC(N)=OC",# https://www.chemicalbook.com/chemicalproductproperty_ru_cb91427852.htm
    '3-(3,4-dichlorophenoxy)benzaldehyde O-methylcarbamoyloxime': "C(C=1(C=C(C=CC=1)OC=2(C=C(C(=CC=2)Cl)Cl)))=NOC(N)=OC", # https://amp.chemicalbook.com/ChemicalProductProperty_EN_CB91440011.htm
    '3-(methylthio) phenylurea': "CSC1=CC=CC(=C1)NC(=O)N", # https://pubchem.ncbi.nlm.nih.gov/compound/3-_Methylthio_phenylurea
    '4,4-sulfonyldianiline': "C1=CC(=CC=C1N)S(=O)(=O)C2=CC=C(C=C2)N.I.I", # https://pubchem.ncbi.nlm.nih.gov/compound/4_4_-Sulfonyldianiline-dihydriodide#section=Computed-Descriptors
    'Ethylether': "CCOCC", # https://pubchem.ncbi.nlm.nih.gov/compound/Diethyl-Ether#section=Names-and-Identifiers
    "p'p-DDT": "C(C1(=CC=C(C=C1)Cl))(C2(=CC=C(C=C2)Cl))C(Cl)(Cl)Cl", # https://webbook.nist.gov/cgi/cbook.cgi?ID=50-29-3
    "3,5-D": "OC(=O)COc1cc(Cl)cc(Cl)c1", # https://www.stenutz.eu/chem/solv6%20(2).php?name=3%2C5-dichlorophenoxyacetic+acid
    "Trimethylphenyl phosphate": "CC1=CC=C(C=C1)OP(=O)(OC2=CC=C(C=C2)C)OC3=CC=C(C=C3)C.CC1=CC(=CC=C1)OP(=O)(OC2=CC=CC(=C2)C)OC3=CC=CC(=C3)C.CC1=CC=CC=C1OP(=O)(OC2=CC=CC=C2C)OC3=CC=CC=C3C", # https://pubchem.ncbi.nlm.nih.gov/compound/Tris_methylphenyl_-phosphate#section=InChI
    "Di-n-butyl phthalate (DBP)": "CCCCOC(=O)C1=CC=CC=C1C(=O)OCCCC", # https://pubchem.ncbi.nlm.nih.gov/compound/Dibutyl-Phthalate
    "4-bromodiphenyl ether (BDE-3)": "C1=CC=C(C=C1)OC2=CC=C(C=C2)Br", # https://pubchem.ncbi.nlm.nih.gov/compound/4-Bromodiphenyl-Ether#section=Names-and-Identifiers
    "gama-hexachlorocyclohexane": "Cl[C@H]1[C@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@H](Cl)[C@H]1Cl", # https://go.drugbank.com/drugs/DB00431
    "4-Chlorobenzaldehyde o-methylcarbamoyloxime": "C(C=1(C=CC(=CC=1)Cl))=NOC(=O)N", # https://m.chemicalbook.com/ProdSupplierGWCB32300154_EN.htm
    '3-Phenoxybenzaldehyde o-methylcarbamoyloxime': "C(C=1(C=C(C=CC=1)OC=2(C=CC=CC=2)))=NOC(N)=OC",# https://www.chemicalbook.com/chemicalproductproperty_ru_cb91427852.htm
    '3-(3,4-Dichlorophenoxy)benzaldehyde o-methylcarbamoyloxime': "C(C=1(C=C(C=CC=1)OC=2(C=C(C(=CC=2)Cl)Cl)))=NOC(N)=OC", # https://amp.chemicalbook.com/ChemicalProductProperty_EN_CB91440011.htm
    '3-(Methylthio) phenylurea': "CSC1=CC=CC(=C1)NC(=O)N", # https://pubchem.ncbi.nlm.nih.gov/compound/3-_Methylthio_phenylurea
    'Perlfuorohexane sulfonate': "C(C(C(C(F)(F)S(=O)(=O)[O-])(F)F)(F)F)(C(C(F)(F)F)(F)F)(F)F", # https://pubchem.ncbi.nlm.nih.gov/compound/Perfluorohexanesulfonate#section=IUPAC-Name
    "3-bromophenyl": "Brc1cccc(c1)", # https://pubchem.ncbi.nlm.nih.gov/compound/3-Bromophenyl-benzoate#section=Names-and-Identifiers
}