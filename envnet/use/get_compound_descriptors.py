import os
import pandas as pd
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
# from collections import Counter
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import MolFromSmiles, MolFromInchi, MolToInchiKey

descriptor_names = list(rdMolDescriptors.Properties.GetAvailableProperties())
get_descriptors = rdMolDescriptors.Properties(descriptor_names)


des_list = [x[0] for x in Descriptors._descList]
calculator = MoleculeDescriptors.MolecularDescriptorCalculator(des_list)

# all_hits = all_hits[pd.notna(all_hits['inchikey'])]
def mol_to_descriptors(mol):
    descriptors = get_descriptors.ComputeProperties(mol)
    return descriptors

def organize_descriptors_dict(ik,mol):
    """
    mol is an rdkit mol object
    """
    out = {'inchikey':ik,'mol':mol}
    d = list(mol_to_descriptors(mol))
    for j,dd in enumerate(d):
        out['property: %s'%descriptor_names[j]] = dd
    d = list(calculator.CalcDescriptors(mol))
    for j,dd in enumerate(d):
        out['descriptor: %s'%des_list[j]] = dd
       
    return out


def calc_descriptor_df(data):
    """
    data is a pandas dataframe with columns 'smiles' and 'inchi_key'
    """
    mols = {}
    out = []
    for i,row in data.iterrows():
        m = None
        try:
            m = MolFromSmiles(row['smiles'])
        except:
            pass
        # mols[row['inchi_key']] = m
        out_dict = organize_descriptors_dict(row['inchi_key'],m)
        out.append(out_dict)
    out = pd.DataFrame(out)
    out.drop(columns = ['mol'],inplace=True)
    return out




# cols = [c for c in descriptors.columns if not 'inchikey' in c]
# descriptors = descriptors.melt(id_vars=['inchikey','new_inchikey'],value_vars=cols)
# descriptors['type'] = descriptors['variable'].apply(lambda x: 'property' if 'property' in x else 'descriptor')
# descriptors['variable'] = descriptors['variable'].apply(lambda x: x.split(':')[-1].strip())