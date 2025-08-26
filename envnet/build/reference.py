import os
import pandas as pd
import numpy as np
from typing import Tuple

def load_p2d2_reference_data(deltas: pd.DataFrame, mz_tol: float = 0.002) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load or generate P2D2 reference data.
    
    Converts reference library spectra into neutral loss spectra using MDM delta values.
    
    Args:
        mdm_df: DataFrame containing mass differences
        mz_tol: m/z tolerance for matching
        
    Returns:
        tuple: (ref, ref2) DataFrames containing reference and neutral loss spectra
    """
    ref_filename = '/global/cfs/cdirs/metatlas/projects/spectral_libraries/deduplicated_merged_library/for_scn_ref_p2d2.pkl'
    ref2_filename = '/global/cfs/cdirs/metatlas/projects/spectral_libraries/deduplicated_merged_library/for_scn_ref2_p2d2.pkl'
    if os.path.isfile(ref_filename) and os.path.isfile(ref2_filename):
        ref = pd.read_pickle(ref_filename)
        ref2 = pd.read_pickle(ref2_filename)
    else:
        from metatlas.io import feature_tools as ft
        my_path = '/global/cfs/cdirs/metatlas/projects/spectral_libraries/deduplicated_merged_library'
        my_file = 'dm_msms_refs.tab'
        ref_filename = os.path.join(my_path,my_file)
        cols = ['name', 'spectrum',  'precursor_mz','formula',
            'polarity', 'adduct', 'inchi_key','smiles']
        ref = pd.read_csv(ref_filename,sep='\t',usecols=cols)
        ref.index.name = 'original_p2d2_index'
        ref.reset_index(inplace=True,drop=False)
        ref = ref[ref['polarity']=='negative']
        ref = ref[ref['adduct']=='[M-H]-']
        ref['spectrum'] = ref['spectrum'].apply(lambda x: np.asarray(eval(x)))
        ref['precursor_mz'] = ref['precursor_mz'].astype(float)
        ref.reset_index(inplace=True,drop=True)


        is_dataframe = isinstance(deltas, pd.DataFrame)
        # is_dictionary = isinstance(deltas, dict)
        if is_dataframe==True:
            deltas = deltas.set_index('difference')['mass'].to_dict()

        out = []
        for i,row in ref.iterrows():
            # def nl_spectra_p2d2(spectrum,precursor_mz,deltas,mz_tol=0.002):
            ms2_df = pd.DataFrame(row['spectrum'].T,columns=['mz','i'])
            ms2_df['original_p2d2_index'] = row['original_p2d2_index']
            ms2_df['precursor_mz'] = row['precursor_mz']

            # iterate through the deltas and shift them up towards the expected precursor m/z
            temp = {}
            for k,v in deltas.items():
                temp[k] = ms2_df['mz'] + v
            temp = pd.DataFrame(temp)
            ms2_df = pd.concat([ms2_df,temp],axis=1)

            ms2_df = ms2_df.melt(id_vars=['mz','i','original_p2d2_index','precursor_mz'],value_vars=deltas.keys())
            ms2_df = ms2_df[abs(ms2_df['value']-ms2_df['precursor_mz'])<mz_tol]
            out.append(ms2_df)


        out = pd.concat(out)
        out = out[['original_p2d2_index','mz','i']]
        out.reset_index(inplace=True,drop=True)

        ref_nl = ft.group_duplicates(out,'original_p2d2_index')#,make_string=False)#,precision={'i':0,'mz':4,'rt':2})
        ref_nl['nl_spectrum'] = ref_nl.apply(lambda x: np.asarray([x['mz'],x['i']]),axis=1)
        ref_nl['nl_spectrum_num_ions'] = ref_nl['mz'].apply(lambda x: len(x))
        ref_nl.drop(columns=['mz','i'],inplace=True)
        ref_nl = ref_nl[ref_nl['nl_spectrum_num_ions']>2]
        ref2 = pd.merge(ref,ref_nl,on='original_p2d2_index',how='inner')
        ref2.reset_index(inplace=True,drop=True)
        ref.to_pickle(ref_filename)
        ref2.to_pickle(ref2_filename)
    return ref,ref2