from dataclasses import dataclass
from typing import List, Optional
import os
import pandas as pd


@dataclass
class ENVnetConstructionParams:
    """
    Parameters for spectrum scoring and filtering
    
    """
    # Scoring parameters
    mz_tol: float = 0.002
    min_score: float = 0.7
    min_matches: int = 3 # this parameter is used for 3 things
    # 1. minimum number of ms2 ions found by deconvolution
    # 2. minimum number of close ms1 spectra with mz within mz_tol of calculated precursor mz
    # 3. cosine match minimum number of matching ions to score it
    override_matches: int = 20
    intensity_power: float = 0.5
    bin_width: float = 0.001
    similarity_cutoff: float = 0.7
    remblink_cutoff: float = 0.075  # cutoff for REMBLINK predictions
    min_rt: float = 0.5  # minimum retention time in minutes for a spectrum to be considered
    max_rt: float = 60.0  # maximum retention time in minutes for a spectrum to be considered
    min_observations: int = 2  # minimum number of duplicate spectra that need to be seen
    network_max_mz_difference: float = 30.0  # maximum m/z difference for network connections
    # Reference data
    ref_spec: Optional[List] = None
    ref_spec_nl: Optional[List] = None  # neutral loss reference spectra
    ref_pmz: Optional[List] = None
    ref_pmz_nl: Optional[List] = None # precursor m/z for neutral loss reference spectra
    ref: Optional[pd.DataFrame] = None # full data frame for reference spectra
    ref2: Optional[pd.DataFrame] = None # full data frame for neutral loss reference spectra
    
    # File paths and data
    metadata_folder: str = '/global/cfs/cdirs/metatlas/projects/carbon_network'
    module_path: str = '/global/homes/b/bpb/repos/envnet'

    # MDM (Mass Defect Matching) data
    mdm_df: Optional[pd.DataFrame] = None
    mdm_masses: Optional[List[float]] = None
    
    def __post_init__(self):
        """Initialize derived parameters after object creation"""
        mdm_path = os.path.join(self.module_path, 'data', 'mdm_neutral_losses.csv')
        self.mdm_df = pd.read_csv(mdm_path)
        self.mdm_masses = [0] + self.mdm_df['mass'].tolist()

        # Load reference data using the static method
        ref, ref2 = self.get_p2d2(self.mdm_df, mz_tol=self.mz_tol)
        bad = []
        for i,row in ref.iterrows():
            num_ions = row['spectrum'].shape[1]
            if num_ions==0:
                bad.append(i)
        ref.drop(index=bad,inplace=True)
        ref.reset_index(inplace=True,drop=True)
        # Populate reference attributes
        self.ref = ref
        self.ref2 = ref2
        self.ref_spec = ref['spectrum'].tolist()
        self.ref_pmz = ref['precursor_mz'].tolist()
        self.ref_spec_nl = ref2['nl_spectrum'].tolist()
        self.ref_pmz_nl = ref2['precursor_mz'].tolist()
    
    def get_p2d2(self,deltas,mz_tol=0.002):
        """
        Load or generate P2D2 reference data
        
        Args:
            deltas: DataFrame or dict containing mass differences
            mz_tol: m/z tolerance for matching

        Returns:
            tuple: (ref, ref2) DataFrames containing reference spectra
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
        
