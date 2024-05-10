import networkx as nx
import pandas as pd
import os
import glob
from typing import List, Tuple

from tqdm.notebook import tqdm

# replace with submodules
import sys
sys.path.insert(0,'/global/homes/t/tharwood/repos/blink')
from blink import open_msms_file
sys.path.insert(0,'/global/homes/t/tharwood/repos/metatlas')
from metatlas.io import feature_tools as ft

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from build.preprocess import run_workflow



def graph_to_df() -> pd.DataFrame:
    
    G = nx.read_graphml('/global/cfs/cdirs/metatlas/projects/carbon_network/CarbonNetwork.graphml')

    node_data = dict(G.nodes(data=True))
    node_data = pd.DataFrame(node_data).T
    node_data.index.name = 'node_id'
    node_data.reset_index(inplace=True,drop=False)
    
    return node_data


def merge_spectral_data(node_data: pd.DataFrame) -> pd.DataFrame:
    
    original_spectra = open_msms_file('/global/cfs/cdirs/metatlas/projects/carbon_network/original_spectra.mgf')
    nl_spectra = open_msms_file('/global/cfs/cdirs/metatlas/projects/carbon_network/nl_spectra.mgf')
    
    if 'orignal_id' in original_spectra:
        original_spectra.rename(columns={'orignal_id': 'original_id'}, inplace=True)
    if 'orignal_id' in nl_spectra:
        nl_spectra.rename(columns={'orignal_id': 'original_id'}, inplace=True)
        
    original_spectra['node_id'] = original_spectra['original_id'].apply(lambda x: str(float(x)))
    nl_spectra['node_id'] = nl_spectra['original_id'].apply(lambda x: str(float(x)))
    
    original_spectra.rename(columns={c: c+'_original_spectra' for c in original_spectra.columns if c not in ['node_id']}, inplace=True)
    nl_spectra.rename(columns={c: c+'_nl_spectra' for c in nl_spectra.columns if c not in ['node_id']}, inplace=True)
    
    merged_node_data = pd.merge(node_data, original_spectra, on='node_id')
    merged_node_data = pd.merge(merged_node_data, nl_spectra, on='node_id')
    
    return merged_node_data


def get_files_df(exp_dir: str) -> pd.DataFrame:
    
    files = glob.glob(os.path.join(exp_dir,'*NEG*.h5'))
    files = [f for f in files if not 'exctrl' in f.lower()]
    files = [f for f in files if not 'qc' in f.lower()]
    
    files = pd.DataFrame(files, columns=['filename'])

    files['experiment'] = files['filename'].apply(lambda x: '_'.join(x.split('/')[-1].split('_')[4:6]))
    files['sampletype'] = files['filename'].apply(lambda x: x.split('/')[-1].split('_')[12])
    
    return files


def make_node_atlas(node_data: pd.DataFrame, rt_range) -> pd.DataFrame:
    
    node_atlas = node_data[['node_id', 'precursor_mz']].copy()
    node_atlas.rename(columns={'precursor_mz': 'mz', 'node_id': 'label'}, inplace=True)
    node_atlas['rt_tolerance'] = 100
    node_atlas['rt_min'] = rt_range[0]
    node_atlas['rt_max'] = rt_range[1]
    node_atlas['rt_peak'] = sum(rt_range) / 2
    
    return node_atlas


def get_sample_ms1_data(node_atlas: pd.DataFrame, sample_files: List[str], mz_ppm_tolerance: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Collect MS1 data from experimental sample data using node attributes."""

    experiment_input = ft.setup_file_slicing_parameters(node_atlas, sample_files, base_dir=os.getcwd(), ppm_tolerance=mz_ppm_tolerance, polarity='negative')

    ms1_data = []

    for file_input in tqdm(experiment_input, unit="file"):

        data = ft.get_data(file_input, save_file=False, return_data=True, ms1_feature_filter=False)
        data['ms1_summary']['lcmsrun_observed'] = file_input['lcmsrun']

        ms1_data.append(data['ms1_summary'])

    ms1_data = pd.concat(ms1_data)

    return ms1_data


def get_sample_ms2_data(sample_files: List[str]) -> pd.DataFrame:
    """Collect all MS2 data from experimental sample data and calculate ."""
    
    delta_mzs = pd.read_csv('/global/cfs/cdirs/metatlas/projects/carbon_network/mdm_neutral_losses.csv')
    
    ms2_data = []
    
    for file in tqdm(sample_files, unit='file'):
        
        data = run_workflow(file,
                            delta_mzs,
                            do_buddy = False,
                            elminate_duplicate_spectra = False)
        
        ms2_data.append(data)
        
    ms2_data = pd.concat(ms2_data).reset_index(drop=True)
        
    return ms2_data