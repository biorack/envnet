import networkx as nx
import pandas as pd
import os
import glob
from scipy import interpolate
import numpy as np
from typing import List, Tuple

from tqdm.notebook import tqdm

# replace with submodules
import sys
sys.path.insert(0,'/global/homes/b/bpb/repos/blink')
from blink import open_msms_file
import blink

sys.path.insert(0,'/global/homes/b/bpb/repos/metatlas')
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


def get_files_df(exp_dir: str,parse_filename=False) -> pd.DataFrame:
    
    files = glob.glob(os.path.join(exp_dir,'*NEG*.h5'))
    files = [f for f in files if not 'exctrl' in f.lower()]
    files = [f for f in files if not 'qc' in f.lower()]
    
    files = pd.DataFrame(files, columns=['filename'])
    if parse_filename==True:
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



def get_best_ms1_rawdata(ms1_data,node_data):
    max_ms1_data = ms1_data.copy()
    max_ms1_data.sort_values('peak_height', ascending=False,inplace=True)
    max_ms1_data.drop_duplicates(subset='label',inplace=True)
    max_ms1_data.rename(columns={'label': 'node_id'},inplace=True)
    max_ms1_data = pd.merge(max_ms1_data, node_data[['node_id', 'precursor_mz']], on='node_id',how='left')
    max_ms1_data['ppm_error'] = max_ms1_data.apply(lambda x: ((x.precursor_mz - x.mz_centroid) / x.precursor_mz) * 1000000, axis=1)
    return max_ms1_data

def get_best_ms2_rawdata(ms2_data):
    max_ms2 = ms2_data.sort_values('score',ascending=False).drop_duplicates('node_id')
    max_ms2.drop(columns=['query','ref'],inplace=True)
    max_ms2.reset_index(inplace=True,drop=True)
    return max_ms2

def get_best_ms1_ms2_combined(max_ms1_data,max_ms2_data):
    cols = ['node_id','score','matches','lcmsrun_observed']
    out = pd.merge(max_ms1_data,
    max_ms2_data[cols].add_prefix('ms2_'),
    left_on='node_id',
    right_on='ms2_node_id',
    how='outer')
    out.sort_values('ms2_score',ascending=False,inplace=True)
    out.reset_index(inplace=True,drop=True)
    return out

def do_blink(discretized_spectra,exp_df,ref_df,msms_score_min=0.7,msms_matches_min=3,mz_ppm_tolerance=5):
    scores = blink.score_sparse_spectra(discretized_spectra)
    # m = blink.reformat_score_matrix(S12)
    scores = blink.filter_hits(scores,min_score=msms_score_min,min_matches=msms_matches_min)
    scores = blink.reformat_score_matrix(scores)
    scores = blink.make_output_df(scores)
    for c in scores.columns:
        scores[c] = scores[c].sparse.to_dense()
    cols = ['query','ref']
    for c in cols:
        scores[c] = scores[c].astype(int)
    scores = pd.merge(scores,exp_df[['precursor_mz']].add_suffix('_exp'),left_on='query',right_index=True,how='left')
    scores = pd.merge(scores,ref_df[['precursor_mz']].add_suffix('_ref'),left_on='ref',right_index=True,how='left')
    scores['mz_diff'] = abs(scores['precursor_mz_exp'] - scores['precursor_mz_ref']) / scores['precursor_mz_ref'] * 1e6
    scores = scores[scores['mz_diff']<mz_ppm_tolerance]
    scores.set_index(cols,inplace=True,drop=True)

    return scores

def merge_or_nl_blink(nl_blink,or_blink):
    t = pd.merge(nl_blink.add_suffix('_nl'),or_blink.add_suffix('_or'),left_index=True,right_index=True,how='outer')
    t['score'] = t[['score_nl','score_or']].apply(lambda x: np.nanmax(x),axis=1)
    t['best_match_method'] = t[['score_nl','score_or']].idxmax(axis=1)
    idx = t['best_match_method']=='score_or'
    t.loc[idx,'matches'] = t.loc[idx,'matches_or']
    idx = t['best_match_method']=='score_nl'
    t.loc[idx,'matches'] = t.loc[idx,'matches_nl']
    cols = ['score','matches','best_match_method']
    t = t[cols]
    t.reset_index(inplace=True,drop=False)
    return t
    

def remove_unnecessary_ms2_data(ms2_data,merged_node_data,ppm_filter=5):


    ms1_mz = merged_node_data['precursor_mz'].sort_values().values
    ms2_data = ms2_data[ms2_data['precursor_mz']>ms1_mz.min()]
    ms2_data = ms2_data[ms2_data['precursor_mz']<ms1_mz.max()]
    f = interpolate.interp1d(ms1_mz,np.arange(ms1_mz.size),kind='nearest',bounds_error=False,fill_value='extrapolate') #get indices of all mz values in the atlas
    idx = f(ms2_data['precursor_mz'].values)   # iterpolate to find the nearest mz in the data for each mz in an atlas
    idx = idx.astype(int)
    ms2_data['nearest_precursor'] = ms1_mz[idx]
    ms2_data['mz_diff'] = abs(ms2_data['nearest_precursor']-ms2_data['precursor_mz']) / ms2_data['precursor_mz'] * 1e6
    ms2_data = ms2_data[ms2_data['mz_diff']<ppm_filter]
    ms2_data.reset_index(inplace=True,drop=True)
    return ms2_data

def calculate_ms1_summary(row):
    """
    Calculate summary properties for features from data
    """
    d = {}
    #Before doing this make sure "in_feature"==True has already occured
    d['num_datapoints'] = row['i'].count()
    d['peak_area'] = row['i'].sum()
    idx = row['i'].idxmax()
    d['peak_height'] = row.loc[idx,'i']
    d['mz_centroid'] = sum(row['i']*row['mz'])/d['peak_area']
    d['rt_peak'] = row.loc[idx,'rt']
    return pd.Series(d)

def get_sample_ms1_data(node_atlas: pd.DataFrame, sample_files: List[str], mz_ppm_tolerance: int, peak_height_min,num_datapoints_min):
    """Collect MS1 data from experimental sample data using node attributes."""
    ms1_data = []
    for f in sample_files:
        node_atlas.sort_values('mz',inplace=True)
        node_atlas['ppm_tolerance'] = mz_ppm_tolerance
        node_atlas['extra_time'] = 0
        node_atlas['group_index'] = ft.group_consecutive(node_atlas['mz'].values[:],
                                             stepsize=mz_ppm_tolerance,
                                             do_ppm=True)
        d = ft.get_atlas_data_from_file(f,node_atlas,desired_key='ms1_neg')
        d = d[d['in_feature']==True].groupby('label').apply(calculate_ms1_summary).reset_index()
        # d = ft.calculate_ms1_summary(d, feature_filter=True).reset_index(drop=True)
        d['lcmsrun_observed'] = f
        ms1_data.append(d)
    ms1_data = pd.concat(ms1_data)
    ms1_data = ms1_data[ms1_data['peak_height']>peak_height_min]
    ms1_data = ms1_data[ms1_data['num_datapoints']>num_datapoints_min]
    # ms1_data = ms1_data.astype({'label': 'string', 'lcmsrun_observed': 'string'})
    ms1_data.reset_index(inplace=True,drop=True)
    return ms1_data


def get_sample_ms2_data(sample_files: List[str],merged_node_data,msms_score_min,msms_matches_min,mz_ppm_tolerance,frag_mz_tolerance) -> pd.DataFrame:
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
    ms2_data = remove_unnecessary_ms2_data(ms2_data,merged_node_data,ppm_filter=mz_ppm_tolerance)
    ms2_data['nl_spectrum'] = ms2_data.apply(lambda x: np.array([x.mdm_mz_vals, x.mdm_i_vals]), axis=1)
    ms2_data['original_spectrum'] = ms2_data.apply(lambda x: np.array([x.original_mz_vals, x.original_i_vals]), axis=1)

    nl_data_spectra = ms2_data['nl_spectrum'].tolist()
    nl_ref_spectra = merged_node_data['spectrum_nl_spectra'].tolist()

    or_data_spectra = ms2_data['original_spectrum']
    or_ref_spectra = merged_node_data['spectrum_original_spectra'].tolist()

    data_pmzs = ms2_data['precursor_mz'].tolist()
    ref_pmzs = merged_node_data['precursor_mz'].tolist()


    discretized_spectra = blink.discretize_spectra(nl_data_spectra, nl_ref_spectra, data_pmzs,  ref_pmzs,
                                            bin_width=0.001, tolerance=frag_mz_tolerance, intensity_power=0.5, trim_empty=False, remove_duplicates=False, network_score=False)
    nl_blink = do_blink(discretized_spectra,ms2_data,merged_node_data,msms_score_min,msms_matches_min,mz_ppm_tolerance)

    discretized_spectra = blink.discretize_spectra(or_data_spectra, or_ref_spectra, data_pmzs,  ref_pmzs,
                                            bin_width=0.001, tolerance=frag_mz_tolerance, intensity_power=0.5, trim_empty=False, remove_duplicates=False, network_score=False)
    or_blink = do_blink(discretized_spectra,ms2_data,merged_node_data,msms_score_min,msms_matches_min,mz_ppm_tolerance)

    ms2_scores = merge_or_nl_blink(nl_blink,or_blink)
    ms2_scores = pd.merge(ms2_scores,merged_node_data[['node_id']],left_on='ref',right_index=True,how='left')
    ms2_scores = pd.merge(ms2_scores,ms2_data[['filename']],left_on='query',right_index=True,how='left')
    ms2_scores.rename(columns={'filename':'lcmsrun_observed'},inplace=True)
    ms2_scores = ms2_scores.astype({'node_id': 'string', 'lcmsrun_observed': 'string'})


    return ms2_scores