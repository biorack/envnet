import networkx as nx
import pandas as pd
import numpy as np
import os
import sys
import glob
from scipy import interpolate
from scipy.stats import ttest_ind

from typing import List, Tuple
from tqdm.notebook import tqdm


from pathlib import Path
module_path = os.path.join(Path(__file__).parents[2])
sys.path.insert(0, module_path)
sys.path.insert(1,os.path.join(module_path,'metatlas'))
sys.path.insert(2,os.path.join(module_path,'blink'))

import json
import requests
import blink as blink
from metatlas.io import feature_tools as ft
from envnet.build.preprocess import run_workflow


def make_output_df(node_data,best_hits,stats_df,filename='output.csv'):
    output = node_data.copy()
    output.set_index('node_id',inplace=True)
    output = output.join(best_hits.set_index('node_id'),rsuffix='_best_hit',how='left')
    output = output.join(stats_df,rsuffix='_stats',how='left')
    output.to_csv(filename)
    return output


def do_basic_stats(ms1_data, files_data, my_groups):

    if 'sample_category' not in files_data.columns:
        return None
    # merge in sample category
    cols = ['filename','sample_category']
    df = pd.merge(ms1_data,files_data[cols],left_on='lcmsrun_observed',right_on='filename',how='inner')
    df = df[pd.notna(df['sample_category'])]
    df.drop(columns=['filename'],inplace=True)

    
    # calcualte group values
    df_agg = df.groupby(['node_id','sample_category'])['peak_area'].agg(['mean', 'median', 'std', lambda x: x.sem()]).reset_index()
    df_agg.columns = ['node_id','sample_category', 'mean', 'median', 'std_dev', 'standard_error']
    df_agg = pd.pivot_table(df_agg,index='node_id',values=['mean','median','std_dev','standard_error'],columns=['sample_category'])
    df_agg.columns = df_agg.columns.map('-'.join)

    # pivot for easy ttest
    d_sample = df.pivot_table(columns='node_id',index=['lcmsrun_observed','sample_category'],values='peak_area',aggfunc='mean',fill_value=300)

    # do ttest on each column
    df_agg['p_value'] = 1
    df_agg['t_score'] = 0
    idx_control = d_sample.index.get_level_values(-1)==my_groups[0]
    idx_treatment = d_sample.index.get_level_values(-1)==my_groups[1]
    for node_id in d_sample.columns:
        control_vals = d_sample.loc[idx_control,node_id].values
        treatment_vals = d_sample.loc[idx_treatment,node_id].values
        t_score,p_val = ttest_ind(control_vals,treatment_vals)
        df_agg.loc[node_id,'p_value'] = p_val
        df_agg.loc[node_id,'t_score'] = t_score

    if 'mean-{}'.format(my_groups[0]) in df_agg.columns:
        df_agg['log2_foldchange'] = np.log2((1+df_agg['mean-{}'.format(my_groups[1])] )/ (1+df_agg['mean-{}'.format(my_groups[0])]))
    else:
        df_agg['log2_foldchange'] = 0
    return df_agg


def graph_to_df(feature='nodes') -> pd.DataFrame:
    G = nx.read_graphml(os.path.join(module_path, 'data/CarbonNetwork.graphml'))
    if feature=='nodes':
        node_data = dict(G.nodes(data=True))
        node_data = pd.DataFrame(node_data).T
        node_data.index.name = 'node_id'
        node_data.reset_index(inplace=True,drop=False)
        return node_data
    elif feature=='edges':
        edge_data = nx.to_pandas_edgelist(G)
        return edge_data


def merge_spectral_data(node_data: pd.DataFrame) -> pd.DataFrame:
    
    original_spectra = blink.open_msms_file(os.path.join(module_path, 'data/original_spectra.mgf'))
    print(original_spectra.shape)
    nl_spectra = blink.open_msms_file(os.path.join(module_path, 'data/nl_spectra.mgf'))
    print(nl_spectra.shape)

        
    original_spectra['node_id'] = original_spectra['original_id'].apply(lambda x: str(float(x)))
    nl_spectra['node_id'] = nl_spectra['original_id'].apply(lambda x: str(float(x)))
    node_data['node_id'] = node_data['original_index'].apply(lambda x: str(float(x)))
    original_spectra.rename(columns={c: c+'_original_spectra' for c in original_spectra.columns if c not in ['node_id']}, inplace=True)
    nl_spectra.rename(columns={c: c+'_nl_spectra' for c in nl_spectra.columns if c not in ['node_id']}, inplace=True)
    
    merged_node_data = pd.merge(node_data, original_spectra, on='node_id')
    merged_node_data = pd.merge(merged_node_data, nl_spectra, on='node_id')
    
    return merged_node_data


def get_files_df(exp_dir, parse_filename=False, groups=None) -> pd.DataFrame:
    if type(exp_dir)==list:
        files = []
        for d in exp_dir:
            files += glob.glob(os.path.join(d,'*NEG*.h5'))
    else:
        files = glob.glob(os.path.join(exp_dir,'*NEG*.h5'))
    files = [f for f in files if not 'exctrl' in f.lower()]
    files = [f for f in files if not 'qc' in f.lower()]
    if type(groups)==dict:
        groups = [groups['control'],groups['treatment']]
    files = pd.DataFrame(files, columns=['filename'])
    if groups is not None:
        group_control = groups[0]
        group_treatment = groups[1]
        idx1 = files['filename'].str.contains(group_control)
        idx2 = files['filename'].str.contains(group_treatment)
        files['sample_category'] = None
        files.loc[idx1,'sample_category'] = group_control
        files.loc[idx2,'sample_category'] = group_treatment

        idx = idx1 | idx2
        files = files[idx]

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
    max_ms1_data.sort_values('peak_area', ascending=False,inplace=True)
    max_ms1_data.drop_duplicates(subset='node_id',inplace=True)
    max_ms1_data = pd.merge(max_ms1_data, node_data[['node_id', 'precursor_mz']], on='node_id',how='left')
    max_ms1_data['ppm_error'] = max_ms1_data.apply(lambda x: ((x.precursor_mz - x.mz_centroid) / x.precursor_mz) * 1000000, axis=1)
    return max_ms1_data

def get_best_ms2_rawdata(ms2_data):
    max_ms2 = ms2_data.sort_values('score',ascending=False).drop_duplicates('node_id')
    max_ms2.drop(columns=['query','ref'],inplace=True)
    max_ms2.reset_index(inplace=True,drop=True)
    return max_ms2

def get_best_ms1_ms2_combined(max_ms1_data,max_ms2_data):
    cols = ['node_id','score','matches','best_match_method','lcmsrun_observed']
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
    if mz_ppm_tolerance is not None:
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

def get_sample_ms1_data(node_atlas: pd.DataFrame, sample_files: List[str], mz_ppm_tolerance: int, peak_height_min, num_datapoints_min):
    """Collect MS1 data from experimental sample data using node attributes."""
    ms1_data = []

    
    for file in tqdm(sample_files, unit='file'):
        if file.endswith('parquet'):
            file = file.replace('.parquet','.h5')
    
        node_atlas.sort_values('mz',inplace=True)
        node_atlas['ppm_tolerance'] = mz_ppm_tolerance
        node_atlas['extra_time'] = 0
        node_atlas['group_index'] = ft.group_consecutive(node_atlas['mz'].values[:],
                                             stepsize=mz_ppm_tolerance,
                                             do_ppm=True)
        
        if file.endswith('mzML') or file.endswith('mzml'):
            d = ft.get_atlas_data_from_mzml(file, node_atlas, desired_key='ms1_neg')
        elif file.endswith('h5'):
            d = ft.get_atlas_data_from_file(file,node_atlas,desired_key='ms1_neg')
        else:
            raise Exception('unrecognized file type')
        
        d = d[d['in_feature']==True].groupby('label').apply(calculate_ms1_summary).reset_index()
        # d = ft.calculate_ms1_summary(d, feature_filter=True).reset_index(drop=True)
        d['lcmsrun_observed'] = file
        ms1_data.append(d)
    ms1_data = pd.concat(ms1_data)
    ms1_data = ms1_data[ms1_data['peak_height']>peak_height_min]
    ms1_data = ms1_data[ms1_data['num_datapoints']>num_datapoints_min]
    ms1_data.rename(columns={'label':'node_id'},inplace=True)
    # ms1_data = ms1_data.astype({'label': 'string', 'lcmsrun_observed': 'string'})
    ms1_data.reset_index(inplace=True,drop=True)
    return ms1_data


def get_sample_ms2_data(sample_files: List[str],merged_node_data,msms_score_min,msms_matches_min,mz_ppm_tolerance,frag_mz_tolerance) -> pd.DataFrame:
    """Collect all MS2 data from experimental sample data and calculate ."""
    
    delta_mzs = pd.read_csv(os.path.join(module_path, 'data/mdm_neutral_losses.csv'))
    
    ms2_scores_out = []
    
    for file in tqdm(sample_files, unit='file'):
        if file.endswith('.parquet'):
            data = pd.read_parquet(file)
        else:
            data = run_workflow(file,
                                delta_mzs,
                                do_buddy = False,
                                elminate_duplicate_spectra = False)
            
        if data.shape[0] == 0:
            continue
        if not 'mdm_mz_vals' in data.columns:
            continue
        # ms2_data = pd.concat(ms2_data).reset_index(drop=True)
        ms2_data = remove_unnecessary_ms2_data(data,merged_node_data,ppm_filter=mz_ppm_tolerance)
        ms2_data.reset_index(drop=True,inplace=True)
        ms2_data['nl_spectrum'] = ms2_data.apply(lambda x: np.array([x.mdm_mz_vals, x.mdm_i_vals]), axis=1)
        ms2_data['original_spectrum'] = ms2_data.apply(lambda x: np.array([x.original_mz_vals, x.original_i_vals]), axis=1)

        nl_data_spectra = ms2_data['nl_spectrum'].tolist()
        nl_ref_spectra = merged_node_data['spectrum_nl_spectra'].tolist()

        or_data_spectra = ms2_data['original_spectrum']
        or_ref_spectra = merged_node_data['spectrum_original_spectra'].tolist()

        data_pmzs = ms2_data['precursor_mz'].tolist()
        ref_pmzs = merged_node_data['precursor_mz'].tolist()

        if len(nl_data_spectra)==0:
            continue
        if len(or_data_spectra)==0:
            continue
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
        ms2_scores['lcmsrun_observed'] = file
        ms2_scores_out.append(ms2_scores)
    if len(ms2_scores_out)==0:
        return None
    # ms2_scores_out = pd.concat(ms2_scores_out)
    # ms2_scores_out.reset_index(drop=True,inplace=True)

    return ms2_scores_out


def query_fasst_peaks(precursor_mz, peaks, database, serverurl="https://fasst.gnps2.org/search", analog=False, precursor_mz_tol=0.05, fragment_mz_tol=0.05, min_cos=0.7):
    """
    database = "gnpsdata_index"
    database = "gnpslibrary"

    spectrum = 54.0348   5
    58.0297 2
    68.0504 2
    81.0455 100
    95.061  6
    138.0663    70
    156.0769    10
    spectrum = spectrum.split('\n')
    spectrum = [x.split() for x in spectrum]

    spectrum = [[float(x[0]), float(x[1])] for x in spectrum]
    precursor = 156.0769
    """
    spectrum_query = {
        "peaks": peaks,
        "precursor_mz": precursor_mz
    }

    params = {
        "query_spectrum": json.dumps(spectrum_query),
        "library": database,
        "analog": "Yes" if analog else "No",
        "pm_tolerance": precursor_mz_tol,
        "fragment_tolerance": fragment_mz_tol,
        "cosine_threshold": min_cos,
    }

    r = requests.post(serverurl, data=params, timeout=50)

    r.raise_for_status()

    return r.json()

def annotate_graphml(output_df, node_data):
    
    G = nx.read_graphml(os.path.join(module_path, 'data/CarbonNetwork.graphml'))

    # get the first node id
    node_id = list(G.nodes())[0]

    # get the names of all node attributes
    node_attributes = list(G.nodes[node_id].keys())

    
    for col in output_df:
        if col in node_attributes:
            continue
        dt = output_df[col].dtype 
        if dt == int or dt == float:
            output_df[col].fillna(0, inplace=True)
        else:
            output_df[col].fillna("", inplace=True)
            
    new_node_attributes = output_df.drop(columns=node_data.drop(columns=['node_id']).columns).to_dict(orient='index')
    new_node_attributes = {str(k):v for k,v in new_node_attributes.items()}

    nx.set_node_attributes(G, new_node_attributes)
    nx.write_graphml(G, 'AnnotatedCarbonNetwork.graphml')
