import pandas as pd
from msbuddy import Msbuddy, MsbuddyConfig
from msbuddy.base import MetaFeature, Spectrum

from pyteomics import mgf
import pandas as pd
from rdkit.Chem.rdMolDescriptors import CalcMolFormula, CalcExactMolWt
from rdkit.Chem import MolFromSmiles
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import re
import os

import multiprocessing as mp
import sys

from collections import defaultdict


from pathlib import Path
module_path = os.path.join(Path(__file__).parents[2])
sys.path.insert(0, module_path)

import blink.blink as blink
from metatlas.metatlas.io import feature_tools as ft

                            
def run_workflow(f,
                 deltas,
                 do_buddy=True,
                 elminate_duplicate_spectra=True,
                 mz_tol=0.002,
                 similarity_cutoff=0.8,
                 isolation_tol=0.5,
                 min_intensity_ratio=2,
                 fraction_required=3,
                 min_rt=1,
                 max_rt=7,
                 my_polarity='negative'):
    
    # this will calculate the neutral loss spectra and the recalculated precursor m/z

    df = build_spectra_dataframe(f,deltas,fraction_required=fraction_required,isolation_tol=isolation_tol,mz_tol=mz_tol,min_rt=min_rt,max_rt=max_rt)
    if df is None:
        print('There is not any data in the dataframe')
        return None
    df.reset_index(inplace=True,drop=True)
    
    if elminate_duplicate_spectra:
        df = eliminate_duplicate_spectra(df,deltas,mz_tol=mz_tol,similarity_cutoff=similarity_cutoff,min_intensity_ratio=min_intensity_ratio)

    df.reset_index(inplace=True,drop=True)
    df.index.name = 'temp_index'
    df.reset_index(inplace=True,drop=False)
    cols = ['temp_index', 'count', 'precursor_mz', 
        'sum_frag_intensity', 'max_frag_intensity', 'obs',
        'isolated_precursor_mz', 'rt', 'basename']
    g_cols = ['basename','rt']

    g = df[cols].groupby(g_cols).agg({'temp_index': 'count', 'precursor_mz': lambda x: list(x)})
    g.rename(columns={'precursor_mz':'coisolated_precursor_mz_list','temp_index':'coisolated_precursor_count'},inplace=True)
    df = pd.merge(df,g,on=g_cols,how='left')

    # idx_1 = df['coisolated_precursor_count']==1
    # idx_gt1 = df['coisolated_precursor_count']>1
    df['mdm_mz_vals'] = None
    df['mdm_i_vals'] = None
    df['original_mz_vals'] = None
    df['original_i_vals'] = None
    df['original_mz_vals'] = df['original_spectrum'].apply(lambda x: x[0].astype(float))
    df['original_i_vals'] = df['original_spectrum'].apply(lambda x: x[1].astype(float))
    df['mdm_mz_vals'] = df['nl_spectrum'].apply(lambda x: x[0].astype(float))
    df['mdm_i_vals'] = df['nl_spectrum'].apply(lambda x: x[1].astype(float))
    
    df['temp_spectrum'] = df.apply(lambda x: np.vstack([x['mdm_mz_vals'],x['mdm_i_vals']]),axis=1)
    
    cols = ['temp_index', 'count', 'nl_spectrum',
        'sum_frag_intensity', 'max_frag_intensity', 'obs', 'original_spectrum',
        'basename', 'temp_spectrum',
        'coisolated_precursor_mz_list']

    if do_buddy:
        result,msb_engine = run_buddy(df,ionization_mode=my_polarity,spectrum_key='temp_spectrum')

        result.rename(columns={'adduct':'assumed_adduct','formula_rank_1':'predicted_formula'},inplace=True)
        cols = [c for c in result.columns if 'rank_' in c]
        result.drop(columns=cols,inplace=True)

        df = pd.merge(df,result.drop(columns=['mz','rt']),left_index=True,right_on='identifier',how='inner')
        # formula_props = get_formula_props(df,formula_key='predicted_formula')
        # df = pd.merge(df,formula_props,left_on='predicted_formula',right_on='formula',how='left')
        df.drop(columns=['identifier'],inplace=True)
        
        cols += ['assumed_adduct', 'buddy_spectrum']

    df.drop(columns=cols,inplace=True)
    df.reset_index(inplace=True,drop=True)

    return df

                            
def build_spectra_dataframe(f,deltas,
                            min_frag_intensity=1e4,
                            isolation_tol=0.5,
                            mz_tol=0.002,
                            fraction_required=3,
                            do_parallel=True,
                            max_cores=15,
                            filter_percent=0.001,
                            min_rt=1,
                            max_rt=7):
    ms2_df = group_ms2file_spectra(f,deltas,isolation_tol=isolation_tol,mz_tol=mz_tol,do_parallel=True,fraction_required=fraction_required,min_rt=min_rt,max_rt=max_rt)
    if ms2_df is None:
        return None
    ms2_df['filename'] = f
    orig_spectra = get_original_spectra(f,filter_percent=0.001)

    orig_spectra['rt'] = orig_spectra['rt'].astype(float)
    orig_spectra['rt'] = orig_spectra['rt'].round(3)

    ms2_df['rt'] = ms2_df['rt'].astype(float)
    ms2_df['rt'] = ms2_df['rt'].round(3)
    ms2_df = pd.merge(ms2_df,orig_spectra.add_prefix('original_'),left_on=['filename','rt'],right_on=['original_filename','original_rt'],how='left')
    cols = ['original_rt','original_precursor_mz','original_precursor_intensity','original_filename']
    ms2_df.drop(columns=cols,inplace=True)
    ms2_df['basename'] = ms2_df['filename'].apply(lambda x: os.path.basename(x))

    ms2_df = ms2_df[ms2_df['max_frag_intensity']>min_frag_intensity]
    return ms2_df


def get_neutral_loss_spectra(ms2_df,deltas,mz_tol=0.002):
    original_spectra = []
    for i,row in ms2_df.iterrows():
        s = row['original_spectrum']
        zeros = np.zeros((s.shape[1]))
        my_index = zeros + i
        neutral_losses = row['precursor_mz'] - s[0]
        original_spectra.append(np.vstack([my_index,neutral_losses,s]).T)
    # temp = [tt.T for tt in temp]
    if len(original_spectra)==0:
        return None
    temp= np.concatenate(original_spectra)

    from scipy import interpolate
    query_points = deltas['mass'].values
    query_indices = range(len(query_points))
    f = interpolate.interp1d(query_points,query_indices,kind='nearest',bounds_error=False,fill_value='extrapolate')
    new_indices = f(temp[:,1]).astype(int)

    d = abs(temp[:,1] - query_points[new_indices])
    idx = d<mz_tol

    temp = pd.DataFrame(temp[idx,:],columns=['ms2_df_index','neutral_loss','mz','intensity'])
    temp['theoretical_neutral_loss'] = new_indices[idx]
    temp = pd.merge(temp,deltas,left_on='theoretical_neutral_loss',right_index=True)

    # using the delta dataframe from the MDM paper
    d = deltas['difference'].apply(lambda x: x.split(','))
    d = pd.DataFrame(d)
    d.fillna(0,inplace=True)

    #using the old delta dataframe I made
    # d = deltas['difference'].apply(lambda x: dict([(dd[1],int(dd[0])) for dd in [d.split(':') for d in x.split(',')]])).tolist()
    # d = pd.DataFrame(d)
    # d.fillna(0,inplace=True)

    # temp = pd.merge(temp,d,left_on='theoretical_neutral_loss',right_index=True)

    temp.sort_values('ms2_df_index',inplace=True)
    temp.reset_index(inplace=True,drop=True)
    nl_mat = np.zeros((deltas.shape[0],ms2_df.shape[0]))
    nl_mat[temp['theoretical_neutral_loss'].values.astype(int),temp['ms2_df_index'].values.astype(int)] = temp['intensity'].values
    
    return nl_mat


def eliminate_duplicate_spectra(ms2_df,deltas,mz_tol=0.002,similarity_cutoff=0.8,min_intensity_ratio=2):
    # This should keep the spectra with the highest sum of fragment intensities
    # and eliminate the rest that are similar to it
    ms2_df.sort_values('sum_frag_intensity',ascending=False,inplace=True)
    ms2_df.reset_index(inplace=True,drop=True)
    nl_mat = get_neutral_loss_spectra(ms2_df,deltas,mz_tol=mz_tol)
    if nl_mat is None:
        return ms2_df
    similarity_matrix = cosine_similarity(nl_mat.T)
    pmz_diff = abs(np.subtract.outer(ms2_df['precursor_mz'].values,ms2_df['precursor_mz'].values))
    frag_intensity = ms2_df['sum_frag_intensity'].values
    intensity_diff = -1*np.subtract.outer(frag_intensity,frag_intensity)
    intensity_diff = intensity_diff / frag_intensity[:,None]
    idx = np.triu_indices(pmz_diff.shape[0],k=0)
    pmz_diff[idx] = 1000
    r,c = np.argwhere((pmz_diff<mz_tol) & ((intensity_diff>min_intensity_ratio) | (similarity_matrix>similarity_cutoff))).T
    ms2_df = ms2_df[~ms2_df.index.isin(r)]
    ms2_df.reset_index(inplace=True,drop=True)
    return ms2_df


def calc_blink_all_by_all(df,spectrum_key='spectrum',precursor_mz_key='precursor_mz',return_matrix=False):
    spec1 = df[spectrum_key].tolist()
    pmz1 = df[precursor_mz_key].tolist()

    d_specs = blink.discretize_spectra(spec1, spec1, pmz1, pmz1,network_score=True)
    scores = blink.score_sparse_spectra(d_specs)
    if return_matrix==True:
        return scores
    # filtered_scores = blink.filter_hits(scores,min_score=0.01)
    out_mat = blink.reformat_score_matrix(scores)

    out_df = blink.make_output_df(out_mat)
    out_df = out_df[out_df['query']<out_df['ref']]
    return out_df

def group_duplicates(df,group_col,make_string=False,precision={'i':0,'mz':4,'rt':2}):
    """
    takes in a list of grouping columns and turns the rest into arrays
    """

    all_cols = np.asarray(df.columns)
    #get the index of the grouping term as array
    idx_group = np.argwhere(all_cols == group_col).flatten()
    #get the indices of all other terms as array
    idx_list = np.argwhere(all_cols != group_col).flatten()
    cols = all_cols[idx_list]

    # create a sorted numpy array (sorted by column=group_col)
    a = df.sort_values(group_col).values.T

    #get the indices of the first instance of each unique identifier
    ukeys, index = np.unique(a[idx_group,:],return_index=True)

    #split the other rows of the array into separate arrays using the
    #unique index
    arrays = np.split(a[idx_list,:],index[1:],axis=1)

    #make a list of dicts with column headings as keys
    #if there are not multiple items then return value
    #If there are multiple items then return list

#     ucpds = [dict([(c,aa) if len(aa)>1 else (c,aa[0]) for c,aa in zip(cols,a)]) for a in arrays ]
    ucpds = [dict([(c,aa) for c,aa in zip(cols,a)]) for a in arrays ]

    #make a dataframe from the list of dicts
    df2 = pd.DataFrame(ucpds,index=ukeys)

    #make strings of array columns if you want to save it in anything useful
    if make_string==True:
        for c in cols:
#             df2[c] = df2[c].apply(lambda x: np.array2string(x, precision=5, separator=','))
            if c in list(precision.keys()):
                pre_str = '{:.%df}'%precision[c]
            else:
                pre_str = '{:.4f}'
            df2[c] = df2[c].apply(lambda x: [pre_str.format(n) for n in x.tolist()])
#             df2[c] = df2[c].apply(lambda x: str(x.tolist()))

    df2.index = df2.index.set_names(group_col)
    df2.reset_index(inplace=True)

    #return dataframe
    return df2



def spectra_agg_func(x):
    d = {}
    d['count'] = x['value'].count()
    d['precursor_mz'] = sum(x['i']*x['value']) / sum(x['i'])
    idx = np.argsort(x['mz'].values)
    d['nl_spectrum'] = np.asarray([x['mz'].tolist(),x['i'].to_list()])
    d['nl_spectrum'] = d['nl_spectrum'][:,idx]
    d['sum_frag_intensity'] = sum(x['i'])
    d['max_frag_intensity'] = max(x['i'])
    d['obs'] = x['variable'].values
    d['obs'] = d['obs'][idx]
    d['isolated_precursor_mz'] = x['precursor_mz'].to_list()[0]
    d['rt'] = x['rt'].to_list()[0]
    return pd.Series(d, index=d.keys())

def split_each_spectrum(grouped_ions,mz_tol,fraction_required):
    grouped_ions.sort_values('value',ascending=True,inplace=True)
    # split each group when they differ by more than m/z tolerance
    grouped_ions['cluster'] = (grouped_ions['value'].diff()>mz_tol).cumsum()
    pure_spectra = grouped_ions.groupby('cluster').apply(spectra_agg_func)
    if fraction_required<1:
        pure_spectra = pure_spectra[pure_spectra['count']>=(len(deltas)*fraction_required)]
    else:
        pure_spectra = pure_spectra[pure_spectra['count']>=fraction_required]
    return pure_spectra

def group_ms2file_spectra(file,deltas,isolation_tol=2.5,mz_tol=0.002,file_key='ms2_neg',min_rt=1,max_rt=7,do_parallel=True,max_cores=15,fraction_required=3):
    
    if file.endswith('h5'):
        ms2_df = pd.read_hdf(file,file_key)
    elif file.endswith('mzML') or file.endswith('mzml'):
        ms2_df = ft.df_container_from_mzml_file(file, file_key)
        
    ms2_df = ms2_df[ms2_df['rt']>min_rt]
    ms2_df = ms2_df[ms2_df['rt']<max_rt]

    ms2_df.columns = [c.lower() for c in ms2_df.columns]

    # convert to dictionary if it is a dataframe
    is_dataframe = isinstance(deltas, pd.DataFrame)
    # is_dictionary = isinstance(deltas, dict)
    if is_dataframe==True:
        deltas = deltas.set_index('difference')['mass'].to_dict()

    # iterate through the deltas and shift them up towards the expected precursor m/z
    temp = {}
    for k,v in deltas.items():
        temp[k] = ms2_df['mz'] + v
    temp = pd.DataFrame(temp)
    ms2_df = pd.concat([ms2_df,temp],axis=1)
    
    ms2_df = ms2_df.melt(id_vars=['mz','i','rt','precursor_mz'],value_vars=deltas.keys())
    ms2_df = ms2_df[abs(ms2_df['value']-ms2_df['precursor_mz'])<isolation_tol]
    
    temp = [(d,mz_tol,fraction_required) for _,d in ms2_df.groupby('rt')]
    
    if do_parallel==True:
        with mp.Pool(min(max_cores,mp.cpu_count())) as pool:
            out = pool.starmap(split_each_spectrum, temp)
    else:
        out = []
        for tt in temp:
            out.append(split_each_spectrum(tt[0],tt[1],tt[2]))
            
    # ms2_df = ms2_df.groupby('rt').apply(split_each_spectrum)
    if len(out)>1:
        ms2_df = pd.concat(out)
    else:
        return None

    ms2_df.sort_values('precursor_mz',inplace=True)
    ms2_df.reset_index(inplace=True,drop=True)
    
    return ms2_df


def group_dict_by_value(original_dict):
    # Grouped dictionary
    grouped_dict = defaultdict(list)

    # Group the dictionary by values and combine keys
    for key, value in original_dict.items():
        grouped_dict[value].append(key)

    new_d = {}
    for value, keys in grouped_dict.items():
        keys_str = ' or '.join(keys)
        new_d[keys_str] = value

    return new_d

def nl_builder(ele_deltas,max_allowed=4):
    from rdkit import Chem
    import itertools
        
    L = [list(np.arange(0,max_allowed+1))] # this value is how many copies you want to multiply it up to
    L = L*len(ele_deltas.keys()) # this value is how many deltas you have to combine
    all_combinations = list(itertools.product(*L))
    pse = Chem.GetPeriodicTable()
    mC= pse.GetMostCommonIsotopeMass('C')
    mH= pse.GetMostCommonIsotopeMass('H')
    mO= pse.GetMostCommonIsotopeMass('O')
    for k,v in ele_deltas.items():
        ele_deltas[k] = mC*v[0] + mH*v[1] + mO*v[2]
    ele_delta_keys = list(ele_deltas.keys())
    new_deltas = {}
    for combo in all_combinations:
        new_delta_name = []
        new_delta_value = 0
        for i,nl in enumerate(ele_delta_keys):
            if combo[i]!=0:
                new_delta_name.append('%d:%s'%(combo[i],nl))
                new_delta_value += combo[i] * ele_deltas[nl]
        if len(new_delta_name)>0:
            new_deltas[','.join(new_delta_name)] = np.round(new_delta_value,4)
    new_deltas = group_dict_by_value(new_deltas)
    return new_deltas

def filter_spectra_by_percent(x,p=0.01):
    # Find the maximum intensity value
    max_intensity = np.max(x[1])

    # Set the threshold to one percent of the maximum intensity
    threshold = p * max_intensity

    # Find the indices where intensity values are above the threshold
    above_threshold_indices = np.where(x[1] > threshold)[0]

    # Filter the original array based on the above-threshold indices
    filtered_array = x[:, above_threshold_indices]

    return filtered_array


def get_original_spectra(f,file_key='ms2_neg',filter_percent=0.01):
    
    if f.endswith('h5'):
        ref = pd.read_hdf(f, file_key)
    elif f.endswith('mzML') or f.endswith('mzml'):
        ref = ft.df_container_from_mzml_file(f, file_key)
        
    ref.columns = [c.lower() for c in ref.columns]
    ref.drop(columns=['collision_energy'],inplace=True)
    ref = ref[pd.notna(ref['precursor_mz'])]
    # query = query[query['i']>1e4]
    ref = group_duplicates(ref,'rt',make_string=False,precision={'i':0,'mz':4,'rt':2})
    ref['precursor_intensity'] = ref['precursor_intensity'].apply(lambda x: x[0])
    ref['precursor_mz'] = ref['precursor_mz'].apply(lambda x: x[0])
    ref['spectrum'] = ref.apply(lambda x: np.asarray([x['mz'],x['i']]),axis=1)
    ref['spectrum'] = ref['spectrum'].apply(lambda x: filter_spectra_by_percent(x,p=filter_percent))
    ref.reset_index(inplace=True,drop=True)
    drop_cols = []
    for c in ref.columns:
        if c in ['mz','i','polarity']:
            drop_cols.append(c)
    ref.drop(columns=drop_cols,inplace=True)
    ref['filename'] = f
    return ref


def formula_to_dict(f):
    import re
    p = re.findall(r'([A-Z][a-z]?)(\d*)', f)
    d = {k.lower():v for k,v in p}
    for k,v in d.items():
        if v == '':
            d[k] = 1
        else:
            d[k] = int(v)
    return d


def run_formula_calcs(f):
    """
    the addition elemental ratios are from Anal. Chem. 2018, 90, 6152−6160
    """
    m = {}
    elements = ['c','h','n','o','s','p']
    for e in elements:
        m[e] = 0
        if e in f.keys():
            m[e] = f[e]
    if m['c']==0:
        return None
    if m['p']==0:
        n_to_p = None
    else:
        n_to_p = m['n']/m['p']
        
    out = {'dbe':calc_dbe(m),
           'dbe_ai':calc_dbe_ai(m),
           'dbe_ai_mod':calc_dbe_ai_mod(m),
           'ai_mod':modified_aromaticity_index(m),
           'ai':aromaticity_index(m),
           'nosc':calc_nosc(m),
           'h_to_c':m['h']/m['c'],
           'o_to_c':m['o']/m['c'],
           'n_to_c':m['n']/m['c'],
           'p_to_c':m['p']/m['c'],
           'n_to_p':n_to_p,
           'c':m['c'],
           'h':m['h'],
           'o':m['o'],
           'n':m['n'],
           's':m['s'],
           'p':m['p']}
    return out
    
    
def calc_dbe(m):
    """
    D’Andrilli, J., Dittmar, T., Koch, B. P., Purcell, J. M., Marshall, A. G., & Cooper, W. T. (2010). Comprehensive characterization of marine dissolved organic matter by Fourier transform ion cyclotron resonance mass spectrometry with electrospray and atmospheric pressure photoionization. Rapid Communications in Mass Spectrometry, 24(5), 643–650. doi:10.1002/rcm.4421 
    """
    dbe = 1 + m['c'] - m['h']/2 + m['n']/2
    return dbe


def calc_dbe_ai(m):
    """
    From mass to structure: an aromaticity index for high-resolution mass data of natural organic matter
    B. P. Koch and T. Dittmar
    Rapid Commun. Mass Spectrom. 2006, 20, 926–932.
    DOI: 10.1002/rcm.2386
    """
    dbe_ai = 1 + m['c'] - m['o'] - m['s'] - m['h']/2 - m['n']/2 - m['p']/2
    return dbe_ai

def calc_dbe_ai_mod(m):
    """
    From mass to structure: an aromaticity index for high-resolution mass data of natural organic matter
    B. P. Koch and T. Dittmar
    Rapid Commun. Mass Spectrom. 2006, 20, 926–932.
    DOI: 10.1002/rcm.2386
    """
    dbe_ai_mod = 1 + m['c'] - m['o']/2 - m['s'] - m['h']/2 - m['n']/2 - m['p']/2
    return dbe_ai_mod


def modified_aromaticity_index(m):
    """
    From mass to structure: an aromaticity index for high-resolution mass data of natural organic matter
    B. P. Koch and T. Dittmar
    Rapid Commun. Mass Spectrom. 2006, 20, 926–932.
    DOI: 10.1002/rcm.2386
    """
    dbe_ai_mod = calc_dbe_ai_mod(m)
    denominator = m['c'] - m['o']/2 - m['n'] - m['s'] - m['p']
    if denominator==0:
        return None
    ai_mod = dbe_ai_mod / denominator
    return ai_mod
    
def aromaticity_index(m):
    """
    From mass to structure: an aromaticity index for high-resolution mass data of natural organic matter
    B. P. Koch and T. Dittmar
    Rapid Commun. Mass Spectrom. 2006, 20, 926–932.
    DOI: 10.1002/rcm.2386
    """
    dbe_ai = calc_dbe_ai(m)
    denominator = m['c'] - m['o'] - m['n'] - m['s'] - m['p']
    if denominator==0:
        return None
    ai = dbe_ai / denominator
    return ai

def calc_nosc(m):
    """
    Molecular Fractionation of Dissolved Organic Matter with Metal Salts
    Thomas Riedel, Harald Biester, and Thorsten Dittmar
    Environmental Science & Technology 2012 46 (8), 4419-4426
    DOI: 10.1021/es203901u
    """
    if m['c']>0:
        nosc = 4 - (4*m['c'] + m['h'] -3*m['n'] - 2*m['o'] - 2*m['s'])/m['c']
    else:
        nosc = np.inf
    return nosc


def specify_mgf_filename(output_dir,my_exp,my_polarity):
    mgf_filename = os.path.join(output_dir,'%s_%s'%(my_exp,my_polarity))
    mgf_filename = os.path.join(mgf_filename,'%s_%s.mgf'%(my_exp,my_polarity))
    if not os.path.isfile(mgf_filename):
        mgf_filename = mgf_filename.replace('.mgf','_MSMS.mgf')
    if os.path.isfile(mgf_filename):
        return mgf_filename
    else:
        return None

def read_mgf(mgf_path: str) -> pd.DataFrame:
    msms_df = []
    with mgf.MGF(mgf_path) as reader:
        for spectrum in reader:
            d = spectrum['params']
            d['spectrum'] = np.array([spectrum['m/z array'],
                                      spectrum['intensity array']])
            if 'precursor_mz' not in d:
                d['precursor_mz'] = d['pepmass'][0]
            else:
                d['precursor_mz'] = float(d['precursor_mz'])
            msms_df.append(d)
    msms_df = pd.DataFrame(msms_df)
    return msms_df

def make_buddy_spec(row,ionization_mode,spectrum_key='spectrum'):
    mf = MetaFeature(mz=row['precursor_mz'],
                     charge=-1 if ionization_mode=='negative' else 1,
                     rt=0,
                     adduct='[M-H]-' if ionization_mode=='negative' else '[M+H]+',
                     ms2=Spectrum(row[spectrum_key][0], row[spectrum_key][1]) if row[spectrum_key].shape[1] > 0 else None,
                     identifier=row.name)
    return mf

def run_buddy(mgf_df,ionization_mode,max_fdr=0.05,spectrum_key='spectrum',
              ms2_tol=0.002,ms1_tol=0.001,
              max_frag_reserved=50,rel_int_denoise_cutoff=0.01):
    # instantiate a MsbuddyConfig object
    msb_config = MsbuddyConfig(# highly recommended to specify
                                ppm=False,
                                ms1_tol=ms1_tol,  # MS1 tolerance in ppm
                                ms2_tol=ms2_tol,  # MS2 tolerance in Da
                                rel_int_denoise_cutoff=rel_int_denoise_cutoff,
                                max_frag_reserved=max_frag_reserved,
                            #    ms_instr='orbitrap',  # supported: "qtof", "orbitrap" and "fticr"
                               # whether to consider halogen atoms FClBrI
                               parallel=True,  # whether to use parallel computing
                               n_cpu=40,
                               halogen=False,
                                batch_size=10000)

    # instantiate a Msbuddy object
    msb_engine = Msbuddy(msb_config)

    msb_engine.data = mgf_df.apply(lambda row: make_buddy_spec(row,ionization_mode=ionization_mode,spectrum_key=spectrum_key),axis=1)

    # annotate molecular formula
    msb_engine.annotate_formula()

    # retrieve the annotation result summary
    result = msb_engine.get_summary()
    r = pd.DataFrame(result)
    
    # Get the explained ions for each top hit
    buddy_spectra = []
    for d in msb_engine.data:
        x = d.ms2_raw.mz_array
        y = d.ms2_raw.int_array
        if len(d.candidate_formula_list)==0:
            buddy_spectra.append(None)
        else:
            h = d.candidate_formula_list[0]
            if h.ms2_raw_explanation is not None:
                idx = [i for i in h.ms2_raw_explanation.idx_array]
                idx = np.asarray(idx)
                buddy_spectra.append(np.asarray([x[idx],y[idx]]))
            else:
                buddy_spectra.append(None)
        
    r['buddy_spectrum'] = buddy_spectra    

    r = r[r['estimated_fdr']<max_fdr]
    r = r[pd.notna(r['buddy_spectrum'])]
    
    return r,msb_engine

def get_formula_props(df,formula_key='formula_rank_1'):
    formula_props = []
    my_formula = df.loc[pd.notna(df[formula_key]),formula_key].unique()
    my_formula = [f for f in my_formula if 'C' in f]
    for f in my_formula:
        d = formula_to_dict(f)
        out = run_formula_calcs(d)
        out['formula'] = f
        formula_props.append(out)
    formula_props = pd.DataFrame(formula_props)
    return formula_props
