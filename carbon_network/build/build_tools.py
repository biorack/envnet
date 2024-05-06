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


sys.path.insert(0,'/global/homes/b/bpb/repos/blink')
import blink


# temp = ms2_df['original_spectrum']
# temp['mod_spectrum'] = temp.apply(lambda x: x.T)
# temp = temp['mod_spectrum'].to_dict()


def do_remblink_networking(query,ref,
                           mass_diffs=[0, 14.0157, 12.000, 15.9949, 2.01565, 27.9949, 26.0157, 18.0106, 30.0106, 42.0106, 1.9792, 17.00284, 24.000, 13.97925, 1.00794, 40.0313],
                           spectra_attr='msv'):
    polarity = 'negative'
    # import pickle as pickle
    from joblib import load
    model_file = '/global/homes/b/bpb/repos/blink/models/mdm_negative_random_forest.joblib'
    if not os.path.isfile(model_file):
        model_file = '/global/homes/b/bpb/repos/blink/models/{}_random_forest.joblib'.format(polarity)
    print('Using %s'%model_file)
    with open(model_file, 'rb') as out:
        regressor = load(out)

    query_spectra = query[spectra_attr].tolist()
    query_precursor_mzs = query.precursor_mz.tolist()

    ref_spectra = ref[spectra_attr].tolist()
    ref_precursor_mzs = ref.precursor_mz.tolist()

    d_specs = blink.discretize_spectra(query_spectra,  ref_spectra, query_precursor_mzs, ref_precursor_mzs, intensity_power=0.5, bin_width=0.001, tolerance=0.01,network_score=True,mass_diffs=mass_diffs)
    scores = blink.score_sparse_spectra(d_specs)
    stacked_scores, stacked_counts = blink.stack_network_matrices(scores)
    rem_scores, predicted_rows = blink.rem_predict(stacked_scores, scores, regressor,min_predicted_score=0.0001)
    score_rem_df, matches_rem_df = blink.make_rem_df(rem_scores, stacked_counts, predicted_rows, mass_diffs=mass_diffs)
    rem_df = pd.concat([score_rem_df,matches_rem_df],axis=1)

    rem_df = rem_df.sparse.to_dense()

    # match_cols = [m for m in rem_df.columns if 'matches' in m]
    # rem_df['matches'] = rem_df[match_cols].max(axis=1)

    # cols = ['formula', 'C', 'H', 'O', 'mw']
    # rem_df = pd.merge(rem_df,query[cols].add_suffix('_query'),left_on='query',right_index=True,how='inner')
    # rem_df = pd.merge(rem_df,ref[cols].add_suffix('_ref'),left_on='ref',right_index=True,how='inner')
    # # rem_df.sort_values('matches',inplace=True,ascending=False)
    # # rem_df = rem_df[rem_df['formula_ref']!=rem_df['formula_query']]

    # rem_df.sort_values('rem_predicted_score',inplace=True,ascending=False)
    # # rem_df.drop_duplicates(['formula_ref','formula_query'],inplace=True)

    # for e in ['mw','C','H','O']:
    #     rem_df['%s_diff'%e] = rem_df['%s_ref'%e] - rem_df['%s_query'%e]
    return rem_df

def get_p2d2(deltas,mz_tol=0.002):
    ref_filename = '/global/cfs/cdirs/metatlas/projects/spectral_libraries/deduplicated_merged_library/for_scn_ref_p2d2.pkl'
    ref2_filename = '/global/cfs/cdirs/metatlas/projects/spectral_libraries/deduplicated_merged_library/for_scn_ref2_p2d2.pkl'
    if os.path.isfile(ref_filename) and os.path.isfile(ref2_filename):
        ref = pd.read_pickle(ref_filename)
        ref2 = pd.read_pickle(ref2_filename)
    else:
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

        ref_nl = group_duplicates(out,'original_p2d2_index')#,make_string=False)#,precision={'i':0,'mz':4,'rt':2})
        ref_nl['nl_spectrum'] = ref_nl.apply(lambda x: np.asarray([x['mz'],x['i']]),axis=1)
        ref_nl['nl_spectrum_num_ions'] = ref_nl['mz'].apply(lambda x: len(x))
        ref_nl.drop(columns=['mz','i'],inplace=True)
        ref_nl = ref_nl[ref_nl['nl_spectrum_num_ions']>2]
        ref2 = pd.merge(ref,ref_nl,on='original_p2d2_index',how='inner')
        ref2.reset_index(inplace=True,drop=True)
        ref.to_pickle(ref_filename)
        ref2.to_pickle(ref2_filename)
    return ref,ref2



def blink_score(query, #this has both the original and neutral loss spectra
                ref, # this is for original spectra in the reference
                ref2, # this is for neutral loss spectra in the reference
                query_spectrum_key='original_spectrum',
                ref_spectrum_key='spectrum',
                precursor_mz_key='precursor_mz',
                min_matches=5,
                min_score=0.7,
                override_matches=20):
    """
    query: dataframe with a spectrum and precursor_mz
    ref: dataframe with a spectrum and precursor_mz
    score the query against the ref
    """
    
    query_spec_nl = query['nl_spectrum'].tolist()
    query_spec_original = query['original_spectrum'].tolist()
    query_pmz = query['precursor_mz'].tolist()

    ref_spec_nl = ref2['nl_spectrum'].tolist()
    ref_pmz_nl = ref2['precursor_mz'].tolist()
    
    ref_spec_original = ref['spectrum'].tolist()
    ref_pmz_original = ref['precursor_mz'].tolist()

    polarity = 'negative'
    # import pickle as pickle
    from joblib import load

    with open('/global/homes/b/bpb/repos/blink/models/{}_random_forest.joblib'.format(polarity), 'rb') as out:
        regressor = load(out)

    mass_diffs = [0, 14.0157, 12.000, 15.9949, 2.01565, 27.9949, 26.0157, 18.0106, 30.0106, 42.0106, 1.9792, 17.00284, 24.000, 13.97925, 1.00794, 40.0313]#, 43.993]

    print('Calculating REM-BLINK on Neutral Loss Spectra')
    d_specs = blink.discretize_spectra(query_spec_nl,  ref_spec_nl, query_pmz, ref_pmz_nl, intensity_power=0.5, bin_width=0.001, tolerance=0.01,network_score=True,mass_diffs=mass_diffs)
    scores = blink.score_sparse_spectra(d_specs)
    stacked_scores, stacked_counts = blink.stack_network_matrices(scores)
    rem_scores, predicted_rows = blink.rem_predict(stacked_scores, scores, regressor,min_predicted_score=0.0001)
    score_rem_df, matches_rem_df = blink.make_rem_df(rem_scores, stacked_counts, predicted_rows, mass_diffs=mass_diffs)
    rem_df = pd.concat([score_rem_df,matches_rem_df],axis=1)
    rem_df = rem_df.sparse.to_dense()
    rem_df_nl = rem_df[rem_df['rem_predicted_score']>0.01]

    print('Calculating REM-BLINK on Original Spectra')
    d_specs = blink.discretize_spectra(query_spec_original,  ref_spec_original, query_pmz, ref_pmz_original, intensity_power=0.5, bin_width=0.001, tolerance=0.01,network_score=True,mass_diffs=mass_diffs)
    scores = blink.score_sparse_spectra(d_specs)
    stacked_scores, stacked_counts = blink.stack_network_matrices(scores)
    rem_scores, predicted_rows = blink.rem_predict(stacked_scores, scores, regressor,min_predicted_score=0.0001)
    score_rem_df, matches_rem_df = blink.make_rem_df(rem_scores, stacked_counts, predicted_rows, mass_diffs=mass_diffs)
    rem_df = pd.concat([score_rem_df,matches_rem_df],axis=1)
    rem_df = rem_df.sparse.to_dense()
    rem_df_original = rem_df[rem_df['rem_predicted_score']>0.01]

    print('Calculating MZ-BLINK on Neutral Loss Spectra')
    d_specs = blink.discretize_spectra(query_spec_nl,  ref_spec_nl, query_pmz, ref_pmz_nl)
    scores = blink.score_sparse_spectra(d_specs)
    filtered_scores = blink.filter_hits(scores,min_score=0.7,min_matches=3,override_matches=20)
    out_mat = blink.reformat_score_matrix(filtered_scores)
    mz_df_nl = blink.make_output_df(out_mat)
    for c in mz_df_nl.columns:
        mz_df_nl[c] = mz_df_nl[c].sparse.to_dense()
        
    print('Calculating MZ-BLINK on Original Spectra')
    d_specs = blink.discretize_spectra(query_spec_original,  ref_spec_original, query_pmz, ref_pmz_original)
    scores = blink.score_sparse_spectra(d_specs)
    filtered_scores = blink.filter_hits(scores,min_score=0.7,min_matches=3,override_matches=20)
    out_mat = blink.reformat_score_matrix(filtered_scores)
    mz_df_original = blink.make_output_df(out_mat)
    for c in mz_df_original.columns:
        mz_df_original[c] = mz_df_original[c].sparse.to_dense()
          
    # out_df = pd.merge(out_df,df.add_suffix('_query'),left_on='query',right_index=True)
    # out_df = pd.merge(out_df,ref.add_suffix('_ref'),left_on='ref',right_index=True)
    return mz_df_original,mz_df_nl,rem_df_original,rem_df_nl

                            
def run_wave_workflow(f,
                      deltas,
                      do_buddy=True,
                      mz_tol=0.002,
                      similarity_cutoff=0.8,
                      isolation_tol=0.5,
                      min_intensity_ratio=2,
                      fraction_required=3,
                      min_rt=1,
                      max_rt=7,
                      my_polarity='negative',
                      ft_icr_filter=None):
    
    # this will calculate the neutral loss spectra and the recalculated precursor m/z
    buddy_pickle_filename = f.replace('.h5','_buddy.pkl')
    if os.path.isfile(buddy_pickle_filename):
        df = pd.read_pickle(buddy_pickle_filename)
    else:
        df = build_spectra_dataframe(f,deltas,fraction_required=fraction_required,isolation_tol=isolation_tol,mz_tol=mz_tol,min_rt=min_rt,max_rt=max_rt)
        if df is None:
            return None
        df.reset_index(inplace=True,drop=True)
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

        if do_buddy==True:
            result,msb_engine = run_buddy(df,ionization_mode=my_polarity,spectrum_key='nl_spectrum')

            result.rename(columns={'adduct':'assumed_adduct','formula_rank_1':'predicted_formula'},inplace=True)
            cols = [c for c in result.columns if 'rank_' in c]
            result.drop(columns=cols,inplace=True)

            df = pd.merge(df,result.drop(columns=['mz','rt']),left_index=True,right_on='identifier',how='inner')
            formula_props = get_formula_props(df,formula_key='predicted_formula')
            df = pd.merge(df,formula_props,left_on='predicted_formula',right_on='formula',how='left')
            df.drop(columns=['identifier','formula'],inplace=True)
            df.reset_index(inplace=True,drop=True)
            df.to_pickle(buddy_pickle_filename)

    if (do_buddy==True) & (ft_icr_filter is not None):
        df = df[df['predicted_formula'].isin(ft_icr_filter['formula'])]
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
    ms2_df = pd.read_hdf(file,file_key)
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
    ref = pd.read_hdf(f,file_key)
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
