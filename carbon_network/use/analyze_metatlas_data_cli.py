import sys
import pandas as pd
import networkx as nx
import argparse

import analysis_tools as at
from get_compound_descriptors import calc_descriptor_df


def arg_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument('-1f', '--files_group1', nargs='+', type=str, action='store', required=True)
    parser.add_argument('-1fn', '--files_group1_name', type=str, action='store', required=True)
    
    parser.add_argument('-2f', '--files_group2', nargs='+', type=str, action='store', required=True)
    parser.add_argument('-2fn', '--files_group2_name', type=str, action='store', required=True)
    
    parser.add_argument('-rd', '--raw_data_dir', type=str, action='store', required=True)
    parser.add_argument('-exn', '--exp_name', type=str, action='store', required=True)
    
    # analysis paramter options
    analysis_parameters = parser.add_argument_group()
    
    # parameters for ms1 matching
    analysis_parameters.add_argument('-mzt', '--mz_tol', type=int, action='store', default=5, required=False)
    analysis_parameters.add_argument('-rtr', '--rt_range', type=list, action='store', default=[1, 7], required=False)
    analysis_parameters.add_argument('-pkm', '--pk_height_min', type=float, action='store', default=1e4, required=False)
    analysis_parameters.add_argument('-ndm', '--num_data_min', type=float, action='store', default=10, required=False)
    
    # parameters for ms2 matching
    analysis_parameters.add_argument('-fmzt', '--frag_mz_tol', type=float, action='store', default=0.05, required=False)
    analysis_parameters.add_argument('-msm', '--msms_score_min', type=float, action='store', default=0.5, required=False)
    analysis_parameters.add_argument('-mmm', '--msms_matches_min', type=int, action='store', default=3, required=False)

    return parser


def main(args):
    
    output_filename = f'OUTPUT_{args.exp_name}_{args.files_group1_name}-vs-{args.files_group2_name}.csv'
    my_groups = (args.files_group1_name, args.files_group2_name)
    
    node_data = at.graph_to_df()
    
    node_atlas = at.make_node_atlas(node_data, args.rt_range)
    merged_node_data = at.merge_spectral_data(node_data)
    files = args.files_group1 + args.files_group2

    cols = ['inchi_key_identity','smiles_identity']
    data = node_data[cols].copy()
    data.drop_duplicates('inchi_key_identity',inplace=True)
    data = data[pd.notna(data['inchi_key_identity'])]
    data.rename(columns={'inchi_key_identity':'inchi_key','smiles_identity':'smiles'},inplace=True)
    data = calc_descriptor_df(data)
    
    file_groups = [args.files_group1_name for i in range(len(args.files_group1))] + [args.files_group2_name for i in range(len(args.files_group2))]
    grouped_files = dict(zip(files, file_groups))
    files_data = pd.DataFrame({'filename': grouped_files.keys(), 'sample_category': grouped_files.values()})
    
    ms1_data = at.get_sample_ms1_data(node_atlas, files, args.mz_tol, args.pk_height_min, args.num_data_min)
    max_ms1_data = at.get_best_ms1_rawdata(ms1_data, node_data)
    ms2_data = at.get_sample_ms2_data(files, merged_node_data, args.msms_score_min, args.msms_matches_min, args.mz_tol, args.frag_mz_tol)
    ms2_data = pd.concat(ms2_data)
    max_ms2_data = at.get_best_ms2_rawdata(ms2_data)
    best_hits = at.get_best_ms1_ms2_combined(max_ms1_data, max_ms2_data)
    stats_df = at.do_basic_stats(ms1_data, files_data, my_groups)
    output_df = at.make_output_df(node_data, best_hits, stats_df, filename=output_filename)
    
    at.annotate_graphml(output_df, node_data)
    
    ms1_data.to_csv('all_ms1_data.csv')
    ms2_data.to_csv('all_ms2_data.csv')
    

if __name__ == '__main__':
    parser = arg_parser()
    args = parser.parse_args()

    main(args)