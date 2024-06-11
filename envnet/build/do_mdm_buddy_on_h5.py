
import sys
import os
import argparse
sys.path.insert(0,'/global/homes/b/bpb/repos/carbon_network')
from carbon_network.build import preprocess as pp
import pandas as pd

# file = '/global/cfs/cdirs/metatlas/projects/rawdata_for_scn/20181217_KBL_TM_Lakes_GEODES_All3_QE-HF_C18_USDAY46918_NEG_MSMS_10_GEO-ME-44-UF_1_Rg80to1200-CE102040-0-7-S1_Run46.h5'

def main(args):
    # Parameters
    mz_tol = args.mz_tol
    similarity_cutoff = args.similarity_cutoff
    isolation_tol = args.isolation_tol
    min_intensity_ratio = args.min_intensity_ratio
    my_polarity = args.my_polarity
    max_rt = args.max_rt
    file = args.file
    deltas = pd.read_csv('/global/homes/b/bpb/repos/scndb/data/mdm_neutral_losses.csv')

    out_file = file.replace('.h5', '.parquet')
    # out_file = '/pscratch/sd/b/bpb/metatlas_mdm_parquet_files' + out_file
    # os.makedirs(os.path.dirname(out_file), exist_ok=True)

    print(out_file)
    try:
        df = pp.run_workflow(file,
                            deltas,
                            mz_tol=mz_tol,
                            similarity_cutoff=similarity_cutoff,
                            isolation_tol=isolation_tol,
                            my_polarity=my_polarity,
                            do_buddy=args.do_buddy,
                            max_rt=max_rt)
        df.to_parquet(out_file)
    except Exception as e:
        print(e)
        print(f'Error processing {file}')
        # Create an empty file using out_file as the filename
        open('%s-failed'%out_file, 'w').close()
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run MDM Buddy on H5 file.')
    parser.add_argument('--mz_tol', type=float, default=0.002, help='m/z tolerance')
    parser.add_argument('--similarity_cutoff', type=float, default=0.8, help='similarity cutoff')
    parser.add_argument('--isolation_tol', type=float, default=0.5, help='isolation tolerance')
    parser.add_argument('--min_intensity_ratio', type=int, default=2, help='minimum intensity ratio')
    parser.add_argument('--my_polarity', type=str, default='negative', choices=['positive', 'negative'], help='polarity')
    parser.add_argument('--max_rt', type=int, default=10000, help='maximum retention time')
    parser.add_argument('--file', type=str, required=True, help='input H5 file')
    parser.add_argument('--do_buddy', action='store_true', help='Run MS-Buddy on the file. If this flag is provided, it will be True. Otherwise, it will be False.')
    args = parser.parse_args()
    main(args)





