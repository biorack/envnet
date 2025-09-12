#!/bin/bash
#SBATCH --job-name="DOM_Interlab-LCMS_Lab018_M_Neg_MS2_rep2"
#SBATCH --output="/pscratch/sd/b/bpb/metatlas_mdm_parquet_files/DOM_Interlab-LCMS_Lab018_M_Neg_MS2_rep2.out"
#SBATCH --error="/pscratch/sd/b/bpb/metatlas_mdm_parquet_files/DOM_Interlab-LCMS_Lab018_M_Neg_MS2_rep2.err"
#SBATCH --time=08:00:00
#SBATCH --qos=regular
#SBATCH --ntasks-per-node=128
#SBATCH --licenses=cfs
#SBATCH --exclusive
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --account=m2650

export PYTHONPATH=/global/homes/b/bpb/repos/envnet

/global/common/software/m2650/msbuddy/bin/python -m envnet.deconvolution.workflows --do_buddy "/global/cfs/cdirs/metatlas/projects/envnet_build_files/analysis_for_manuscript/20240125_EB_MdR_101544-059_WAVESTAB3_20231222_EXP120A_C18-EP_USDAY72349_NEG_MS2_38_supern-CentExp-OMT1d2-NatCom-d7-Lo_2_6uL_117.h5"