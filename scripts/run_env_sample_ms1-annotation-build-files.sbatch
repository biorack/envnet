#!/bin/bash
#SBATCH --job-name="ms1-env-annotation"
#SBATCH --output="/pscratch/sd/b/bpb/metatlas_mdm_parquet_files/ms1-env-annotation.out"
#SBATCH --error="/pscratch/sd/b/bpb/metatlas_mdm_parquet_files/ms1-env-annotation.err"
#SBATCH --time=08:00:00
#SBATCH --qos=regular
#SBATCH --ntasks-per-node=128
#SBATCH --licenses=cfs
#SBATCH --exclusive
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --account=m2650

# Set up environment
export PYTHONPATH=/global/homes/b/bpb/repos/envnet
#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONBINARY=/global/common/software/m2650/msbuddy/bin/python
export REF_DIR="/global/homes/b/bpb/repos/envnet/results/full_build_20250908_181404"
export OUT_DIR="/pscratch/sd/b/bpb/envnet_annotation_results"

# # Create output directory with timestamp
# OUTPUT_DIR="/global/homes/b/bpb/repos/envnet/results/full_env-annotation_$(date +%Y%m%d_%H%M%S)"
# mkdir -p $OUTPUT_DIR

# Change to scripts directory
cd /global/homes/b/bpb/repos/envnet/scripts

$PYTHONBINARY -m envnet.annotation.workflows ms1 \
    --output-dir "$OUT_DIR/ms1_results_build_files_for_paper" \
    --csv-file "/global/homes/b/bpb/repos/envnet/scripts/build_files.csv" \
    --envnet-graphml "$REF_DIR/envnet_network.graphml" \
    --envnet-deconv-mgf "$REF_DIR/envnet_deconvoluted_spectra.mgf" \
    --envnet-original-mgf "$REF_DIR/envnet_original_spectra.mgf" \
    --spectrum-types deconvoluted \
    --min-library-match-score 0.6 \
    --min-matches 3

