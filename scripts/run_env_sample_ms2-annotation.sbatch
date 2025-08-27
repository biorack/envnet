#!/bin/bash
#SBATCH --job-name="ms2-env-annotation"
#SBATCH --output="/pscratch/sd/b/bpb/metatlas_mdm_parquet_files/ms2-env-annotation.out"
#SBATCH --error="/pscratch/sd/b/bpb/metatlas_mdm_parquet_files/ms2-env-annotation.err"
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
export REF_DIR="/global/homes/b/bpb/repos/envnet/envnet/data"


# Create output directory with timestamp
OUTPUT_DIR="/global/homes/b/bpb/repos/envnet/results/full_env-annotation_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

# Change to scripts directory
cd /global/homes/b/bpb/repos/envnet/scripts

# Run the full build
$PYTHONBINARY -m envnet.annotation.workflows ms2 \
    --output-dir ./ms2_results \
    --google-sheets \
    --envnet-graphml $REF_DIR/envnet_network.graphml \
    --envnet-deconv-mgf $REF_DIR/envnet_deconvoluted_spectra.mgf \
    --envnet-original-mgf $REF_DIR/envnet_original_spectra.mgf \
    --spectrum-types deconvoluted original \
    --min-score 0.6 \
    --min-matches 3
