#!/bin/bash
#SBATCH --job-name="build_envnet"
#SBATCH --output="/pscratch/sd/b/bpb/metatlas_mdm_parquet_files/build_envnet.out"
#SBATCH --error="/pscratch/sd/b/bpb/metatlas_mdm_parquet_files/build_envnet.err"
#SBATCH --time=04:00:00
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


OUTPUT_DIR="/global/u2/b/bpb/repos/envnet/results/full_build_20250908_181404"


# Change to scripts directory
cd /global/homes/b/bpb/repos/envnet/scripts



# Run the full build
/global/common/software/m2650/msbuddy/bin/python build_envnet.py sirius \
    --output "$OUTPUT_DIR" \
    --network-file "$OUTPUT_DIR/envnet_network.graphml" \
    --sirius-dir "$OUTPUT_DIR"
