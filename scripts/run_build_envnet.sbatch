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

# Create output directory with timestamp
OUTPUT_DIR="/global/homes/b/bpb/repos/envnet/results/full_build_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

# Change to scripts directory
cd /global/homes/b/bpb/repos/envnet/scripts

# Print job info
echo "Starting ENVnet full build at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Output directory: $OUTPUT_DIR"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 128GB"
echo "Time limit: 6 hours"
echo "=========================================="

# Run the full build
/global/common/software/m2650/msbuddy/bin/python build_envnet.py \
    --output $OUTPUT_DIR \
    --config build_from_csv.yaml \
    --verbose \
    build

# Print completion info
echo "=========================================="
echo "ENVnet full build completed at $(date)"
echo "Results saved to: $OUTPUT_DIR"

# Print final statistics
if [ -f "$OUTPUT_DIR/envnet_network.graphml" ]; then
    echo "Network file created successfully"
else
    echo "WARNING: Network file not found!"
fi

ls -lh $OUTPUT_DIR/