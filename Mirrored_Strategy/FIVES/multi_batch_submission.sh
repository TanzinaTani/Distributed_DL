#!/bin/bash
#SBATCH -J myjob_mirror         # Job name
#SBATCH -o myjob_mirror_msd_%j  # Name of stdout output file (%j expands to jobId)
#SBATCH -p gpu-a100
#SBATCH -N 2                    # Total number of nodes (must be the same across all jobs)
#SBATCH -n 6                    # Total number of mpi tasks (should be 1 for OpenMP applications)
#SBATCH --time=01:40:00         # Run time (hh:mm:ss)
#SBATCH -A ASC23013              # Allocation name

# Load modules
module load python3/3.7.13 cuda/11.3 cudnn nccl 
 
 
source /scratch/09825/dtu14/new_env/bin/activate

# Retrieve node information
nodes=($(scontrol show hostnames $SLURM_NODELIST))
nodename=$(hostname)

# Set TF_CONFIG for each node
if [[ " ${nodes[0]} " == " $nodename " ]]; then
    export TF_CONFIG='{
        "cluster": {
            "worker": ["'${nodes[0]}':12345", "'${nodes[1]}':12345"]
        },
        "task": {"type": "worker", "index": 0}
    }'
else
    export TF_CONFIG='{
        "cluster": {
            "worker": ["'${nodes[0]}':12345", "'${nodes[1]}':12345"]
        },
        "task": {"type": "worker", "index": 1}
    }'
fi

# Output the settings for debugging
echo "Running on node: $nodename"
echo "TF_CONFIG: $TF_CONFIG"

# Run the training script
python3 /scratch/09825/dtu14/Final_project/cs7389D_HPScaleProject/tani/transformer/result/data_mirror_compile.py  
# Unload module
module unload cuda/11.4 cudnn/8.2.4 nccl/2.11.4

