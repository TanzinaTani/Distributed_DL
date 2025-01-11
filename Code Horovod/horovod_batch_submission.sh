#!/bin/bash
#SBATCH -J myjob_mirror       # Job name
#SBATCH -o myjob_horovod_msd_five_normal.o%j   # Name of stdout output file
#SBATCH -p gpu-a100       # Queue (partition) name
#SBATCH -N 1          # Total # of nodes
#SBATCH -n 1           # Total # of mpi tasks. Should be 1 for openmp apps
#SBATCH -t 04:30:00    # Run time (hh:mm:ss)
#SBATCH -A ASC23013    # Allocation name (req'd if you have mmore than 1)

#load modules
module load cuda/11.4 cudnn/8.2.4 nccl/2.11.4   

source $SCRATCH/python-envs/test/bin/activate

#run the code
python3 /scratch/09819/xeh13/oldgitproject/tani/transformer_Horovod/compile_horovod_normal.py

#unload module
module unload cuda/11.4 cudnn/8.2.4 nccl/2.11.4

echo $SLURM_NODELIST

echo $SLURMD_NODENAME


