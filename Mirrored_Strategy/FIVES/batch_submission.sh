#!/bin/bash
#SBATCH -J myjob_normal_fin       # Job name
#SBATCH -o multicheck_dataN2_n6_256_fin.o%j   # Name of stdout output file
#SBATCH -p gpu-a100-dev       # Queue (partition) name
#SBATCH -N 2           # Total # of nodes
#SBATCH -n 6           # Total # of mpi tasks. Should be 1 for openmp apps
#SBATCH -t 02:00:00    # Run time (hh:mm:ss)
#SBATCH -A ASC23013    # Allocation name (req'd if you have mmore than 1)

#load modules
module load python3/3.7.13 cuda/11.3 cudnn nccl 
 
source /scratch/09825/dtu14/new_env/bin/activate

#run the code
python3 /scratch/09825/dtu14/Final_project/cs7389D_HPScaleProject/tani/transformer/result/multi_worker_port.py  


#unload module
module unload cuda/11.4 cudnn/8.2.4 nccl/2.11.4

echo $SLURM_NODELIST

echo $SLURMD_NODENAME
