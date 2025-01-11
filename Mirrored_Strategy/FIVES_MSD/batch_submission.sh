#!/bin/bash
#SBATCH -J myjob_msd       # Job name
#SBATCH -o msd_N2_n3_256.o%j   # Name of stdout output file
#SBATCH -p gpu-a100       # Queue (partition) name
#SBATCH -N 2           # Total # of nodes
#SBATCH -n 3           # Total # of mpi tasks. Should be 1 for openmp apps
#SBATCH -t 05:00:00    # Run time (hh:mm:ss)
#SBATCH -A ASC23013    # Allocation name (req'd if you have mmore than 1)

#load modules
module load python3/3.7.13 cuda/11.3 cudnn nccl	

source /scratch/09825/dtu14/new_env/bin/activate

#run the code
python3 /scratch/09825/dtu14/Final_project/cs7389D_HPScaleProject/tani/transformer/MSD_code/compile_mirror.py


#unload module
module unload cuda/11.4 cudnn/8.2.4 nccl/2.11.4

echo $SLURM_NODELIST

echo $SLURMD_NODENAME
