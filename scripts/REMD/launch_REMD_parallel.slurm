#!/bin/bash
#SBATCH -J REMD_NUP98_test   # Job name
#SBATCH -o REMD_NUP98_test_%j.out # Specify stdout output file (%j expands to jobId)
#SBATCH -p deeplearning                   # Partition name
#SBATCH -n 1                     # Total number of tasks
#SBATCH -c 16                     # CPUs per task
#SBATCH --gres=gpu:8
#SBATCH -t 18:00:00              # Run time (hh:mm:ss) - 0.5 hours
#SBATCH -A m2_komet331hpc       #Specify allocation to charge against

# Activate micromamba environment
eval "$(micromamba shell hook --shell bash )"
micromamba activate calvados

python launch_replicas_parallel_pairs.py --sysname NUP98 --path "NUP98/NUP98_WT/phase_diagram_lowT"  --platform CUDA
