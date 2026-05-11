#!/bin/bash
#SBATCH -J REMD_NUP98_test   # Job name
#SBATCH -o REMD_NUP98_test_%j.out # Specify stdout output file (%j expands to jobId)
#SBATCH -p m2_gpu                   # Partition name
#SBATCH -n 1                     # Total number of tasks
#SBATCH -c 12                     # CPUs per task
#SBATCH --gres=gpu:6
#SBATCH -t 120:00:00              # Run time (hh:mm:ss) - 0.5 hours
#SBATCH -A m2_komet331hpc       #Specify allocation to charge against

# Activate micromamba environment
eval "$(micromamba shell hook --shell bash )"
micromamba activate calvados

cd $HOME/sgruenew/Simulations/T-REMD_testing/NUP98/NUP98_WT/phase_diagram_lowT_newpy

python $HOME/projects/calvados-remd/scripts/launch_replicas_persistent.py --sysname NUP98 --path "." --platform CUDA --time_per_script 120
