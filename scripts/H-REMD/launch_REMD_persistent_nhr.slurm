#!/bin/bash
#SBATCH -J REMD_NUP98_test   # Job name
#SBATCH -o persistent_logs/TREMD_NUP98_persistent_test_%j.out # Specify stdout output file (%j expands to jobId)
#SBATCH -p a100dl                   # Partition name
#SBATCH -n 1                     # Total number of tasks
#SBATCH -c 12                     # CPUs per task
#SBATCH --gres=gpu:4
#SBATCH -t 24:00:00              # Run time (hh:mm:ss) - 0.5 hours
#SBATCH -A nhr-dyndisphase      #Specify allocation to charge against
#SBATCH --exclude=gpu0001

# Activate micromamba environment
eval "$(micromamba shell hook --shell bash )"
micromamba activate calvados

cd $HOME/projects_nhr/Simulations/T-REMD_testing/NUP98/phase_diagram_lowT_newpy

python $HOME/projects/calvados-remd-b694e8b/scripts/REMD/launch_replicas_persistent.py --sysname NUP98 --path "." --platform CUDA --total-gpus 4 --time_per_script 144
