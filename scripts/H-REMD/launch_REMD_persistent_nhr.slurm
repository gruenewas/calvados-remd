#!/bin/bash
#SBATCH -J REMD_NUP98_test   # Job name
#SBATCH -o REMD_NUP98_test_%j.out # Specify stdout output file (%j expands to jobId)
#SBATCH -p a100dl                   # Partition name
#SBATCH -n 1                     # Total number of tasks
#SBATCH -c 12                     # CPUs per task
#SBATCH --gres=gpu:4
#SBATCH -t 144:00:00              # Run time (hh:mm:ss) - 0.5 hours
#SBATCH -A nhr-dyndisphase      #Specify allocation to charge against

# Activate micromamba environment
eval "$(micromamba shell hook --shell bash )"
micromamba activate calvados

cd $HOME/projects_nhr/Simulations/T-REMD_testing/NUP98/phase_diagram_lowT_newpy

python $HOME/projects/calvados-remd/scripts/REMD/launch_replicas_persistent.py --sysname NUP98 --path "." --platform CUDA --time_per_script 144 
