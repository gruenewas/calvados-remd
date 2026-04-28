#!/bin/bash
#SBATCH -J HREMD_test_ladder_opt   # Job name
#SBATCH -o HREMD_test_ladder_opt%j.out # Specify stdout output file (%j expands to jobId)
#SBATCH -p a100dl                   # Partition name
#SBATCH -n 1                     # Total number of tasks
#SBATCH -c 12                     # CPUs per task
#SBATCH --gres=gpu:4
#SBATCH -t 72:00:00              # Run time (hh:mm:ss) - 0.5 hours
#SBATCH -A nhr-dyndisphase       #Specify allocation to charge against
#SBATCH --exclude=gpu0001

# Activate micromamba environment
eval "$(micromamba shell hook --shell bash )"
micromamba activate calvados

cd $HOME/projects_nhr/Simulations/H-REMD_testing/test_exchange/hremd-test_hpl2+lin13_ladder_opt

python $HOME/projects/calvados-remd/scripts/REMD/launch_replicas_parallel_pairs_hremd.py --sysname hpl2+lin13 --path "."  --platform CUDA --max_walltime 72 --sims_per_gpu 3
