#!/bin/bash
#SBATCH -J TREMD_nup98_298-315  # Job name
#SBATCH -o TREMD_nup98_298-315_%j.out # Specify stdout output file (%j expands to jobId)
#SBATCH -p a100dl                   # Partition name
#SBATCH -n 1                     # Total number of tasks
#SBATCH -c 12                     # CPUs per task
#SBATCH --gres=gpu:4
#SBATCH -t 36:00:00              # Run time (hh:mm:ss) - 0.5 hours
#SBATCH -A nhr-dyndisphase      #Specify allocation to charge against
#SBATCH --exclude=gpu0001

# Activate micromamba environment
eval "$(micromamba shell hook --shell bash )"
micromamba activate calvados

#cd /home/sgruenew/projects_nhr/Simulations/H-REMD_testing/test_exchange/hremd-test_NUP98_highT_ladder/
#cd ~/projects_nhr/Simulations/H-REMD_testing/test_exchange/hremd-test_NUP98_300K/
#cd /home/sgruenew/projects_nhr/Simulations/H-REMD_testing/test_exchange/hremd-test_hpl2+lin13_highT_long
#cd ~/projects_nhr/Simulations/T-REMD_testing/NUP98/phase_diagram_lowT_nhr/
#cd ~/projects_nhr/Simulations/T-REMD_testing/NUP98/tremd_NUP98_highT_largestep
cd ~/projects_nhr/Simulations/T-REMD_testing/NUP98/tremd_NUP98_298-315_largestep/
#cd ~/projects_nhr/Simulations/T-REMD_testing/NUP98/tremd_NUP98_284K-300K_largestep/
#cd ~/projects_nhr/Simulations/T-REMD_testing/NUP98/tremd_NUP98_294-308_largestep/

python $HOME/projects/calvados-remd/scripts/REMD/launch_replicas_persistent.py --sysname NUP98 --path "." --platform CUDA --total-gpus 4 --time_per_script 36
#python $HOME/projects/calvados-remd/scripts/REMD/launch_replicas_persistent.py --sysname hpl2+lin13 --path "." --platform CUDA --total-gpus 4 --time_per_script 144
