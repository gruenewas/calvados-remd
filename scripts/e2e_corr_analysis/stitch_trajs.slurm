#!/bin/bash
#SBATCH -J stitch_trajs   # Job name
#SBATCH -o slurm_logs/slurm_%j.out
#SBATCH -e slurm_logs/slurm_%j.err
#SBATCH -p smallcpu                   # Partition name
#SBATCH -n 1                     # Total number of tasks
#SBATCH -c 1                     # CPUs per task
#SBATCH -t 01:30:00              # Run time (hh:mm:ss) - 0.5 hours
#SBATCH -A nhr-dyndisphase       #Specify allocation to charge against
#SBATCH --mem=55G

# Activate micromamba environment
eval "$(micromamba shell hook --shell bash )"
micromamba activate calvados

export PYTHONUNBUFFERED=1
#cd /home/projects_nhr/Simulations/H-REMD_testing/test_exchange/hremd-test_NUP98_highT_ladder
#cd /home/sgruenew/projects_nhr/Simulations/T-REMD_testing/NUP98/phase_diagram_lowT_nhr
#cd ~/projects_nhr/Simulations/T-REMD_testing/NUP98/phase_diagram_lowT_newpy/
#cd /home/sgruenew/projects_nhr/Simulations/T-REMD_testing/NUP98/tremd_NUP98_highT_largestep
cd ~/projects_nhr/Simulations/H-REMD_testing/test_exchange/hremd-test_hpl2+lin13_highT_long/

python -u -c "from md_analysis_lib.remd_tools import stitch_traj; stitch_traj('stitched_trajs', log='remd_log.csv', folder_pre='hpl2+lin13', ex_col='rt',dt=1000,step=10)" > stitch_python.log 2>&1
