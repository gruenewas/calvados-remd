>#!/bin/bash
#SBATCH -J e2e_analysis   # Job name
#SBATCH -o e2e_analysis_%j.out # Specify stdout output file (%j expands to jobId)
#SBATCH -p smallcpu                   # Partition name
#SBATCH -n 1                     # Total number of tasks
#SBATCH -c 12
#SBATCH --mem=50G                     # CPUs per task
#SBATCH -t 2:00:00              # Run time (hh:mm:ss) - 0.5 hours
#SBATCH -A nhr-dyndisphase       #Specify allocation to charge against

# Activate micromamba environment
eval "$(micromamba shell hook --shell bash )"
micromamba activate calvados

#cd $HOME/projects_nhr/Simulations/H-REMD_testing/test_exchange/hremd-test_hpl2+lin13_long/stitched_trajs
#cd $HOME/projects_nhr/Simulations/H-REMD_testing/test_exchange/hremd-test_NUP98_highT_ladder/stitched_trajs
#cd /home/sgruenew/projects_nhr/Simulations/singleNUP98/phase_diagram_lowT
#cd /home/sgruenew/projects_nhr/Simulations/T-REMD_testing/NUP98/phase_diagram_lowT_nhr/stitched_trajs
#cd ~/projects_nhr/Simulations/T-REMD_testing/NUP98/phase_diagram_lowT_newpy/stitched_trajs/
#cd ~/projects_nhr/Simulations/T-REMD_testing/NUP98/tremd_NUP98_highT_largestep/stitched_trajs/
cd ~/projects_nhr/Simulations/H-REMD_testing/test_exchange/hremd-test_hpl2+lin13_highT_long/stitched_trajs
#cd /home/sgruenew/projects_nhr/Simulations/H-REMD_testing/test_ladder/rt_ladder_test

python /home/sgruenew/projects/calvados-remd/scripts/e2e_corr_analysis/e2e_corr_analysis.py --wildcard 'replica*' --comp_dict '{"Lin-13":"0,39", "Hpl-dimer":"40,79"}'
#python /home/sgruenew/projects/calvados-remd/scripts/e2e_corr_analysis/e2e_corr_analysis.py --wildcard '*K' --comp_dict '{"NUP98":"0,99"}' --overwrite True

