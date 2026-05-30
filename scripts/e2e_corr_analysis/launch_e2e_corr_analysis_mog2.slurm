#!/bin/bash
#SBATCH -J e2e_analysis_NUP   # Job name
#SBATCH -o e2e_analysis_NUP_%j.out # Specify stdout output file (%j expands to jobId)
#SBATCH -p bigmem                   # Partition name
#SBATCH -n 1                     # Total number of tasks
#SBATCH -c 8
                  # CPUs per task
#SBATCH -t 3:00:00              # Run time (hh:mm:ss) - 0.5 hours
#SBATCH -A m2_komet331hpc       #Specify allocation to charge against

# Activate micromamba environment
eval "$(micromamba shell hook --shell bash )"
micromamba activate calvados

#cd $HOME/sgruenew/Simulations/singleNUP98/phase_diagram_lowT
#cd $HOME/sgruenew/Simulations/T-REMD_testing/NUP98/NUP98_WT/phase_diagram_lowT/stitched_trajs
#cd $HOME/sgruenew/Simulations/H-REMD_testing/kappa_testing/sc-test_hpl2+lin13_long
cd ~/sgruenew/Simulations/chromatin_condensates/multi_component_sims/hpl2+lin13_bigger_box

python /home/sgruenew/projects/calvados-remd/scripts/e2e_corr_analysis/e2e_corr_analysis.py --wildcard '*K' --top_name "starting_config.pdb" --comp_dict '{"Lin-13":"0,39", "Hpl-dimer":"40,79"}' --overwrite "True" 
#python $HOME/e2e_corr_analysis/e2e_corr_analysis.py --wildcard '*K' --comp_dict '{"NUP98":"0,99"}' --step 100 --overwrite "True"
#python $HOME/e2e_corr_analysis/e2e_corr_analysis.py --wildcard 'replica*' --comp_dict '{"NUP98":"0,99"}' --step 10 --overwrite "True"
