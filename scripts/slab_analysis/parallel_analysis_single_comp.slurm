#!/bin/bash
#SBATCH -J parallel_analysis
#SBATCH -o ana_logs/parallel_analysis_%j.out
#SBATCH -p smallcpu
#SBATCH --mem=120G
#SBATCH -n 1               # single batch step
#SBATCH -c 12             # number of CPUs you want to use (>= number of tasks below
#SBATCH -t 04:00:00
#SBATCH -A nhr-dyndisphase

set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# --- (optional) environment activation ---
eval "$(micromamba shell hook --shell bash)"
micromamba activate calvados

# Tasks to run (example: four temperatures)
prots=(NUP98)

# Expand the job's allowed CPU list (e.g. "112-115,224-227" -> 112 113 114 115 224 225 226 227)
cpulist_raw=$(awk '/^Cpus_allowed_list:/ {print $2}' /proc/self/status)
expand_cpulist() {
  local s="$1" part
  IFS=, read -ra parts <<< "$s"
  for part in "${parts[@]}"; do
    if [[ "$part" == *-* ]]; then
      seq "${part%-*}" "${part#*-}"
    else
      echo "$part"
    fi
  done
}
mapfile -t CPUS < <(expand_cpulist "$cpulist_raw")
mkdir -p ana_logs
# Launch tasks in parallel, one per CPU
for p in "${prots[@]}"; do
  #cd /home/sgruenew/projects_nhr/Simulations/H-REMD_testing/test_exchange/hremd-test_NUP98_highT_ladder/
  #cd /home/sgruenew/projects_nhr/Simulations/singleNUP98/phase_diagram/WT/
  #cd /home/sgruenew/projects_nhr/Simulations/T-REMD_testing/NUP98/tremd_NUP98_highT_largestep/
  #cd ~/projects_nhr/Simulations/singleNUP98/cdil-test_NUP98_300K/
  #dirs=(*7.00mus_0)
  #cd ~/projects_nhr/Simulations/singleNUP98/phase_diagram_lowT/
  #dirs=(*284.14K)
   
   cd ~/projects_nhr/Simulations/T-REMD_testing/NUP98/tremd_NUP98_284K-300K_largestep/
   dirs=(*K)

  #cd /home/sgruenew/projects_nhr/Simulations/H-REMD_testing/test_exchange/hremd-test_NUP98_highT_ladder/
  #cd ~/projects_nhr/Simulations/H-REMD_testing/test_exchange/hremd-test_NUP98_300K/
  #dirs=(*rt0.000*)
  
  #dirs=(NUP98_WT_302.29K NUP98_WT_299.50K NUP98_WT_296.78K NUP98_WT_294.13K NUP98_WT_281.95K NUP98_WT_277.59K)
  for i in "${!dirs[@]}"; do
    sysname="${dirs[$i]}"
    CPU="${CPUS[$i]}"
    ts="$(date +%Y-%m-%d_%H-%M-%S)"
    mkdir -p /home/sgruenew/projects/calvados-remd/scripts/slab_analysis/ana_logs
    #mkdir -p data_half_traj
    log="/home/sgruenew/projects/calvados-remd/scripts/slab_analysis/ana_logs/${sysname}_${ts}.out"
    (
    taskset -c "$CPU" bash -lc "
      echo \"\$(date +%T) start $sysname on \$(hostname) (CPU $CPU)\"
       #python ../csat_vs_t.py --name "NUP98" --path "." --sysname "$sysname" --ref_name "NUP98"
       python /home/sgruenew/projects/calvados-remd/scripts/slab_analysis/single_comp_slab_analysis.py --name NUP98 --sysname $sysname --path . --ref_name "NUP98"
       echo \"\$(date +%T) done $sysname (CPU $CPU)\"
    ") >>"$log" 2>&1 &
  done
done 

wait
echo "$(date +%T) all done"

