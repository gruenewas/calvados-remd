#!/bin/bash
#SBATCH -J parallel_analysis
#SBATCH -o parallel_analysis_%j.out
#SBATCH -p smallcpu
#SBATCH --mem=150G
#SBATCH -n 1               # single batch step
#SBATCH -c 4               # number of CPUs you want to use (>= number of tasks below
#SBATCH -t 03:00:00
#SBATCH -A nhr-dyndisphase

set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# --- (optional) environment activation ---
eval "$(micromamba shell hook --shell bash)"
micromamba activate calvados


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
#mkdir -p ana_logs/archive
#mv ana_logs/*.out ana_logs/archive

# Launch tasks in parallel, one per CPU
cd ~/projects_nhr/Simulations/H-REMD_testing/sc-test_rt_hpl2+lin13_long

dirs=(hpl2*)
for i in "${!dirs[@]}"; do
  sysname="${dirs[$i]}"
  CPU="${CPUS[$i]}"
  ts="$(date +%Y-%m-%d_%H-%M-%S)"
  mkdir -p /home/sgruenew/projects/calvados-remd/scripts/slab_analysis/ana_logs
  log="/home/sgruenew/projects/calvados-remd/scripts/slab_analysis/ana_logs/${sysname}_${ts}.out"
  (
  taskset -c "$CPU" bash -lc "
    echo \"\$(date) start $sysname on \$(hostname) (CPU $CPU)\"
     python /home/sgruenew/projects/calvados-remd/scripts/slab_analysis/multi_comp_slab_analysis.py --name "hpl2+lin13" --sysname $sysname --path . --ref_name "Lin-13" --start -3000
    echo \"\$(date +%T) done (CPU $CPU)\"
  ") >>"$log" 2>&1 &
done

wait
echo "$(date +%T) all done"

