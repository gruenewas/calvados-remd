# CALVADOS (patched) + REMD Utilities

This repository contains:

- Modified CALVADOS source code (`./calvados/`)
- REMD orchestration scripts (`./scripts/`)
- Analysis utilities
- A minimal REMD example (`./examples/`)

The intended workflow is:

1) Install official CALVADOS following the official GitHub instructions  
2) Overlay the patched CALVADOS version from this repository  
3) Run the example or your own REMD simulations

---

# Step 1 — Install Official CALVADOS

Follow the installation instructions from the official CALVADOS repository:

https://github.com/KULL-Centre/CALVADOS

This should create a working conda/mamba environment with all required dependencies (OpenMM, numpy, etc.).

Activate that environment:

    conda activate <calvados_env_name>

Verify official installation:

    python -c "import calvados; print(calvados.__file__)"

This should point to the official site-packages installation.

---

# Step 2 — Apply Patched Version from This Repository

Clone this repository:

    git clone https://github.com/gruenewas/calvados-remd.git
    cd calvados-remd

Inside the activated CALVADOS environment, replace the installed version with the patched one:

    pip uninstall -y calvados
    pip install -e .

Verify that Python now imports CALVADOS from this repository:

    python -c "import calvados; print(calvados.__file__)"

The printed path must point inside this repository (not site-packages).

---

# Step 3 — Run the Minimal REMD Example

A minimal REMD example is provided in:

    examples/tremd_NUP98

This example is designed to run on CPU by default, so it can be executed on any machine without requiring a GPU.

### 1. Prepare the replica folders

From the repository root run:

    cd examples/tremd_NUP98
    python prepare_minimal_replica_exchange_NUP.py

This script prepares the replica directories and generates the necessary input files.

---

### 2. Launch the REMD simulation

Move to the scripts directory:

    cd ../../scripts

Then start the replicas:

    python launch_replicas_parallel_pairs.py \
        --sysname "NUP98" \
        --path "../examples/tremd_NUP98/tremd-test_NUP98/" \
        --platform "CPU"

This launches the minimal REMD example using the CPU platform.

---

# Running the Example on GPU

The minimal example defaults to CPU execution for portability.

To run the simulation on a GPU:

1. Modify the prepare script to use CUDA:

    examples/tremd_NUP98/prepare_minimal_replica_exchange_NUP.py

Run the preparation step with:

    python prepare_minimal_replica_exchange_NUP.py --platform CUDA

2. Launch the simulation using the same platform flag:

    python launch_replicas_parallel_pairs.py \
        --sysname "NUP98" \
        --path "../examples/tremd_NUP98/tremd-test_NUP98/" \
        --platform "CUDA"

Make sure that:
- CUDA drivers are installed
- an NVIDIA GPU is available
- OpenMM was installed with CUDA support

---

# Notes

- The example provided here is minimal and intended for testing the REMD workflow.
- Larger production simulations will typically run on GPUs and may require multiple devices.
- The original CALVADOS installation can be restored at any time by reinstalling it from the official repository.

