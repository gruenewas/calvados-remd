# CALVADOS (patched) + REMD Utilities

This repository contains:

- Modified CALVADOS source code (`./calvados/`)
- REMD orchestration scripts (`./scripts/`)
- Analysis utilities

The intended workflow is:

1) Install official CALVADOS following the official GitHub instructions
2) Overlay the patched CALVADOS version from this repository

---

## Step 1 — Install Official CALVADOS

Follow the installation instructions from the official CALVADOS repository:

    https://github.com/KULL-Centre/CALVADOS

This should create a working conda/mamba environment with all required dependencies (OpenMM, numpy, etc.).

Activate that environment:

    conda activate <calvados_env_name>

Verify official installation:

    python -c "import calvados; print(calvados.__file__)"

This should point to the official site-packages installation.

---

## Step 2 — Apply Patched Version from This Repository

Clone this repository:

    git clone <YOUR_REPO_URL>
    cd calvados-remd

Inside the activated CALVADOS environment, replace the installed version with the patched one:

    pip uninstall -y calvados
    pip install -e .

Verify that Python now imports CALVADOS from this repository:

    python -c "import calvados; print(calvados.__file__)"

The printed path must point inside this repository (not site-packages).

---

## Notes

- We use an editable install (`pip install -e .`) so that any local code changes
  immediately affect the Python package.
- Do NOT manually copy files into site-packages.
- If needed, the original CALVADOS can be restored by reinstalling it
  from the official repository.

