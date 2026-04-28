# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

Research code from a Fermilab stay (Fall 2022) for DES (Dark Energy Survey) cluster cosmology work. The project is "undergoing" and the README explicitly warns things are "a bit messy and not well documented." Treat scripts as exploratory — expect hardcoded paths, commented-out code blocks, and WIP notebooks alongside the Python modules.

## Top-level sub-projects

Each directory under `fnalFall2022/` is an independent sub-project with its own scripts and notebooks. There is no root-level build, test suite, or dependency manifest.

- `smass/` — ANN-based stellar mass estimator trained on COSMOS (31-band Hubble deep field) and validated on SDSS+GALEX+WISE. Notebooks in `smass/notebooks/` are numbered (`0.*` → `15.*`) and represent a roughly linear R&D progression; `smass/scripts/createInputFile.py` builds the `.npy` training vectors consumed by downstream notebooks.
- `r200/` — Improvements to the copacabana $R_{200}$ cluster-radius estimator using stellar masses. `corcovadoShape.py` (class `corcovadoShape`) measures cluster shape from ra/dec + stellar masses; `projector.py` handles sky→physical-Mpc projection.
- `qmodel/` — Quenching model R&D (currently just red-sequence notebooks).
- `y3kp/` — DES Year 3 Key Project clustering / cluster-cosmology work, split into:
  - `buzzardMock/` — Buzzard v1.9.8 mock construction, 2PCF and data-vector building.
  - `correlationFunction/` — TreeCorr-based 2PCF runs with k-means jackknife covariance (`run.py`, driven by `jobscript.sh`).
  - `powerSpec/` — `nbodykit` 3D power spectrum and multipole measurements (`scripts/run_nbodyKit.py`).
  - `boostFactor/` — CosmoSIS pipeline modules (`boost_factor1.py`, `Boost_Factor_like.py`) and `.ini` configs for fitting a cluster boost-factor model with emcee. Outputs under `boostFactor/outputs/` are emcee chains; `backup/` holds older versions of the same files.
  - `hankelFunction/`, `xpipeTutorial/`, `debugBuzzardY3/` — smaller utility / tutorial notebooks.

## How things are run

There is no `pip install` target, `Makefile`, or pytest suite. Code is run three ways:

1. **Jupyter notebooks** — the vast majority of files. Notebooks in a sub-project generally assume you have already run the numbered predecessor (e.g. `smass/notebooks/0.*` must run before `1.*`).
2. **Argparse-driven Python scripts submitted via SLURM** on NERSC. Pattern: each runnable dir has a `run*.py` or similar plus a `jobscript.sh`. Example from `y3kp/correlationFunction/jobscript.sh`:
   ```
   python run.py 'mock_test' --is_3d 1 --nPatches 10 --nCores 31
   ```
   Jobscripts request `--constraint=haswell` on Cori and activate an nbodykit/treecorr conda env before running.
3. **CosmoSIS pipelines** — `y3kp/boostFactor/*.ini`. The `[pipeline]` section lists the Python modules (`BoostFactor`, `Boost_Factor_like`) that CosmoSIS drives; `[emcee]`/`[polychord]` select the sampler; `[output]` sets the chain path. Run via `cosmosis <name>.ini` inside a CosmoSIS environment. Several `.ini` files point to `/global/u2/j/jesteves/...` — these paths must be updated when re-running under a different user.

## Cross-cutting conventions

- **Environment switching via `FileLocs`** — `y3kp/buzzardMock/fileLoc.py` and `y3kp/powerSpec/scripts/fileLoc.py` both define a `FileLocs(machine='nersc' | 'fnal')` class that centralizes absolute paths (e.g. `/global/cfs/cdirs/des/...` on NERSC vs `/data/des61.a/...` at FNAL). When adding a script that reads catalogs, follow this pattern rather than hardcoding a single absolute path. The two `fileLoc.py` copies have drifted — do not assume they are identical.
- **Hardcoded user paths** — many scripts reference `/global/u2/j/jesteves/...` or `/data/des61.a/data/johnny/...`. These are the original author's paths and will not resolve in this checkout. Check and edit before running.
- **Binning conventions** — lambda (richness) bins and redshift bins are set up in `y3kp/correlationFunction/set_bins_files.py` (`lbd_bins`, `z_bins`) and referenced from ini files in `boostFactor/`. Keep them consistent across the stack.
- **Large artifacts** — `.fits` / `.hdf5` / `.npy` files live next to the scripts rather than under a `data/` directory. The `.gitignore` excludes `*tmp*.fits` and `*.hdf5`, so small `.npy` and `.npz` artifacts under `smass/data/`, `y3kp/buzzardMock/`, and `y3kp/powerSpec/data/` are intentionally committed — don't delete them assuming they're regeneratable without checking the producing notebook.

## External dependencies (observed in imports)

No lockfile exists; the code expects a scientific-Python env with: `numpy`, `scipy`, `astropy`, `healpy`, `joblib`, `treecorr`, `nbodykit`, `halotools`, `fitsio`, `cosmosis` (for `boostFactor`), and sklearn / tensorflow / keras variants for the ANN work in `smass`. The NERSC jobscripts assume a pre-built conda env activated outside the script.
