# Xin_MockDataVector — Buzzard mock with optical selection effects

Forward model for a DES-Y3-like cluster data vector built on Buzzard
halos, with the Costanzi et al. 2026 optical-selection systematics
layered on. Produces:

1. `N(λ_obs, z)` — observed cluster number counts.
2. `γ_obs(R, λ_obs-bin, z-bin)` — `B_sel`-weighted stacked tangential shear
   (and the matching `ΔΣ` stacks for reference).

Boost-factor contamination is **not** included in this pass (future work).

## Schematic

```
  raw halos + profiles (fitsio.read floc.halo_run_fname,
                                    floc.profile_output_fname)
        |  select_good (pid==-1, cosi∈[0,1], drop 0.33≤z≤0.37 seam)
        |  + 0.2 ≤ z ≤ 0.65, log10 M_vir ≥ 13   (mirror 0-MakeMock.ipynb)
        v
  halo table: Mvir, M200m, redshift, RA, DEC, DeltaSigma[15]
        |
        |  ---- Costanzi 2026 Eq. 15  (compound Poisson + lognormal) ----
        |        <λ_true | M, z> = 1 + ((M - M_min)/M_pivot)^α
        |                             × ((1+z)/(1+z_piv))^ε
        |        λ_true ~ pltr_M(λ | M, z)
        v
  λ_true per halo
        |
        |  ---- P(λ_obs | λ_true, z)  (δ + positive exponential) ----
        |        with prob (1−f_prj) →  λ_obs = λ_true
        |        else                →  λ_obs = λ_true + Exp(1/τ)
        |        f_prj(λ_true,z), τ(λ_true,z) from
        |        prj_params_DESY3_lss_lin_dep_getdist_v1.txt (posterior mean)
        v
  λ_obs per halo  ──────────► OUTPUT 1: N(λ_obs, z)
        |
        |  ---- B_sel weight (one-halo / small-scale limit, App. C) ----
        |        <Δ_prj | λ_obs, z>  ≡  f_prj / τ
        |        w_i = 1 + ((λ_obs_i − λ_true_i) − <Δ_prj>) / <Δ_prj>
        |        (population mean over δ+exp mixture → 1)
        v
  γ_mock_i(R) = ΔΣ_i(R) × Σ_crit^{-1}(z_i)    (β_eff table, same as
                                                  createDataVector.ipynb)
        |
        |  ---- weighted stack over (λ_obs, z) bin ----
        |        γ_obs(R) = Σ_i w_i γ_mock_i(R) / Σ_i w_i
        v
  γ_obs(R, λ_obs-bin, z-bin) ───► OUTPUT 2
```

## Parameters (DES-Y1 NC + 3×2pt best fit, Costanzi notebook defaults)

Hard-coded in `costanzi_selection.py`.

| Symbol       | Value                              |
|--------------|------------------------------------|
| `M_min`      | 10^11.38528 M_⊙                    |
| `α`          | 0.85869                            |
| `M_1`        | 10^12.69644 M_⊙                    |
| `M_pivot`    | M_1 − M_min                        |
| `σ_intr`     | 0.18095                            |
| `ε`          | 0.28389                            |
| `z_piv`      | 0.4544                             |

Projection coefficients (`a_tau, b_tau, a_fprj, b_fprj, …`) are loaded as
the posterior mean of the 15 rows in
`prj_params_DESY3_lss_lin_dep_getdist_v1.txt` (downloaded from
`MCostanzi/SelectionBias`).

## Key equations (verify against the PDF)

- **Eq. 15 (mass-richness mean + scatter)**:
  `<λ_true | M, z> = 1 + [(M - M_min)/M_pivot]^α  [(1+z)/(1+z_piv)]^ε`,
  scatter `σ_intr · l_sat(M,z)`, compound Poisson + lognormal P(λ_true|M,z).

- **Projection convolution** (δ + positive exponential mixture):
  `P(λ_obs | λ_true, z) = (1 − f_prj) δ_D(λ_obs − λ_true)
                         + f_prj · τ · e^{−τ(λ_obs − λ_true)} Θ_H(λ_obs − λ_true)`.

- **Eq. C.1 (selection-bias weight, single-scale limit)**:
  `B_sel(λ_obs, λ_true, z) = 1 + (Δ_prj − <Δ_prj>) / <Δ_prj>`, with
  `<Δ_prj>(λ_obs, z) = f_prj / τ` and `Δ_prj = max(λ_obs − λ_true, 0)`.
  The full scale-dependent Costanzi form (with `eff_bias_ltr` and
  `bar_delta_prj_Beff`) is **not** implemented here — see "Known
  simplifications" below.

Copy the exact equation numbering/text from the PDF into this section
before circulating.

## Binning

Match the edges already used by `createDataVector.ipynb`:

- `LBDBINS = [5, 15, 25, 40, 160]` (4 λ_obs bins).
- `ZMIN_LIST = [0.20, 0.37, 0.51]`, `ZMAX_LIST = [0.32, 0.51, 0.64]`
  (3 redshift bins; avoids the Buzzard 0.33–0.37 seam).

These are the bin edges a future Costanzi fit should adopt; feel free to
retune once you have Y1-style λ distributions from this mock.

## How to run

1. On NERSC Perlmutter, launch Jupyter with the `desc-python` kernel:
   ```
   ssh $NERSC && module load python && source activate desc-python
   jupyter lab
   ```
   (Or use jupyter.nersc.gov and pick `desc-python`.)

2. Open `fnalFall2022/y3kp/buzzardMock/Xin_MockDataVector.ipynb`
   and run top to bottom. Expected wall-clock time: a few minutes.

3. Output:
   `floc.mock_fname`-sibling file
   `dataVec_mock_buzzard_xin_v0.hdf5` with groups `NC`, `GT`, `params`.

## Files the notebook reads

All resolved via `FileLocs(machine='nersc')` in `fileLoc.py`:

- `halo_run_fname` — raw halo catalogue (Mvir, z, RA, DEC, pid, cosi, …).
- `profile_output_fname` — per-halo ΔΣ(R) profiles.
- `mock_boost_factor_1d` → `beta_table_zl_y1_like.npz` (β_eff, z_lens
  grid used to build Σ_crit^{-1}).

Plus local to this directory:

- `prj_params_DESY3_lss_lin_dep_getdist_v1.txt` — Costanzi posterior-sample
  table for the projection model.
- `radial_bins_phys_mpc.py` — `rp_phys_mpc` 15-bin radial grid.
- `costanzi_selection.py` — helper module (mass-richness + projection +
  `b_sel_one_halo`).

## Verification checks run in the notebook

1. End-to-end execution — all cells succeed, HDF5 is written, plots render.
2. λ_true diagnostic — `<λ_true>`, fraction `≥5`, `≥20` printed; hexbin of
   `log λ_true` vs `log M` with the `l_tr(M, z)` mean curves overlaid at
   z = 0.25, 0.45, 0.60.
3. Projection conservation — `assert len(λ_obs) == len(λ_true)` (no halos
   added or removed by the remapping).
4. Projection tail — histogram of `λ_obs − λ_true > 0` is a decaying
   exponential; `<Δλ>` is printed next to the predicted `<f_prj / τ>`.
5. `B_sel` sanity — `<B_sel>` printed over all halos and over
   `λ_obs ≥ 20`; should sit near 1 (by construction).
6. Weight regression — `stacker.run(weights=ones)` equals the unweighted
   mean stack (asserted with `np.allclose`).

## Known simplifications / deferred work

- **Single-scale B_sel**: the one-halo (small-scale) limit of
  Costanzi App. C is used. The full scale-dependent form (nested over
  halo bias, M–λ_true joint, redshift kernel) requires the Costanzi
  `bar_delta_prj_Beff` / `eff_bias_ltr` machinery and a radial argument
  in the weight. Add in a follow-up.
- **Boost-factor contamination** not modelled yet — this is the other
  main Y3 systematic the user flagged.
- Projection parameters use the posterior mean; row-by-row covariance
  propagation is not done.
- Masking fraction `f_msk` and centering mis-identification are not
  included.

## References

- Costanzi et al. 2026, *Forward analytical model for the optical
  selection bias on galaxy cluster lensing profiles*, arXiv:2604.05833.
  Eqs. 15 (mass-richness) and C.1 (selection-bias weight) drive the
  implementation.
- `MCostanzi/SelectionBias` — upstream reference notebook and the
  `prj_params_DESY3_lss_lin_dep_getdist_v1.txt` parameter file.
- `0-MakeMock.ipynb` — mock halo selection recipe (copied verbatim).
- `createDataVector.ipynb` — Σ_crit construction, `compute_number_counts`,
  and the stacking / binning scheme this notebook mirrors.
