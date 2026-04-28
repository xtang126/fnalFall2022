"""Costanzi et al. 2026 (arXiv:2604.05833) optical-selection forward model.

Ported from https://github.com/MCostanzi/SelectionBias (notebook
``Analytical modeling optical selection effects on cluster density profile``).
Implements:

- Eq. 15 mass-richness relation  <lambda_true | M, z>  (compound Poisson +
  lognormal intrinsic scatter) and the corresponding PDF ``pltr_M``.
- P(lambda_obs | lambda_true, z) as a delta + positive-exponential mixture.
- B_sel(lambda_obs, lambda_true, z) one-halo (small-scale) limit weight.

Parameters are the DES-Y1 NC+3x2pt best fit hard-coded in the upstream
notebook.
"""
import os
import numpy as np
import scipy.special as spc
from scipy.interpolate import InterpolatedUnivariateSpline as ius

# ---------------------------------------------------------------------------
# DES-Y1 NC+3x2pt best-fit intrinsic scaling relation parameters
# ---------------------------------------------------------------------------
M_MIN = 10. ** 1.13852818e+01
ALPHA = 8.58693714e-01
M_1 = 10. ** 1.26964410e+01
M_PIVOT = M_1 - M_MIN
SIGM_INTR = 1.80949022e-01
EPSI = 2.83887020e-01
PIVOT_Z0 = 0.4544

DEFAULT_PRJ_PARAM_FILE = os.path.join(
    os.path.dirname(__file__), 'prj_params_DESY3_lss_lin_dep_getdist_v1.txt'
)
PRJ_Z_BINS = np.linspace(0.10, 0.80, 15)


# ---------------------------------------------------------------------------
# Eq. 15:  <lambda_true | M, z>  and P(lambda_true | M, z)
# ---------------------------------------------------------------------------
def l_sat(M, z, Mmin=M_MIN, Mpivot=M_PIVOT, a=ALPHA, epsilon=EPSI,
          pivot_z=PIVOT_Z0):
    return ((M - Mmin) / Mpivot) ** a * ((1. + z) / (1. + pivot_z)) ** epsilon


def l_tr(M, z):
    return 1. + l_sat(M, z)


def sig_intr(M, z):
    return SIGM_INTR * l_sat(M, z)


def pltr_M(ltr, M, z):
    """P(lambda_true | M, z). Compound Poisson + lognormal — Eq. 15."""
    m = l_sat(M, z)
    std = np.sqrt(m + (m * SIGM_INTR) ** 2.)
    x = ltr + (m * SIGM_INTR) ** 2.
    lam = std ** 2.
    ln_gamma_fun = spc.gammaln(x)
    return np.exp(-lam + (x - 1.) * np.log(lam) - ln_gamma_fun,
                  dtype='float128')


def sample_lambda_true(M, z, rng=None):
    """Monte-Carlo draw of lambda_true from pltr_M via its compound-Poisson
    generative form:  N_sat ~ Poisson(m), with m = l_sat(M,z) broadened by
    a lognormal of width SIGM_INTR (i.e. the 'compound Poisson + lognormal'
    construction of Costanzi+2019).

    Returns an integer-valued ``lambda_true`` per halo.
    """
    rng = np.random.default_rng() if rng is None else rng
    m = np.asarray(l_sat(M, z))
    # lognormal broadening of the mean
    m_broad = m * rng.lognormal(mean=-0.5 * SIGM_INTR ** 2, sigma=SIGM_INTR,
                                size=m.shape)
    m_broad = np.clip(m_broad, 0., None)
    n_sat = rng.poisson(m_broad)
    return 1 + n_sat  # the central BCG contributes the "1+" in l_tr


# ---------------------------------------------------------------------------
# Projection-parameter file: column layout (a_tau, b_tau, a_mu, b_mu, a_sig,
# b_sig, a_fprj, b_fprj, a_fmsk, b_fmsk) on a grid of 15 redshift knots.
# ---------------------------------------------------------------------------
def load_prj_interpolators(path=DEFAULT_PRJ_PARAM_FILE, row=None):
    """Return a dict of ``ius(z)`` interpolators for each of the 10 columns.

    If ``row`` is an integer we use that single posterior sample (constant
    in z). Otherwise we treat the 15 rows as 15 redshift knots, matching the
    upstream notebook's convention (PRJ_Z_BINS).
    """
    params = np.loadtxt(path).T  # shape (10, 15)
    if row is None:
        interp = {}
        labels = ['a_tau', 'b_tau', 'a_mu', 'b_mu', 'a_sig', 'b_sig',
                  'a_fprj', 'b_fprj', 'a_fmsk', 'b_fmsk']
        for i, key in enumerate(labels):
            interp[key] = ius(PRJ_Z_BINS, params[i, :], k=1)
        return interp

    const = params[:, row]
    labels = ['a_tau', 'b_tau', 'a_mu', 'b_mu', 'a_sig', 'b_sig',
              'a_fprj', 'b_fprj', 'a_fmsk', 'b_fmsk']
    return {k: (lambda v=const[i]: (lambda z: np.full_like(np.asarray(z,
                                                                      dtype=float), v)))()
            for i, k in enumerate(labels)}


# Upstream notebook's parameter-to-function forms (lin := lambda_true)
def _tau_model(lin, a, b):
    return b / lin ** a


def _fprj_model(lin, a, b):
    return b / (1. + np.exp(-lin / 1.)) ** a


def _mu_model(lin, a, b):
    return a + b * lin


def _sig_model(lin, a, b):
    return b * lin ** a


# ---------------------------------------------------------------------------
# P(lambda_obs | lambda_true, z) — delta + exponential projection mixture
# ---------------------------------------------------------------------------
def projection_params(ltr, z, interp):
    """Return (f_prj, tau) arrays at each (ltr, z)."""
    ltr = np.asarray(ltr, dtype=float)
    z = np.asarray(z, dtype=float)
    f_prj = _fprj_model(ltr, interp['a_fprj'](z), interp['b_fprj'](z))
    tau = _tau_model(ltr, interp['a_tau'](z), interp['b_tau'](z))
    f_prj = np.clip(f_prj, 0., 1.)
    tau = np.clip(tau, 1e-6, None)
    return f_prj, tau


def sample_lambda_obs(ltr, z, interp, rng=None):
    """Draw lambda_obs ~ P(lambda_obs | lambda_true, z).

    Mixture:   with prob (1 - f_prj)  lambda_obs = lambda_true
               else                   lambda_obs = lambda_true + Exp(1/tau)
    """
    rng = np.random.default_rng() if rng is None else rng
    ltr = np.asarray(ltr, dtype=float)
    z = np.asarray(z, dtype=float)
    f_prj, tau = projection_params(ltr, z, interp)
    u = rng.uniform(size=ltr.shape)
    boost = rng.exponential(scale=1. / tau)
    lob = np.where(u < f_prj, ltr + boost, ltr)
    return lob, f_prj, tau


# ---------------------------------------------------------------------------
# B_sel weight (one-halo / small-scale limit)
# ---------------------------------------------------------------------------
def b_sel_one_halo(lob, ltr, z, interp):
    """Per-halo B_sel weight in the small-scale (single-scale) limit.

    We use the closed-form  <Delta_prj | lob, z> = f_prj / tau  ≡ DPRJ(lob,z)
    expectation of the exponential mixture tail, so that each halo's weight is

        w = (Delta_prj_obs - DPRJ) / DPRJ + 1.

    The "+1" absorbs the Costanzi "boost_bias = 1 + slope * (DPRJ_i - DPRJ)/DPRJ"
    normalization. In this form the population mean over the δ+exp mixture
    recovers 1, so B_sel = 1 corresponds to no selection bias.
    """
    lob = np.asarray(lob, dtype=float)
    ltr = np.asarray(ltr, dtype=float)
    z = np.asarray(z, dtype=float)
    f_prj, tau = projection_params(lob, z, interp)
    dprj_bar = f_prj / tau           # mean projection boost at given lob,z
    dprj = np.maximum(lob - ltr, 0.)  # per-halo projection excess
    with np.errstate(divide='ignore', invalid='ignore'):
        w = 1. + (dprj - dprj_bar) / np.where(dprj_bar > 0, dprj_bar, 1.)
    return np.clip(w, 0., None)


# ---------------------------------------------------------------------------
# Convenience: posterior-mean projection interpolator
# ---------------------------------------------------------------------------
def load_prj_posterior_mean(path=DEFAULT_PRJ_PARAM_FILE):
    """Collapse the 15 posterior-sample rows to a single posterior-mean set
    of constants (useful when the 15 rows are interpreted as samples rather
    than redshift knots). Returns the same interpolator-dict signature so
    the rest of the pipeline does not change.
    """
    params = np.loadtxt(path)           # (15, 10)
    mean = params.mean(axis=0)          # (10,)
    labels = ['a_tau', 'b_tau', 'a_mu', 'b_mu', 'a_sig', 'b_sig',
              'a_fprj', 'b_fprj', 'a_fmsk', 'b_fmsk']
    return {k: (lambda v=mean[i]: (lambda z: np.full_like(
                np.asarray(z, dtype=float), v)))()
            for i, k in enumerate(labels)}
