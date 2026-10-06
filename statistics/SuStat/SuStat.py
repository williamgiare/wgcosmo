"""Gaussian approximation to suspiciousness from weighted MCMC chains.

The chi-square calibration assumes approximately Gaussian posteriors in the
chosen common parameter space; it is not the general evidence-based statistic.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm


def integrand(x, d):
    """Chi-square density, retained for compatibility with older notebooks."""
    return chi2.pdf(x, d)


def get_chain(root, params, fburn=0.3, verbose=False, include_weights=False):
    """Read every numbered Cobaya chain, discarding a fraction of each file.

    With include_weights=False the historical parameter-only DataFrame is
    returned. Statistical calculations must request include_weights=True.
    """
    if not np.isfinite(fburn) or not 0 <= fburn < 1:
        raise ValueError("fburn must be a fraction in [0, 1)")
    params = list(params)
    if not params or len(set(params)) != len(params) or "weight" in params:
        raise ValueError("params must contain distinct parameter names, excluding weight")
    root = Path(root)
    pattern = re.compile(re.escape(root.name) + r"\.(\d+)\.txt$")
    files = sorted(
        (p for p in root.parent.glob(root.name + ".*.txt") if pattern.fullmatch(p.name)),
        key=lambda p: int(pattern.fullmatch(p.name).group(1)),
    )
    if not files:
        raise FileNotFoundError(f"No numbered chains found with root: {root}")
    chains = []
    for path in files:
        with path.open() as handle:
            header = handle.readline().lstrip("#").split()
        if not header or len(header) != len(set(header)):
            raise ValueError(f"Invalid chain header in {path}")
        frame = pd.read_csv(path, sep=r"\s+", comment="#", skiprows=1, names=header)
        required = ["weight"] + params
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        frame = frame.iloc[int(fburn * len(frame)):][required].astype(float)
        if frame.empty or not np.isfinite(frame.to_numpy()).all():
            raise ValueError(f"Empty or non-finite chain after burn-in: {path}")
        if (frame.weight < 0).any():
            raise ValueError(f"Negative chain weights in {path}")
        frame = frame.loc[frame.weight > 0]
        if frame.empty:
            raise ValueError(f"No positive chain weights in {path}")
        chains.append(frame)
        if verbose:
            print(f"Reading {path}: {len(frame)} rows after burn-in")
    result = pd.concat(chains, ignore_index=True)
    return result if include_weights else result[params]


def _weighted_moments(chain, params):
    values = chain[params].to_numpy(dtype=float)
    weights = chain.weight.to_numpy(dtype=float)
    mean = np.average(values, axis=0, weights=weights)
    centered = values - mean
    # Posterior population moments: invariant under rescaling or splitting weights.
    covariance = (centered.T * weights) @ centered / weights.sum()
    return mean, covariance


def get_sus(root_A, root_B, params, fburn=0.3, verbose=True,
            get_results=False, get_latex=False, model=None):
    """Return (chi2, logS, sigma) when get_results=True, as in the old API."""
    params = list(params)
    chain_A = get_chain(root_A, params, fburn, verbose, include_weights=True)
    chain_B = get_chain(root_B, params, fburn, verbose, include_weights=True)
    mean_A, cov_A = _weighted_moments(chain_A, params)
    mean_B, cov_B = _weighted_moments(chain_B, params)
    delta = mean_A - mean_B
    covariance = cov_A + cov_B
    try:
        factor = np.linalg.cholesky(covariance)
        whitened = np.linalg.solve(factor, delta)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Combined posterior covariance must be positive definite; "
                         "remove fixed or linearly dependent parameters") from exc
    statistic = float(whitened @ whitened)
    d = len(params)
    logS = (d - statistic) / 2
    # Survival functions avoid quadrature over an unnecessarily large interval
    # and the cancellation in erfinv(1-p) at large tensions.
    p = float(chi2.sf(statistic, d))
    sigma = float(norm.isf(p / 2))
    if verbose:
        print("SuStat (Gaussian approximation, weighted posterior moments)")
        print(f"Dimension: {d}; Chi2={statistic:.6g}; p={p:.6g}; "
              f"logS={logS:.6g}; sigma={sigma:.6g}")
    if get_results:
        return statistic, logS, sigma
    if get_latex:
        print(f"{model} & ${d}$ & ${statistic:.3g}$ & ${p:.3g}$ & "
              f"${logS:.3g}$ & ${sigma:.3g}\\,\\sigma$ \\\\")
        return sigma
