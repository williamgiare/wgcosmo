"""Bayesian evidence from the full, normalized Cobaya posterior density.

All sampled parameters (including nuisance parameters) must be included. The
old cosmological-only selection and Gaussian-to-box prior approximation are
not valid full-model evidence calculations and are rejected explicitly.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import yaml

try:
    from .MCEvidence import MCEvidence
except ImportError:
    from MCEvidence import MCEvidence


def _chain_files(root):
    root = Path(root)
    pattern = re.compile(re.escape(root.name) + r"\.(\d+)\.txt$")
    files = sorted(
        (p for p in root.parent.glob(root.name + ".*.txt") if pattern.fullmatch(p.name)),
        key=lambda p: int(pattern.fullmatch(p.name).group(1)),
    )
    if not files:
        raise FileNotFoundError(f"No numbered chains found with root: {root}")
    return files


def _read_chain(path, burnlen=0):
    if not np.isfinite(burnlen) or burnlen < 0 or (burnlen >= 1 and burnlen != int(burnlen)):
        raise ValueError("burnlen must be a fraction in [0, 1) or an integer row count")
    with Path(path).open() as handle:
        names = handle.readline().lstrip("#").split()
    if not names or len(names) != len(set(names)):
        raise ValueError(f"Invalid chain header in {path}")
    chain = pd.read_csv(path, sep=r"\s+", comment="#", skiprows=1, names=names)
    start = int(burnlen * len(chain)) if burnlen < 1 else int(burnlen)
    chain = chain.iloc[start:].copy()
    if chain.empty:
        raise ValueError(f"No samples left after burn-in in {path}")
    return chain


def _metadata(root):
    path = Path(str(root) + ".updated.yaml")
    if not path.is_file():
        raise FileNotFoundError(f"Need {root}.updated.yaml to identify all sampled parameters and priors")
    with path.open() as handle:
        info = yaml.safe_load(handle)
    if not isinstance(info, dict) or not isinstance(info.get("params"), dict):
        raise ValueError(f"Invalid params metadata in {path}")
    return info


def _parameter_names(params, info):
    sampled = [name for name, spec in info["params"].items()
               if isinstance(spec, dict) and spec.get("prior") is not None]
    if not sampled:
        raise ValueError("No sampled parameters with priors found in Cobaya metadata")
    if params is None:
        return sampled
    requested = list(params)
    # Changing the prior after sampling requires importance reweighting, rather
    # than just changing a box volume or a Gaussian confidence interval.
    if any(":" in p or ";" in p for p in requested):
        raise ValueError("Prior overrides (:min/max or ;CL) are no longer supported. "
                         "Use the priors stored in the Cobaya YAML; a different prior "
                         "requires a new or importance-reweighted chain")
    if len(set(requested)) != len(requested) or set(requested) != set(sampled):
        missing = sorted(set(sampled) - set(requested))
        extra = sorted(set(requested) - set(sampled))
        raise ValueError(f"Evidence requires every sampled parameter, including nuisance parameters. "
                         f"Missing: {missing}; extra: {extra}. Use params=None for automatic selection")
    return requested


def _posterior_arrays(root, params=None, burnlen=0.3, external_prior_log_normalization=None):
    info = _metadata(root)
    names = _parameter_names(params, info)
    mcmc = (info.get("sampler") or {}).get("mcmc") or {}
    if isinstance(mcmc, dict) and float(mcmc.get("temperature", 1)) != 1:
        raise ValueError("Evidence requires an untempered posterior chain (temperature=1)")
    arrays = []
    has_external = bool(info.get("prior"))
    for path in _chain_files(root):
        frame = _read_chain(path, burnlen)
        missing = set(["weight"] + names) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        external_columns = [c for c in frame if c.startswith("minuslogprior__") and c != "minuslogprior__0"]
        has_external = has_external or bool(external_columns)
        if "minuslogpost" in frame:
            neglogdensity = frame.minuslogpost.to_numpy(dtype=float)
        elif {"chi2", "minuslogprior"}.issubset(frame.columns):
            neglogdensity = (0.5 * frame.chi2 + frame.minuslogprior).to_numpy(dtype=float)
        else:
            raise ValueError(f"Need minuslogpost or both chi2 and minuslogprior in {path}")
        if {"minuslogpost", "chi2", "minuslogprior"}.issubset(frame.columns):
            if not np.allclose(neglogdensity, 0.5 * frame.chi2 + frame.minuslogprior, rtol=1e-6, atol=1e-5):
                raise ValueError(f"Posterior, likelihood and prior columns are inconsistent in {path}")
        array = np.column_stack([frame.weight, neglogdensity, frame[names]])
        if not np.isfinite(array).all() or np.any(array[:, 0] < 0):
            raise ValueError(f"Non-finite values or negative weights in {path}")
        array = array[array[:, 0] > 0]
        if not len(array):
            raise ValueError(f"No positive weights in {path}")
        arrays.append(array)
    if has_external and external_prior_log_normalization is None:
        raise ValueError("External priors have unknown normalization. Supply "
                         "external_prior_log_normalization=log(integral of the joint prior)")
    normalization = 0.0 if external_prior_log_normalization is None else float(external_prior_log_normalization)
    if not np.isfinite(normalization):
        raise ValueError("External prior log-normalization must be finite")
    for array in arrays:
        array[:, 1] += normalization
    return arrays, names


def _thin_arrays(arrays, thin_factor=None):
    """Systematic thinning in stored multiplicity units, separately per chain.

    The default uses the largest multiplicity, so retained points have unit
    weights. This mitigates the inverse-weight bias of the nearest-neighbour
    estimator on compressed rejection chains; it is not a convergence test.
    Set zero to retain original weights or choose a larger factor to further
    reduce autocorrelation.
    """
    if thin_factor is None:
        all_weights = np.concatenate([array[:, 0] for array in arrays])
        if np.all(all_weights == all_weights[0]):
            return arrays
        thin_factor = float(all_weights.max())
    if not np.isfinite(thin_factor) or thin_factor < 0:
        raise ValueError("thin_factor must be finite and non-negative")
    if thin_factor == 0:
        return arrays
    thinned = []
    for array in arrays:
        cumulative = np.cumsum(array[:, 0]) / thin_factor
        counts = np.diff(np.r_[0, np.floor(np.nextafter(cumulative, np.inf))])
        keep = counts > 0
        if not keep.any():
            raise ValueError("Thinning removes every sample in a chain; reduce thin_factor")
        result = array[keep].copy()
        result[:, 0] = counts[keep]
        thinned.append(result)
    return thinned


def BayesianEvidence(root, burnlen=0.3, params=None, verbose=False,
                     external_prior_log_normalization=None, thin_factor=None):
    """Return the k=1 estimate of log Z as a one-element array.

    The normalized one-dimensional priors, including Gaussian priors, are
    already in minuslogpost. MCEvidence integrates that posterior density with
    priorvolume=1, so no extra box-volume correction is applied.
    """
    arrays, names = _posterior_arrays(root, params, burnlen, external_prior_log_normalization)
    arrays = _thin_arrays(arrays, thin_factor)
    if verbose:
        print(f"Evidence uses {sum(len(a) for a in arrays)} rows in {len(names)} dimensions after burn-in/thinning")
    # Repeated coordinates have zero nearest-neighbour distance. Merge their
    # multiplicities without changing the posterior distribution.
    combined = np.concatenate(arrays)
    _, first, inverse = np.unique(combined[:, 2:], axis=0, return_index=True, return_inverse=True)
    if len(first) != len(combined):
        if not np.allclose(combined[:, 1], combined[first[inverse], 1], rtol=1e-7, atol=1e-6):
            raise ValueError("Identical parameter points have inconsistent posterior densities")
        unique = combined[first].copy()
        unique[:, 0] = np.bincount(inverse, weights=combined[:, 0])
        arrays = [unique]
    if sum(len(a) for a in arrays) <= len(names) + 1:
        raise ValueError("Too few posterior samples for evidence estimation")
    mce = MCEvidence(arrays, ndim=len(names), priorvolume=1.0, kmax=2,
                     verbose=int(verbose), burnlen=0, thinlen=0)
    return mce.evidence(pos_lnp=False, nproc=1)


def MCMC_Evidence(root, params=None, burnlen=0.3, verbose=True,
                  get_results=False, labels=False, external_prior_log_normalization=None, thin_factor=None):
    """Estimate the full model's log Z without writing intermediate chain files."""
    logZ = float(BayesianEvidence(root, burnlen, params, verbose,
                                 external_prior_log_normalization, thin_factor)[0])
    if verbose:
        prefix = f"[{labels}] " if labels else ""
        print(f"{prefix}log Z [k=1] = {logZ:.8g}")
    if get_results:
        return logZ


def match_CosmoMC_chains(root, params, verbose=False):
    """Legacy export helper; MCMC_Evidence itself no longer needs this export."""
    names = [p.split(":", 1)[0].split(";", 1)[0] for p in params]
    for path in _chain_files(root):
        frame = _read_chain(path)
        frame["loglike"] = 0.5 * frame["chi2"]
        columns = ["weight", "loglike"] + names
        destination = path.with_name(path.name.replace(Path(root).name + ".", Path(root).name + "_BE.", 1))
        frame[columns].to_csv(destination, header=False, index=False, sep=" ")
        if verbose:
            print(f"Exported {destination}")


def get_dot_ranges(root, params, verbose=False):
    """Export finite uniform-prior ranges for legacy MCEvidence callers.

    Gaussian priors cannot be represented by a finite uniform box.
    """
    info = _metadata(root)
    names = _parameter_names(params, info)
    rows = []
    for name in names:
        prior = info["params"][name]["prior"]
        if isinstance(prior, (list, tuple)) and len(prior) == 2:
            lower, upper = map(float, prior)
        elif isinstance(prior, dict) and prior.get("dist", "uniform") == "uniform":
            lower = float(prior.get("min", prior.get("loc", 0)))
            upper = float(prior.get("max", lower + prior.get("scale", 1)))
        else:
            raise ValueError(f"Prior for {name} is not uniform; use MCMC_Evidence directly")
        if not np.isfinite([lower, upper]).all() or upper <= lower:
            raise ValueError(f"Invalid uniform prior for {name}")
        rows.append((name, lower, upper))
    destination = str(root) + "_BE.ranges"
    pd.DataFrame(rows).to_csv(destination, header=False, index=False, sep=" ")
    if verbose:
        print(f"Exported {destination}")


def cleaning_up(root):
    """Remove only explicitly exported legacy _BE files."""
    for path in Path(root).parent.glob(Path(root).name + "_BE.*"):
        if re.fullmatch(re.escape(Path(root).name) + r"_BE\.(\d+\.txt|ranges)", path.name):
            path.unlink()
