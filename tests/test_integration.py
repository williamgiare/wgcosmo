"""Integration against real theory engines; no external CMB data download."""
from pathlib import Path

import numpy as np
import pytest
from cobaya.model import get_model

from conftest import ROOT, load_module


def test_cobaya_camb_all_distance_likelihoods():
    pytest.importorskip("camb")
    likelihoods = {}
    for path in (ROOT / "likelihoods").rglob("*.py"):
        relative = path.relative_to(ROOT)
        if any(part.startswith(".") for part in relative.parts) or "CMB_compressed" in path.parts or "Theory" in path.parts:
            continue
        module = load_module(relative.as_posix())
        key = "_".join(relative.with_suffix("").parts)
        likelihoods[key] = {"external": getattr(module, path.stem)}
    info = {
        "theory": {"camb": {}},
        "params": {"H0": 67.4, "ombh2": 0.0224, "omch2": 0.12,
                   "As": 2.1e-9, "ns": 0.965, "tau": 0.054, "mnu": 0.06, "M": -19.3},
        "likelihood": likelihoods,
    }
    with get_model(info) as model:
        result = model.logposterior([])
        assert len(result.loglikes) == 22
        assert np.isfinite(result.loglikes).all()


@pytest.mark.parametrize("basis", [["omega_b", "omega_m", "theta_s"],
                                  ["omega_b", "omega_m", "1/theta_s"],
                                  ["omega_b", "omega_m", "1/theta_drag"]])
def test_cobaya_class_compressed_cmb(basis):
    pytest.importorskip("classy")
    cls = load_module("likelihoods/CMB_compressed/compressed_CMB.py").CompressedCMB
    info = {
        "theory": {"classy": {}},
        "params": {"H0": 67.4, "omega_b": 0.0224, "omega_cdm": 0.12,
                   "A_s": 2.1e-9, "n_s": 0.965, "tau_reio": 0.054,
                   "N_ncdm": 1, "m_ncdm": 0.06, "N_ur": 2.0328, "z_d": {"derived": True}},
        "likelihood": {"compressed_CMB": {"external": cls, "compression_basis": basis}},
    }
    with get_model(info) as model:
        result = model.logposterior([])
        assert len(result.loglikes) == 1
        assert np.isfinite(result.loglikes[0])


def test_real_cobaya_chain_evidence(tmp_path):
    from cobaya.run import run
    be = load_module("statistics/MCMC_Evidence/Cobaya_wrapper.py")
    info = {
        "likelihood": {"gaussian": {"external": "lambda x, y: -0.5*(x*x + y*y)"}},
        "params": {name: {"prior": {"dist": "norm", "loc": 0, "scale": 1},
                           "ref": 0, "proposal": 0.7} for name in ["x", "y"]},
        "sampler": {"mcmc": {"seed": 831, "max_samples": 10000,
                             "learn_proposal": False, "Rminus1_stop": 0,
                             "Rminus1_cl_stop": 0}},
        "output": str(tmp_path / "chain"),
    }
    _, sampler = run(info)
    sampler.model.close()
    actual = be.MCMC_Evidence(tmp_path / "chain", burnlen=0.3, verbose=False, get_results=True)
    # Integral of exp(-|x|^2/2) against a normalized 2D N(0,I) prior.
    assert actual == pytest.approx(-np.log(2), abs=0.20)
