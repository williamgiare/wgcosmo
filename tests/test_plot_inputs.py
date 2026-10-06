"""Exercise notebook helpers with supplied inputs, without replacing saved figures."""

import ast
import json
from types import SimpleNamespace
import warnings

import camb
from getdist import MCSamples
import numpy as np
import pandas as pd
import pytest

from conftest import ROOT


def cells(relative):
    return json.loads((ROOT / "plots" / relative).read_text())["cells"]


def execute(relative, indices, namespace):
    for index in indices:
        exec("".join(cells(relative)[index]["source"]), namespace)
    return namespace


def test_plot_notebook_syntax():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        for path in (ROOT / "plots").rglob("*.ipynb"):
            if "AxionLimits-master" in path.parts or ".ipynb_checkpoints" in path.parts:
                continue
            for cell in json.loads(path.read_text())["cells"]:
                if cell["cell_type"] == "code":
                    source = "\n".join(
                        "" if line.lstrip().startswith(("%", "!")) else line
                        for line in "".join(cell["source"]).splitlines()
                    )
                    ast.parse(source, filename=str(path))


def test_bao_bestfit_handles_numbered_chains_and_tied_rows(tmp_path):
    root = tmp_path / "sample"
    np.savetxt(str(root) + ".2.txt", [[1, 5, 147]], header="weight chi2 rs_drag")
    np.savetxt(
        str(root) + ".12.txt", [[1, 3, 148], [1, 3, 149]], header="weight chi2 rs_drag"
    )
    ns = execute("BAO/BAOs.ipynb", [1, 2], {"np": np, "pd": pd})
    assert ns["get_chi2"](root) == 3
    assert ns["get_bestfit"](root)["rs_drag"] == 148
    with pytest.raises(FileNotFoundError):
        ns["get_bestfit"](tmp_path / "missing")


def test_sn_bestfit_accepts_a_single_row_dataframe():
    bestfit = pd.DataFrame([{
        "H0,": 67.4, "omega_b,": 0.022, "omega_cdm,": 0.12,
        "tau_reio,": 0.054, "n_s,": 0.965, "ln10^{10}A_s,": np.log(21),
        "w0_fld,": -0.9, "M,": -19.3,
    }])
    ns = execute("SN/SN.ipynb", [2, 3], {
        "np": np, "pd": pd, "camb": camb, "model": camb.model,
    })
    params = ns["set_cosmo_params"](bestfit)
    assert params.H0 == pytest.approx(67.4)
    assert params.DarkEnergy.w == pytest.approx(-0.9)
    assert params.InitPower.As == pytest.approx(2.1e-9)
    distances = ns["get_distance_moduli"](bestfit, np.array([0.1, 0.5, 1.0]), verbose=False)
    assert np.isfinite(distances).all()
    assert (np.diff(distances) > 0).all()
    with pytest.raises(ValueError, match="exactly one"):
        ns["set_cosmo_params"](pd.concat([bestfit, bestfit]))


def test_cpl_accepts_a_numpy_redshift_grid():
    rng = np.random.default_rng(731)
    samples = MCSamples(
        samples=rng.normal([-1, 0.2], [0.1, 0.2], size=(5000, 2)),
        weights=rng.integers(1, 5, 5000), names=["w", "wa"], labels=["w", "w_a"],
    )
    ns = execute("DE_EoS/CPL.ipynb", [1, 2], {
        "np": np, "pd": pd,
        "getdist": SimpleNamespace(loadMCSamples=lambda *args, **kwargs: samples),
    })
    z = np.array([0, 0.5, 2])
    result = ns["get_w"]("provided-chain", redshift=z)
    expected = samples.mean("w") + samples.mean("wa") * z / (1 + z)
    np.testing.assert_array_equal(result["z"], z)
    np.testing.assert_allclose(result["mean"], expected)
    assert (result["lower_68"] <= result["upper_68"]).all()
