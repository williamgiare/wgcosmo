from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT, load_module

PATHS = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "likelihoods").rglob("*.py")
               if not any(part.startswith(".") for part in p.relative_to(ROOT).parts))


class Provider:
    def get_param(self, name):
        return {"rdrag": 147.0, "M": -19.3, "w": -0.9, "wa": 0.1,
                "w0_fld": -1.1, "wa_fld": -0.1, "omega_b": 0.0223,
                "Omega_m": 0.31, "H0": 67.4, "theta_s_100": 1.0418,
                "z_d": 1060, "rs_drag": 147}[name]

    def get_angular_diameter_distance(self, z):
        z = np.asarray(z)
        return 4200 * z / ((1 + z) * (1 + 0.3 * z))

    def get_Hubble(self, z, units=None):
        return 67.4 * np.sqrt(0.31 * (1 + np.asarray(z))**3 + 0.69)


def instance(relative, options=None):
    module = load_module(relative)
    name = Path(relative).stem
    cls = getattr(module, name if name != "compressed_CMB" else "CompressedCMB")
    return cls(options or {}, name="test_" + name)


@pytest.mark.parametrize("relative", PATHS)
def test_all_likelihoods_initialize_and_evaluate(relative, monkeypatch, tmp_path):
    # A different working directory must not change the packaged data paths.
    monkeypatch.chdir(tmp_path)
    like = instance(relative)
    like.provider = Provider()
    assert isinstance(like.get_requirements(), dict)
    value = like.logp()
    assert np.asarray(value).size == 1
    assert np.isfinite(value)
    assert float(value) <= 0


@pytest.mark.parametrize("name", ["Pantheon_Plus", "Pantheon_Plus_SH0ES"])
def test_supernova_matches_full_matrix_reference(name):
    like = instance(f"likelihoods/SN/{name}/{name}.py")
    shared = ROOT / "likelihoods/SN/data/Pantheon+SH0ES_STAT+SYS.cov"
    assert Path(like.path_covmat).resolve() == shared
    like.provider = Provider()
    selected = like.light_curve_params.loc[like._mask]
    calibrator = selected.IS_CALIBRATOR.to_numpy() == 1 if name.endswith("SH0ES") else np.zeros(len(selected), bool)
    moduli = np.empty(len(selected))
    z, zh = selected.zHD.to_numpy(), selected.zHEL.to_numpy()
    moduli[~calibrator] = 5 * np.log10((1 + z[~calibrator]) * (1 + zh[~calibrator]) * like.provider.get_angular_diameter_distance(z[~calibrator])) + 25
    if calibrator.any():
        moduli[calibrator] = selected.CEPH_DIST.to_numpy()[calibrator]
    residual = selected.m_b_corr.to_numpy() + 19.3 - moduli
    covariance = like.C00[np.ix_(like._mask, like._mask)]
    covariance = np.tril(covariance) + np.tril(covariance, -1).T
    expected = -0.5 * residual @ np.linalg.solve(covariance, residual)
    assert like.logp() == pytest.approx(expected, rel=1e-10)


def test_ddr_uses_shared_supernova_covariance(monkeypatch, tmp_path):
    from getdist import IniFile

    monkeypatch.chdir(tmp_path)
    shared = ROOT / "likelihoods/SN/data/Pantheon+SH0ES_STAT+SYS.cov"
    like = instance("likelihoods/SN/Pantheon_Plus/Pantheon_Plus.py")
    data = load_module("plots/DDR_plots/Data.py").Data(theory=SimpleNamespace())
    assert data.SN_covmat_path == shared
    np.testing.assert_array_equal(data.SN_covariance(), like.C00)
    assert len(data.SN()) > 0
    for name in ["Pantheon_Plus", "Pantheon_Plus_SH0ES"]:
        base = ROOT / "likelihoods/SN" / name
        ini = IniFile(str(base / "data/pantheon_plus.dataset"))
        assert (base / ini.string("mag_covmat_file")).resolve() == shared


def test_supernova_selection_works_with_interleaved_rows(tmp_path):
    rows = pd.DataFrame({"zHD": [0.1, 0.001, 0.2, 0.003], "zHEL": [0.1, 0.001, 0.2, 0.003],
                         "m_b_corr": [19, 10, 21, 11]})
    data, cov = tmp_path / "sn.dat", tmp_path / "sn.cov"
    rows.to_csv(data, sep=" ", index=False)
    covariance = np.diag([1.0, 100.0, 2.0, 200.0])
    with cov.open("w") as handle:
        handle.write("4\n")
        np.savetxt(handle, covariance.reshape(-1))
    like = instance("likelihoods/SN/Pantheon_Plus/Pantheon_Plus.py", {"path_lc": str(data), "path_covmat": str(cov)})
    np.testing.assert_allclose(like.cov @ like.cov.T, np.diag([1, 2]))
    assert like.true_size == 2


def test_cc_reorders_both_covariance_axes(tmp_path):
    data, cov = tmp_path / "cc.txt", tmp_path / "cc.cov"
    data.write_text("# z Hz errHz stat met\n0.3,90,3,1,1\n0.1,70,1,1,1\n0.2,80,2,1,1\n")
    covariance = np.diag([9.0, 1.0, 4.0])
    np.savetxt(cov, covariance)
    like = instance("likelihoods/CC/CC.py", {"CC_path": str(data), "CovMat_path": str(cov)})
    np.testing.assert_allclose(like.z, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(like.covmat, np.diag([1, 4, 9]))


@pytest.mark.parametrize("name", ["ET", "LISA"])
def test_gw_forecast_has_zero_residual_at_data_mean(name):
    like = instance(f"likelihoods/Forecast/GWs/{name}_Like/{name}_Like.py")
    assert np.all(np.diff(like.z) >= 0)
    like.provider = SimpleNamespace(get_angular_diameter_distance=lambda z: like.data / (1 + z)**2)
    assert like.logp() == pytest.approx(0, abs=1e-20)


@pytest.mark.parametrize("name", ["DESI", "EUCLID"])
def test_bao_forecast_has_zero_residual_at_data_mean(name):
    like = instance(f"likelihoods/Forecast/BAO/{name}_Like/{name}_Like.py")
    like.provider = SimpleNamespace(get_param=lambda name: 147,
        get_angular_diameter_distance=lambda z: 147 / (np.deg2rad(like.data) * (1 + z)))
    assert like.logp() == pytest.approx(0, abs=1e-20)


def test_lrg_dr1_effective_redshift_and_zero_residual():
    like = instance("likelihoods/BAO/desi_BAO_DR1/desi_lrg.py")
    assert like.z_eff == [0.510, 0.706]
    def da(z):
        dm = 13.62 if z == 0.510 else 16.85
        return dm * 147 / (1 + z)
    def hubble(z, units=None):
        dh = 20.98 if z == 0.510 else 20.08
        return 299800 / (dh * 147)
    like.provider = SimpleNamespace(get_param=lambda name: 147,
        get_angular_diameter_distance=da, get_Hubble=hubble)
    assert like.logp() == pytest.approx(0, abs=1e-20)


@pytest.mark.parametrize("basis", [["omega_b", "omega_m", "theta_s"], ["omega_b", "theta_s"],
                                  ["omega_b", "omega_m", "1/theta_s"], ["omega_b", "1/theta_drag"]])
def test_compressed_cmb_gaussian_reference(basis):
    like = instance("likelihoods/CMB_compressed/compressed_CMB.py", {"compression_basis": basis})
    like.provider = Provider()
    theory = like._theory_vector()
    residual = theory - like.data
    expected = -0.5 * residual @ np.linalg.solve(like.cov, residual)
    assert like.logp() == pytest.approx(expected)


def test_cpl_endpoints_and_boundary():
    ph = load_module("likelihoods/Theory/Exclude_Phanthom/Exclude_Phanthom.py").Phantom
    qu = load_module("likelihoods/Theory/Exclude_Quintessential/Exclude_Quintessential.py").Phantom
    assert ph.loglike(-1, 0) == 0
    assert qu.loglike(-1, 0) == 0
    assert ph.loglike(-0.9, -0.2) == -np.inf
    assert qu.loglike(-1.1, 0.2) == -np.inf
