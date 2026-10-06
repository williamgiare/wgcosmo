import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.stats import norm

from conftest import load_module

sus = load_module("statistics/SuStat/SuStat.py")
be = load_module("statistics/MCMC_Evidence/Cobaya_wrapper.py")


def write_chain(root, array, names, number=1):
    np.savetxt(str(root) + f".{number}.txt", array, header=" ".join(names))


def test_weighted_moments_and_all_chain_ids(tmp_path):
    root = tmp_path / "sample"
    write_chain(root, [[100, 0, 0], [1, 0, 10]], ["weight", "minuslogpost", "x"], 2)
    write_chain(root, [[1, 0, 20]], ["weight", "minuslogpost", "x"], 12)
    chain = sus.get_chain(root, ["x"], fburn=0, include_weights=True)
    mean, cov = sus._weighted_moments(chain, ["x"])
    expanded = np.repeat([0, 10, 20], [100, 1, 1])
    assert mean[0] == pytest.approx(expanded.mean())
    assert cov[0, 0] == pytest.approx(expanded.var())
    assert len(chain) == 3
    assert list(sus.get_chain(root, ["x"], 0)) == ["x"]
    chain["weight"] *= 0.1
    scaled = sus._weighted_moments(chain, ["x"])
    np.testing.assert_allclose(scaled[0], mean)
    np.testing.assert_allclose(scaled[1], cov)


@pytest.mark.parametrize("shift", [0, 3, 40])
def test_suspiciousness_matches_analytic_gaussians(tmp_path, shift):
    a, b = tmp_path / "a", tmp_path / "b"
    write_chain(a, [[1, 0, -1], [1, 0, 1]], ["weight", "minuslogpost", "x"])
    write_chain(b, [[1, 0, shift - 1], [1, 0, shift + 1]], ["weight", "minuslogpost", "x"])
    statistic, logS, sigma = sus.get_sus(a, b, ["x"], fburn=0, verbose=False, get_results=True)
    assert statistic == pytest.approx(shift**2 / 2)
    assert logS == pytest.approx((1 - shift**2 / 2) / 2)
    assert sigma == pytest.approx(abs(shift) / np.sqrt(2))


def test_chain_validation(tmp_path):
    root = tmp_path / "sample"
    with pytest.raises(FileNotFoundError):
        sus.get_chain(root, ["x"])
    write_chain(root, [[-1, 0, 0]], ["weight", "minuslogpost", "x"])
    with pytest.raises(ValueError, match="Negative"):
        sus.get_chain(root, ["x"], 0)
    with pytest.raises(ValueError, match="fburn"):
        sus.get_chain(root, ["x"], 1)
    write_chain(root, [[1, 0, 0], [1, 0, 0]], ["weight", "minuslogpost", "x"])
    with pytest.raises(ValueError, match="positive definite"):
        sus.get_sus(root, root, ["x"], fburn=0, verbose=False)


def posterior_fixture(tmp_path, kind="uniform", dimension=2):
    root = tmp_path / "posterior"
    rng = np.random.default_rng(731)
    samples = rng.normal(size=(18000, dimension))
    names = ["new_parameter"] + [f"nuisance_{i}" for i in range(dimension - 1)]
    if kind == "uniform":
        prior = {"min": -10, "max": 10}
        neglogprior = dimension * np.log(20)
        target = dimension / 2 * np.log(2 * np.pi) - neglogprior
        logdensity = -0.5 * np.sum(samples**2, axis=1) - neglogprior
    else:
        prior = {"dist": "norm", "loc": 0, "scale": 1}
        logdensity = np.sum(norm.logpdf(samples), axis=1)
        target = 0.0  # likelihood=1, normalized Gaussian prior
    info = {"params": {name: {"prior": prior} for name in names}, "sampler": {"mcmc": {"temperature": 1}}}
    with open(str(root) + ".updated.yaml", "w") as handle:
        yaml.safe_dump(info, handle, sort_keys=False)
    for number, part in zip([2, 12], np.array_split(np.column_stack([np.ones(len(samples)), -logdensity, samples]), 2)):
        write_chain(root, part, ["weight", "minuslogpost"] + names, number)
    return root, names, target


@pytest.mark.parametrize("kind,dimension", [("uniform", 2), ("gaussian", 2), ("gaussian", 1)])
def test_evidence_matches_analytic_integral(tmp_path, kind, dimension):
    root, names, target = posterior_fixture(tmp_path, kind, dimension)
    result = be.MCMC_Evidence(root, burnlen=0, verbose=False, get_results=True)
    assert result == pytest.approx(target, abs=0.10)
    # Name-based reordering must leave the normalization invariant.
    if dimension > 1:
        reordered = be.MCMC_Evidence(root, list(reversed(names)), burnlen=0, verbose=False, get_results=True)
        assert reordered == pytest.approx(result, abs=1e-10)
    assert not list(tmp_path.glob("*_BE.*"))


def test_evidence_rejects_partial_or_modified_priors(tmp_path):
    root, names, _ = posterior_fixture(tmp_path)
    with pytest.raises(ValueError, match="every sampled parameter"):
        be.MCMC_Evidence(root, names[:1], verbose=False)
    with pytest.raises(ValueError, match="Prior overrides"):
        be.MCMC_Evidence(root, [names[0] + ";0.95", names[1]], verbose=False)


def test_external_prior_normalization(tmp_path):
    root, names, _ = posterior_fixture(tmp_path)
    p = str(root) + ".updated.yaml"
    with open(p) as handle:
        info = yaml.safe_load(handle)
    info["prior"] = {"extra": "lambda new_parameter: 0"}
    with open(p, "w") as handle:
        yaml.safe_dump(info, handle)
    with pytest.raises(ValueError, match="External priors"):
        be._posterior_arrays(root)
    base, _ = be._posterior_arrays(root, external_prior_log_normalization=0)
    shifted, _ = be._posterior_arrays(root, external_prior_log_normalization=2)
    np.testing.assert_allclose(shifted[0][:, 1], base[0][:, 1] + 2)


def test_evidence_burnin_is_per_chain(tmp_path):
    root, _, _ = posterior_fixture(tmp_path)
    arrays, _ = be._posterior_arrays(root, burnlen=0.5)
    assert [len(a) for a in arrays] == [4500, 4500]
    arrays, _ = be._posterior_arrays(root, burnlen=2)
    assert [len(a) for a in arrays] == [8998, 8998]


def test_legacy_thinning_and_single_parameter_ranges(tmp_path):
    engine = load_module("statistics/MCMC_Evidence/MCEvidence.py")
    data = np.array([[2, 0, -1], [2, 0, 1], [2, 0, 2]])
    sample = engine.MCSamples([data], names=["x"], labels=["x"])
    indices, weights = sample.thin_indices(2, weights=data[:, 0])
    assert len(indices) > 0
    path = tmp_path / "one"
    (tmp_path / "one.ranges").write_text("new_parameter -1 1\n")
    assert engine.params_info(str(path))["volume"] == pytest.approx(2)
    with pytest.raises(ValueError, match="Unrecognized"):
        engine.params_info(str(path), cosmo=True)


def test_duplicate_samples_preserve_evidence(tmp_path):
    root, _, _ = posterior_fixture(tmp_path, "gaussian", 1)
    baseline = be.MCMC_Evidence(root, burnlen=0, verbose=False, get_results=True)
    # Duplicate every row and halve the multiplicity to preserve the posterior.
    for path in be._chain_files(root):
        with path.open() as handle:
            names = handle.readline().lstrip("#").split()
        data = np.loadtxt(path)
        repeated = np.repeat(data, 2, axis=0)
        repeated[:, 0] *= 0.5
        np.savetxt(path, repeated, header=" ".join(names))
    result = be.MCMC_Evidence(root, burnlen=0, verbose=False, get_results=True)
    assert result == pytest.approx(baseline, abs=1e-8)


def test_thinning_retains_correct_multiplicities():
    engine = load_module("statistics/MCMC_Evidence/MCEvidence.py")
    data = np.array([[5, 0, -1], [1, 0, 0], [4, 0, 1]])
    sample = engine.MCSamples([data])
    indices, weights = sample.thin_indices(2, weights=data[:, 0])
    expanded = np.repeat(np.arange(3), data[:, 0].astype(int))[1::2]
    expected_indices, expected_weights = np.unique(expanded, return_counts=True)
    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_array_equal(weights, expected_weights)
    np.testing.assert_array_equal(sample.thin(1, chain=data), data)


def test_tables_with_real_getdist(capsys):
    from getdist import MCSamples
    tables = load_module("utils/Tables/Tables.py")
    rng = np.random.default_rng(92)
    x = rng.normal(size=3000)
    chi = x**2
    samples = MCSamples(samples=np.column_stack([x, chi, chi, chi]),
                        names=["x", "chi2", "chi2__experiment", "chi2__BAO"],
                        labels=["x", "chi^2", "chi^2_{experiment}", "chi^2_{BAO}"],
                        weights=np.ones(len(x)), loglikes=0.5 * chi)
    statistics = tables.get_chi2_statistics(samples, silent=True)
    assert statistics["chi2_min"] == pytest.approx(chi.min())
    assert statistics["sum_single"] == pytest.approx(chi.min())
    tables.get_table([samples], ["x:both"], col_labels=["Gaussian"], chi2=True)
    text = capsys.readouterr().out
    assert "\\begin{table*}" in text
    assert "\\end{table*}" in text
    assert "chi2_experiment" in text


def test_thinning_is_invariant_to_weight_units():
    arrays = [np.array([[5, 1, -1], [1, 2, 0], [4, 3, 1]], dtype=float)]
    base = be._thin_arrays(arrays)
    scaled = [arrays[0].copy()]
    scaled[0][:, 0] *= 0.1
    other = be._thin_arrays(scaled)
    np.testing.assert_allclose(base[0], other[0])


def test_evidence_with_informative_gaussian_prior_and_shifted_likelihood(tmp_path):
    from scipy.stats import multivariate_normal
    root = tmp_path / "informative"
    prior_cov = np.diag([1.0, 4.0])
    likelihood_cov = np.array([[0.8, 0.3], [0.3, 1.2]])
    likelihood_mean = np.array([0.5, -0.4])
    posterior_cov = np.linalg.inv(np.linalg.inv(prior_cov) + np.linalg.inv(likelihood_cov))
    posterior_mean = posterior_cov @ np.linalg.solve(likelihood_cov, likelihood_mean)
    samples = np.random.default_rng(82).multivariate_normal(posterior_mean, posterior_cov, size=18000)
    logprior = multivariate_normal.logpdf(samples, cov=prior_cov)
    loglike = multivariate_normal.logpdf(samples, mean=likelihood_mean, cov=likelihood_cov)
    write_chain(root, np.column_stack([np.ones(len(samples)), -logprior - loglike,
                                      samples, -logprior, -2 * loglike]),
                ["weight", "minuslogpost", "x", "y", "minuslogprior", "chi2"])
    info = {"params": {"x": {"prior": {"dist": "norm", "loc": 0, "scale": 1}},
                       "y": {"prior": {"dist": "norm", "loc": 0, "scale": 2}}}}
    with open(str(root) + ".updated.yaml", "w") as handle:
        yaml.safe_dump(info, handle)
    expected = multivariate_normal.logpdf(likelihood_mean, cov=prior_cov + likelihood_cov)
    actual = be.MCMC_Evidence(root, burnlen=0, verbose=False, get_results=True)
    assert actual == pytest.approx(expected, abs=0.10)
