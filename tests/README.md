# Numerical and integration checks

From the repository root, install the dependencies in a dedicated environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest tests -q
```

The suite checks weighted posterior moments and analytic Gaussian tension,
Bayesian evidence against known Gaussian integrals, parameter reordering,
per-chain burn-in, duplicate samples and thinning multiplicities. It also
checks every distributed likelihood with the included data, covariance row
selection, the DESI DR1 LRG redshift, and LaTeX table generation with GetDist.
Notebook checks cover syntax and selected helpers using synthetic inputs,
without overwriting saved figures or outputs.

Integration tests evaluate all distance likelihoods through a real Cobaya/CAMB
model, the three shipped compressed-CMB bases through Cobaya/CLASS, and the
evidence of a short, reproducible Cobaya MCMC run. All generated chains are
written to pytest's temporary directory. If CAMB or CLASS is missing, the
corresponding integration tests are skipped; install `requirements.txt`
to run them all.

These checks do not execute the historical publication notebooks, download
external datasets, reproduce full Planck/ACT/SPT fits, or validate modified
CLASS forks. Those require the inputs and software indicated in their READMEs.
