# Bayesian evidence from Cobaya chains

The revised wrapper estimates **log Z**, using every sampled parameter,
including nuisance parameters, from the `.updated.yaml` file. The posterior
density already contains Cobaya's normalized one-dimensional priors, so it is
integrated directly with no additional box-volume correction. Gaussian priors
are included as densities, not approximated by a finite confidence interval.

```python
from Cobaya_wrapper import MCMC_Evidence
logZ = MCMC_Evidence("chains/model", params=None, burnlen=0.3, get_results=True)
```

Every numbered chain is read, including IDs above 9 or missing intermediate
IDs. Burn-in is applied separately to each chain. No intermediate `_BE` files
are written by this entry point. For chains with varying multiplicities, automatic
systematic thinning uses the largest weight to retain unit-weight samples.
`thin_factor=0` preserves the original weights; a larger explicit factor can
reduce autocorrelation further. Thinning does not establish convergence.

The returned quantity is log evidence; a log
Bayes factor is `logZ_A - logZ_B` for two models with the same data and consistent
likelihood normalization.

An explicit `params` list must include every sampled parameter; its order is
respected. Old `parameter:min/max` or `parameter;CL` overrides are rejected:
changing a prior requires resampling or correct importance reweighting, not
just changing the final normalization. Metadata is required to distinguish
sampled and derived parameters. Tempered chains are rejected.

External priors are not automatically normalized by Cobaya. When present,
supply `external_prior_log_normalization`, the logarithm of the integral of the
full joint prior before normalization. Zero is appropriate only if that joint
prior is already normalized.

See [the test documentation](../../tests/README.md) for analytic and real-Cobaya validation, including uniform priors and informative Gaussian priors with a shifted, correlated Gaussian likelihood. This is a
nearest-neighbour estimator; correlated samples, finite sample size and high
dimension affect its accuracy. Converged chains remain necessary, and a nested
sampler is preferable when accurate evidence is the primary goal.

## Historical validation

The comparisons below belong to the previous cosmological-only wrapper. They
are retained for provenance and have **not** been rerun with the revised,
full-parameter estimator. New results may differ.

#### Validation and Consistency Checks

The results obtained with this wrapper have been compared to those from **CosmoMC**, showing good agreement across several model extensions. The comparison was performed using the **Planck TTTEEE + lowL + lowE + lensing** dataset. Specifically:

---

##### ΛCDM + $m_\nu$
-  **Cobaya**:  $\log Z_{\Lambda\text{CDM}+m_\nu} - \log Z_{\Lambda\text{CDM}} = -3.66$
-  **CosmoMC**:  $\log Z_{\Lambda\text{CDM}+m_\nu} - \log Z_{\Lambda\text{CDM}} = -3.64$
    
---

##### ΛCDM + $\Omega_k$
- **Cobaya**:  $\log Z_{\Lambda\text{CDM}+\Omega_k} - \log Z_{\Lambda\text{CDM}} = -2.42$
- **CosmoMC**:  $\log Z_{\Lambda\text{CDM}+\Omega_k} - \log Z_{\Lambda\text{CDM}} = -2.40$

---

##### ΛCDM + $A_\mathrm{lens}$ [^1]
- **Cobaya**:  $\log Z_{\Lambda\text{CDM}+A_\mathrm{lens}} - \log Z_{\Lambda\text{CDM}} = -2.99$
- **CosmoMC**:  $\log Z_{\Lambda\text{CDM}+A_\mathrm{lens}} - \log Z_{\Lambda\text{CDM}} = -3.33$

[^1] (*Note* A different prior on $A_\mathrm{lens}$ was used, which likely explains the observed discrepancy)

---

## Accuracy

The historical comparisons above do not establish an uncertainty bound for the
revised estimator. The regression tests compare estimates against known
Gaussian integrals and include an actual Cobaya MCMC chain. These checks verify
normalization and implementation; they do not guarantee a particular accuracy
for arbitrary cosmological chains. Convergence, autocorrelation, sample size,
and dimension still matter. For precise evidence calculations, use a nested
sampler and compare results with consistent prior and likelihood normalization.
