# SuStat

`get_sus` computes the Gaussian approximation to suspiciousness in a selected
common parameter space. It reads every numbered Cobaya chain, applies burn-in
to each file, and uses the `weight` column for the posterior population mean
and covariance. Multiplicities need not be integers.

```python
from SuStat import get_sus
chi2, logS, sigma = get_sus(
    "chains/dataset_A", "chains/dataset_B", ["H0", "omegam"],
    fburn=0.3, get_results=True,
)
```

The posterior covariance uses population normalization, so rescaling or
splitting weights leaves the results unchanged. Fixed or linearly dependent
parameters must be removed. The chi-square-to-significance conversion assumes
approximately Gaussian posteriors; the routine does not calculate general
suspiciousness from evidences and KL divergences.
