## Bayesian Evidence Estimation

This directory provides a wrapper for computing the **Bayesian Evidence (BE)** from Markov Chain Monte Carlo (MCMC) chains produced with **Cobaya**, using [MCEvidence](https://github.com/yabebalFantaye/MCEvidence).

---
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

## Recommendation

This level of agreement (typically within $\Delta \log Z \lesssim 0.05$) is acceptable for most applications, especially given the intrinsic uncertainty associated with BE estimation. 

However, as shown in [arXiv:2212.11926](https://arxiv.org/pdf/2212.11926.pdf), the expected uncertainty in $\log B_{ij}$ is approximately $\sigma(\log B_{ij}) \sim 0.5$ when comparing results from MCEvidence and PolyChord for a multivariate Gaussian distribution. Therefore if high-precision evidence calculations are required—particularly for models with a large number of parameters beyond ΛCDM—we recommend using a **nested sampler** such as [**PolyChord**](https://github.com/PolyChord/PolyChordLite) directly.
