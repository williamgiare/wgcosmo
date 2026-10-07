# PostModes

**Eigenmode analysis of posterior geometry.**

[PostModes](https://github.com/williamgiare/postmodes) is a separate Python package
for comparing posterior constraints through the geometry of their covariance
matrices. This folder points to that project; its source code is maintained in
the linked repository.

Given a reference posterior `A` and an alternative posterior `B` on the same
physical parameter space, `postmodes` provides:

- PCA of each posterior;
- mean-shift diagnostics when posterior means are available;
- rotation diagnostics between the PCA bases;
- generalized eigenmode analysis of
  $C = C_A^{-1/2} C_B C_A^{-1/2}$.

The outputs include mode-by-mode variance ratios `rho`, standard-deviation
ratios `sqrt(rho)`, generalized parameter combinations in the original basis,
isotropic and anisotropic deformation metrics, and changes in correlation
structure. Inputs can be covariance matrices, MCMC chains, or both.

For source code, installation instructions, examples, and documentation, see
the [main PostModes repository](https://github.com/williamgiare/postmodes).

## Citation

The main repository requests citation of the accompanying manuscript:

Pedrotti, **Giarè**, Cheng, Di Valentino,
*When, Why, and How CMB Compression Fails* (2026),
[arXiv:2610.08728](https://arxiv.org/abs/2610.08728).

<details>
<summary>BibTeX</summary>

```bibtex
@article{Pedrotti:2026ncd,
    author = "Pedrotti, Davide and Giar\`{e}, William and Cheng, Hanyu and Di Valentino, Eleonora",
    title = "{When, Why, and How CMB Compression Fails}",
    eprint = "2610.08728",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    month = "10",
    year = "2026"
}
```

</details>
