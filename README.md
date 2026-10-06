# Cosmology Tools

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Repo Status](https://img.shields.io/badge/repo-public-brightgreen)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23197101.svg)](https://doi.org/10.5281/zenodo.23197101)

---

This repository contains a collection of numerical tools, code snippets, and methodologies for cosmological analyses. It is designed to help researchers and students alike in their work with cosmology, statistical methods, and data analysis. 

---
 
## Overview

This repository is organized as follows.

- **`likelihoods/`**  
  This folder contains a number of likelihoods I’ve implemented—many of them out of personal curiosity or interest. In several cases, public versions of these likelihoods already exist in packages like [`cobaya`](https://github.com/CobayaSampler/cobaya) and [`montepython`](https://github.com/brinckmann/montepython_public). Still, I was interested in exploring them directly to experiment or run some independent tests at various points during my research.

- **`notebooks/`**  
  This folder includes a few Jupyter notebooks demonstrating how to use well-known codes commonly used in cosmology or statistical data analysis. I’ve used these either in teaching contexts or while preparing tutorials and seminars. They might serve as useful references or starting points for others.

- **`notes/`**  
  This folder contains various notes I’ve written or used at different times—some were part of personal reading, others tied to specific projects, lectures, or seminars. They’re not polished for publication but could still be useful for anyone working on related topics.

- **`plots/`**  
  This folder includes scripts for generating and reproducing plots related to cosmological data analysis. These scripts are mainly for visualizing results from different datasets or theoretical models. Some were created for specific projects, while others were made for quick checks or exploratory analysis.

- **`statistics/`**  
  This directory includes scripts related to statistical analysis. Some are tools I’ve used in my own work, others are small experiments or utilities I developed while exploring specific methods or ideas.

- **`talks/`**  
  This folder contains materials used for presentations, talks, lectures, and similar events. It includes slides, notes, and any associated scripts or data used for these talks. 

- ** `tests/`**
  This folder contains numerical regression checks for the likelihoods and statistical tools, integration checks with Cobaya/CAMB/CLASS, and checks of selected plotting helpers.

- **`utils/`**  
  This folder collects general-purpose utility functions and scripts that I’ve accumulated over time. Many are reusable tools that I’ve found handy across different projects.

- **`yamls/`**  
  This folder contains configuration files I’ve been collecting and using for different pipelines or runs—mostly related to [`cobaya`](https://github.com/CobayaSampler/cobaya). They’re here for reference or reuse.

---

## Verification

Dependencies are listed in `requirements.txt`. For numerical regression
checks and real Cobaya/CAMB/CLASS integration tests, see [tests/README.md](tests/README.md).
The tests use the packaged data and temporary synthetic chains; historical
publication notebooks can also require external chains, datasets or modified
theory codes, as documented in their folders.

---

## Credits, Contributions & Usage

This repository, `wgcosmo`, is primarily maintained by me ([William Giarè](https://github.com/williamgiare)) and is shared in the spirit of open science. Most of the scripts, notebooks, and methods in this repository were developed for personal research, teaching, student mentorship, or as a way to explore ideas in a more informal context.

### How to cite

If you use material from this repository in your research or presentations, please cite:

<details>
<summary>BibTeX</summary>

```bibtex
@misc{Giare2026wgcosmo,
  author       = {Giar\`{e}, William},
  title        = {wgcosmo},
  year         = {2026},
  howpublished = {Zenodo},
  doi          = {10.5281/zenodo.23197101},
  url          = {https://doi.org/10.5281/zenodo.23197101},
  note         = {Software}
}
```

</details>

### Important:

- Although I usually review and debug code quite carefully, *bugs or mistakes can still be present*: this repository only includes a subset of my material, and not everything here has been used in publications (for which my cross-checking and testing become significantly more rigorous).

- Contributions and suggestions are welcome. Feel free to open an issue or submit a pull request if you'd like to collaborate or improve something. For any feedback, feel free to [contact me](mailto:giare@hawaii.edu).

---
