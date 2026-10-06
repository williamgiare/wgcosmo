## DDR-Plots

This folder contains Jupyter notebooks that reproduce some of the plots from [arXiv:2504.10464](https://arxiv.org/abs/2504.10464).

The modified version of the CLASS code required to run these notebooks can be found in:

* [`class_ddr`](https://github.com/elsateixeira/class_ddr)

For completeness, the same repository is also available at:

* [`class_ddr/DDR_FIGURES`](https://github.com/elsateixeira/class_ddr/tree/master/DDR_FIGURES)

The notebooks and associated codes were developed in collaboration with [Elsa M. Teixeira](https://github.com/elsateixeira).

The full Pantheon+SH0ES covariance is shared with the likelihoods in
[`likelihoods/SN/data`](../../likelihoods/SN/data/). `Data.SN_covariance()` reads
that matrix. The existing figures continue to use the binned SN data and errors
from `data/binned_PanP.dat`.
