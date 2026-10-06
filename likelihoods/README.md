# Likelihoods

## Overview

This folder contains `python` likelihoods that can be used with `cobaya` (or as standalone Python modules, if needed). [^1]

- The `BAO/` folder includes various likelihoods for Baryon Acoustic Oscillation measurements. Most of them should be fairly self-explanatory. 
- The `CC/` folder contains likelihoods for Cosmic Chronometer data, including proper handling of the covariance matrix.
- The `CMB_compressed` folder contains a compressed likelihood for Cosmic Microwave Background data, configurable for both 3x3 and 2x2 compressions, together with the files needed to build the compressed CMB data vectors and covariance matrices.
- The `Forecast/` folder includes likelihoods for future experiments, based on mock data generated assuming a $\Lambda$CDM cosmology.
- The `SN/` folder contains likelihoods for Supernova datasets.
- The `Theory/` folder provides examples of more ‘theoretical’ likelihoods—useful for including priors or constraints from theoretical considerations.

Each folder should contain an `example.yaml` file demonstrating how to use the likelihood with `cobaya`.

[^1]: If you notice any typos or issues with the materials shared, please [let me know](mailto:giare@hawaii.edu)!

## Running the examples

The `example.yaml` files are configuration fragments. Run from their own
likelihood directory, where `python_path: .` resolves the Python module, and
combine them with the theory, parameters and sampler needed for the analysis.
Packaged data are located relative to the module, independent of the working
directory. The real-model tests in `../tests/` demonstrate full Cobaya setup.

The fixed BAO/CC covariance matrices are factored once during initialization.
For supernovae, the same row mask selects data and both covariance axes. The
small rounding asymmetry in the distributed SH0ES covariance is handled using
the lower triangle, matching the original likelihood's Cholesky convention.
