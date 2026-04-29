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

[^1]: If you notice any typos or issues with the materials shared, please [let me know](mailto:w.giare@sheffield.ac.uk)!
