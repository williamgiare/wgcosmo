"""Pantheon+ supernova Gaussian likelihood.

Data and both covariance axes are selected with exactly the same row mask.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cholesky, solve_triangular

try:
    from cobaya.likelihood import Likelihood
except ImportError:
    class Likelihood:
        pass


class Pantheon_Plus(Likelihood):
    name = "Pantheon_Plus"
    z_min = 0.01
    path_covmat = None
    path_lc = None

    def initialize(self):
        here = Path(__file__).resolve().parent
        self.path_covmat = self.path_covmat or str(here.parent / "data" / "Pantheon+SH0ES_STAT+SYS.cov")
        self.path_lc = self.path_lc or str(here / "data" / "Pantheon+SH0ES.dat")
        with open(self.path_lc) as handle:
            names = handle.readline().lstrip("#").split()
        self.light_curve_params = pd.read_csv(self.path_lc, sep=r"\s+", skiprows=1, names=names)
        table = self.light_curve_params
        required = {"zHD", "zHEL", "m_b_corr"}
        if not required.issubset(table.columns):
            raise ValueError(f"Missing supernova columns: {sorted(required - set(table.columns))}")
        with open(self.path_covmat) as handle:
            length = int(handle.readline())
        if length != len(table):
            raise ValueError("Supernova covariance dimensions do not match the data")
        self.C00 = np.loadtxt(self.path_covmat, skiprows=1).reshape(length, length)
        self.z = table.zHD.to_numpy(dtype=float)
        self._calibrator = np.zeros(length, dtype=bool)
        self._mask = (self.z > self.z_min) | self._calibrator
        self.true_size = int(self._mask.sum())
        if not self.true_size:
            raise ValueError("No supernovae pass the selection")
        covariance = self.C00[np.ix_(self._mask, self._mask)]
        if not np.allclose(covariance, covariance.T, rtol=0, atol=5e-8):
            raise ValueError("Supernova covariance is not symmetric")
        # The released SH0ES matrix has rounding differences up to 3e-8.
        # Preserve the lower-triangle convention of the original Cholesky.
        self.selected_covariance = np.tril(covariance) + np.tril(covariance, -1).T
        self.cov = cholesky(self.selected_covariance, lower=True)
        self._selected = table.loc[self._mask]
        self._selected_calibrator = self._calibrator[self._mask]
        self._cosmological = ~self._selected_calibrator
        self._z_cmb = self._selected.zHD.to_numpy(dtype=float)[self._cosmological]
        self._z_hel = self._selected.zHEL.to_numpy(dtype=float)[self._cosmological]
        self._observed = self._selected.m_b_corr.to_numpy(dtype=float)

    def get_requirements(self):
        return {"angular_diameter_distance": {"z": self._z_cmb}, "M": None}

    def logp(self, **params_values):
        moduli = np.empty(self.true_size)
        if self._cosmological.any():
            da = np.asarray(self.provider.get_angular_diameter_distance(self._z_cmb))
            dl = (1 + self._z_cmb) * (1 + self._z_hel) * da
            if not np.isfinite(dl).all() or np.any(dl <= 0):
                return -np.inf
            moduli[self._cosmological] = 5 * np.log10(dl) + 25
        
        residual = self._observed - self.provider.get_param("M") - moduli
        whitened = solve_triangular(self.cov, residual, lower=True)
        return -0.5 * float(whitened @ whitened)
