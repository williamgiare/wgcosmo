"""Angular BAO Gaussian likelihood with a fixed covariance matrix."""
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

try:
    from cobaya.likelihood import Likelihood
except ImportError:
    class Likelihood:
        pass


class BAO_2D(Likelihood):
    name = "BA02D_MM"
    covmath_path = None
    bao_data = None

    def initialize(self):
        here = Path(__file__).resolve().parent
        self.covmath_path = self.covmath_path or str(here / "BAO_2D_CovMat.txt")
        self.bao_data = self.bao_data or str(here / "BAO_2D_data.txt")
        table = np.loadtxt(self.bao_data, ndmin=2)
        self.z, self.data, self.error = table[:, 0], table[:, 2], table[:, 5]
        self.num_BAO = len(self.z)
        self.covmat = np.loadtxt(self.covmath_path)
        if self.covmat.shape != (self.num_BAO, self.num_BAO):
            raise ValueError("BAO covariance dimensions do not match the data")
        self._chol = cho_factor(self.covmat, lower=True)

    def get_requirements(self):
        return {"angular_diameter_distance": {"z": self.z}, "rdrag": None}

    def logp(self, **params_values):
        da = np.asarray(self.provider.get_angular_diameter_distance(self.z))
        rs = self.provider.get_param("rdrag")
        residual = self.data - np.rad2deg(rs / (da * (1 + self.z)))
        return -0.5 * float(residual @ cho_solve(self._chol, residual))
