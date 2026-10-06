"""Diagonal Gaussian forecast likelihood for gravitational-wave distances."""
from pathlib import Path

import numpy as np

try:
    from cobaya.likelihood import Likelihood
except ImportError:
    class Likelihood:
        pass


class LISA_Like(Likelihood):
    name = "LISA_Like"
    LISA_path = None

    def initialize(self):
        self.LISA_path = self.LISA_path or str(Path(__file__).resolve().parent / "data/LISA.txt")
        table = np.loadtxt(self.LISA_path, ndmin=2)
        table = table[np.argsort(table[:, 0], kind="stable")]
        self.z, self.data, self.error = table.T
        self.num_GW = len(table)
        if not np.isfinite(table).all() or np.any(self.error <= 0):
            raise ValueError("Forecast data must be finite with positive errors")

    def get_requirements(self):
        return {"angular_diameter_distance": {"z": self.z}}

    def logp(self, **params_values):
        da = np.asarray(self.provider.get_angular_diameter_distance(self.z))
        residual = (self.data - da * (1 + self.z)**2) / self.error
        return -0.5 * float(residual @ residual)
