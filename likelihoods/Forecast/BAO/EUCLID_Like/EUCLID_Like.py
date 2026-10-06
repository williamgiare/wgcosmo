"""Diagonal Gaussian forecast likelihood for angular BAO measurements."""
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from cobaya.likelihood import Likelihood
except ImportError:
    class Likelihood:
        pass


class EUCLID_Like(Likelihood):
    name = "EUCLID_Like"
    EUCLID_path = None

    def initialize(self):
        self.EUCLID_path = self.EUCLID_path or str(Path(__file__).resolve().parent / "data/EUCLID.txt")
        table = pd.read_csv(self.EUCLID_path, header=None, skiprows=1, names=["z", "DA", "dDA", "theta", "dtheta"]).to_numpy(dtype=float)
        table = table[np.argsort(table[:, 0], kind="stable")]
        self.z, self.data, self.error = table[:, 0], table[:, 3], table[:, 4]
        self.num_BAO = len(table)
        if not np.isfinite(table).all() or np.any(self.error <= 0):
            raise ValueError("Forecast data must be finite with positive errors")

    def get_requirements(self):
        return {"angular_diameter_distance": {"z": self.z}, "rdrag": None}

    def logp(self, **params_values):
        da = np.asarray(self.provider.get_angular_diameter_distance(self.z))
        rs = self.provider.get_param("rdrag")
        theory = np.rad2deg(rs / (da * (1 + self.z)))
        residual = (self.data - theory) / self.error
        return -0.5 * float(residual @ residual)
