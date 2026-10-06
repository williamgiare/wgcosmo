"""Cosmic chronometer likelihood, with covariance aligned to the data rows."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

try:
    from cobaya.likelihood import Likelihood
except ImportError:
    class Likelihood:
        pass


class CC(Likelihood):
    name = "CC"
    CC_path = None
    CovMat_path = None

    def initialize(self):
        here = Path(__file__).resolve().parent / "data"
        self.CC_path = self.CC_path or str(here / "CC.txt")
        self.CovMat_path = self.CovMat_path or str(here / "CovMat.txt")
        table = pd.read_csv(self.CC_path, header=None, skiprows=1, names=["z", "Hz", "errHz", "stat_contr", "met_contr"]).to_numpy(dtype=float)
        covariance = np.loadtxt(self.CovMat_path)
        self.num_CC = len(table)
        if covariance.shape != (self.num_CC, self.num_CC):
            raise ValueError("CC covariance dimensions do not match the data")
        order = np.argsort(table[:, 0], kind="stable")
        self.z, self.data, self.error = table[order, :3].T
        self.covmat = covariance[np.ix_(order, order)]
        self._chol = cho_factor(self.covmat, lower=True)

    def get_requirements(self):
        return {"Hubble": {"z": self.z}}

    def logp(self, **params_values):
        theory = np.asarray(self.provider.get_Hubble(self.z, units="km/s/Mpc"))
        residual = self.data - theory
        return -0.5 * float(residual @ cho_solve(self._chol, residual))
