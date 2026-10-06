
from pathlib import Path

import sys, os
import numpy as np
import pandas as pd

c=299792.458

class Data:

    def __init__(self, theory=None, SN_path='data/binned_PanP.dat', BAO_path='data/DESI.dat', SDSS_path='data/SDSS.dat', SN_covmat_path=None):
        self.SN_path = self._data_path(SN_path)
        self.SN_covmat_path = (
            self._data_path(SN_covmat_path) if SN_covmat_path is not None
            else Path(__file__).resolve().parents[2] / 'likelihoods/SN/data/Pantheon+SH0ES_STAT+SYS.cov'
        )
        self.BAO_path = self._data_path(BAO_path)
        self.SDSS_path = self._data_path(SDSS_path)
        if theory is None:
            from Theory_class import TheoryCalculator
            theory = TheoryCalculator()
        self.theory = theory

    @staticmethod
    def _data_path(value):
        path = Path(value)
        return path if path.is_absolute() or path.exists() else Path(__file__).resolve().parent / path

    def SN(self):
        with open(self.SN_path, 'r') as f:
            header = f.readline().strip().replace('# ', '')  # Read the first line, remove the '#'
            self.mu_obs_SN = pd.read_csv(self.SN_path, sep=r"\s+", comment='#', names=['z', 'data', 'error'])

        return self.mu_obs_SN

    def SN_covariance(self):
        """Return the full unbinned covariance, distinct from the binned SN() data."""
        with self.SN_covmat_path.open() as handle:
            length = int(handle.readline())
        return np.loadtxt(self.SN_covmat_path, skiprows=1).reshape(length, length)

    def BAO(self):
        df = pd.read_csv(self.BAO_path, sep=r'\s+', header=None, names=['z', 'data', 'error', 'type'])

        # Separate data by type for DESI BAO
        self.BAO_DH_rd = df[df['type'] == 'DH_rd'].drop(columns='type')
        self.BAO_DM_rd = df[df['type'] == 'DM_rd'].drop(columns='type')
        self.BAO_Dv_rd = df[df['type'] == 'Dv_rd'].drop(columns='type')

        # Calculate BAO values
        r_drag = self.theory.get_r_drag()
        H_values = [self.theory.get_H(z) for z in self.BAO_Dv_rd['z']]

        # DL from DM/rd for DESI
        DL_obs_BAO_from_DM = self.BAO_DM_rd['data'] * r_drag * (1 + self.BAO_DM_rd['z'])
        DL_obs_BAO_from_DM_error = self.BAO_DM_rd['error'] * r_drag * (1 + self.BAO_DM_rd['z'])

        # DL from Dv/rd for DESI
        DL_obs_BAO_from_Dv = (1 + self.BAO_Dv_rd['z']) * r_drag**1.5 * self.BAO_Dv_rd['data']**1.5 * (np.array(H_values) / (c * self.BAO_Dv_rd['z']))**0.5
        DL_obs_BAO_from_Dv_error = self.BAO_Dv_rd['error'] * (1.5 * self.BAO_Dv_rd['data']**0.5) * (1 + self.BAO_Dv_rd['z']) * r_drag**1.5 * (np.array(H_values) / (c * self.BAO_Dv_rd['z']))**0.5

        # Concatenate arrays for DESI
        self.z_BAO = np.concatenate([self.BAO_DM_rd['z'], self.BAO_Dv_rd['z']])
        self.DL_obs_BAO = np.concatenate([DL_obs_BAO_from_DM, DL_obs_BAO_from_Dv])
        self.DL_obs_BAO_error = np.concatenate([DL_obs_BAO_from_DM_error, DL_obs_BAO_from_Dv_error])

        # Observed mu_BAO for DESI
        self.mu_obs_BAO = 5 * np.log10(self.DL_obs_BAO) + 25
        self.mu_obs_BAO_error = 5 * (1 / (np.log(10) * self.DL_obs_BAO)) * self.DL_obs_BAO_error

        # Observed mu_BAO DataFrame for DESI
        self.mu_obs_BAO = pd.DataFrame({
            'z': self.z_BAO,
            'data': self.mu_obs_BAO,
            'error': self.mu_obs_BAO_error
        })

        return self.mu_obs_BAO

    def SDSS(self):
        df = pd.read_csv(self.SDSS_path, sep=r'\s+', header=None, names=['z', 'data', 'error', 'type'])

        # Separate data by type for SDSS
        self.SDSS_DH_rd = df[df['type'] == 'DH_rd'].drop(columns='type')
        #print(self.SDSS_DH_rd)
        self.SDSS_DM_rd = df[df['type'] == 'DM_rd'].drop(columns='type')
        self.SDSS_Dv_rd = df[df['type'] == 'Dv_rd'].drop(columns='type')

        # Calculate SDSS values
        r_drag = self.theory.get_r_drag()
        H_values = [self.theory.get_H(z) for z in self.SDSS_Dv_rd['z']]
        #print(H_values)

        # DL from DM/rd for SDSS
        DL_obs_SDSS_from_DM = self.SDSS_DM_rd['data'] * r_drag * (1 + self.SDSS_DM_rd['z'])
        DL_obs_SDSS_from_DM_error = self.SDSS_DM_rd['error'] * r_drag * (1 + self.SDSS_DM_rd['z'])

        # DL from Dv/rd for SDSS
        DL_obs_SDSS_from_Dv = (1 + self.SDSS_Dv_rd['z']) * r_drag**1.5 * self.SDSS_Dv_rd['data']**1.5 * (np.array(H_values) / (c * self.SDSS_Dv_rd['z']))**0.5
        DL_obs_SDSS_from_Dv_error = self.SDSS_Dv_rd['error'] * (1.5 * self.SDSS_Dv_rd['data']**0.5) * (1 + self.SDSS_Dv_rd['z']) * r_drag**1.5 * (np.array(H_values) / (c * self.SDSS_Dv_rd['z']))**0.5
        #print(DL_obs_SDSS_from_Dv)
        # Concatenate arrays for SDSS
        self.z_SDSS = np.concatenate([self.SDSS_DM_rd['z'], self.SDSS_Dv_rd['z']])
        self.DL_obs_SDSS = np.concatenate([DL_obs_SDSS_from_DM, DL_obs_SDSS_from_Dv])
        self.DL_obs_SDSS_error = np.concatenate([DL_obs_SDSS_from_DM_error, DL_obs_SDSS_from_Dv_error])

        # Observed mu_SDSS for SDSS
        self.mu_obs_SDSS = 5 * np.log10(self.DL_obs_SDSS) + 25
        self.mu_obs_SDSS_error = 5 * (1 / (np.log(10) * self.DL_obs_SDSS)) * self.DL_obs_SDSS_error

        # Observed mu_SDSS DataFrame for SDSS
        self.mu_obs_SDSS = pd.DataFrame({
            'z': self.z_SDSS,
            'data': self.mu_obs_SDSS,
            'error': self.mu_obs_SDSS_error
        })

        return self.mu_obs_SDSS
