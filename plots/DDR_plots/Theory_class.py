import sys, os
import numpy as np
from classy import Class

c = 299792.458

class TheoryCalculator:
    def __init__(self, H0=67.97, ombh2=0.02277699188, omch2=0.1190083805, tau=0.0544, ns=0.9649, As=3.044, omk=0,
                 a0=0, a1=-0.0, eta=0, zstar=0.9, w0=None, wa=None):
        
        # Set parameters based on input, including dark energy parameters if provided
        self.params = {
            'output': 'mPk',
            'H0': H0,
            'omega_b': ombh2,
            'omega_cdm': omch2,
            'tau_reio': tau,
            'n_s': ns,
            'A_s': np.exp(As) * (10 ** -10),
            'alpha0_ddr': a0,
            'alpha1_ddr': a1,
            'zstar_ddr': zstar,
            'eta_function': eta,
            'Omega_k': omk,
        }
        
        # Add dark energy parameters if they are provided (if None, they won't be included)
        if w0 is not None and wa is not None:
            self.params.update({
                'Omega_Lambda': 0,
                'w0_fld': w0,
                'wa_fld': wa
            })
        
        # Initialize the CLASS instance
        self.cosmo = Class()
        self.cosmo.set(self.params)
        self.cosmo.compute()
    
    def get_all(self):
        return self.cosmo

    def get_luminosity_distance(self, z):
        return self.cosmo.angular_distance(z) * (1. + z) * (1. + z)

    def get_distance_moduli(self, z):
        return 5 * np.log10(self.get_luminosity_distance(z)) + 25

    def get_luminosity_distance_ddr(self, z):
        return self.cosmo.angular_distance_ddr(z) * (1. + z) * (1. + z)

    def get_eta_ddr(self, z):
        return self.cosmo.angular_distance_ddr(z)/self.cosmo.angular_distance(z)

    def get_distance_moduli_ddr(self, z):
        return 5 * np.log10(self.get_luminosity_distance_ddr(z)) + 25

    def get_H(self, z):
        return self.cosmo.Hubble(z) * c

    def get_r_drag(self):
        return self.cosmo.rs_drag()

    def __del__(self):
        self.cosmo.struct_cleanup()
        self.cosmo.empty()
