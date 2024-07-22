__all__ = ("StellarProfile")

import numpy as np
import pyccl as ccl
import matplotlib.pyplot as plt
from scipy import signal

class StellarProfile(ccl.halos.profiles.profile_base.HaloProfile): 
    """Creating a class for the stellar density profile
    where: """  # could put in the equations used

    def __init__(self, mass_def):
        super(StellarProfile, self).__init__(mass_def=mass_def)

    def _real(self, r, M, centre_pt=None, # want delta centred at r=0 (& since log scale, can't do negative or zero values in array)
              scale_a=1): 
        r_use = np.atleast_1d(r) 
        M_use = np.atleast_1d(M)
        len_r = len(r_use) 

        prefix = M_use / scale_a**3
        prof = prefix[:, None] * signal.unit_impulse(len_r, centre_pt)[None,:] # If centre_pt=None, defaults to index at the 0th element.

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
                                                          
        return prof

    def _fourier(self, k, M, scale_a=1):
        k_use = np.atleast_1d(k)
        M_use = np.atleast_1d(M)

        prefix = M_use / scale_a**3
        prof = k_use[None,:] + prefix[:, None] * 1 # as g(k) = 1

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)

        return prof
    