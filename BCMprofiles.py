__all__ = ("StellarProfile", "EjectedGasProfile", "BoundGasProfile", "CombinedGasProfile", "CombinedStellarGasProfile" , "CombinedAllBCMProfile")

import numpy as np
import pyccl as ccl
import matplotlib.pyplot as plt
from scipy import signal
import scipy.integrate as integrate
import scipy.interpolate as interpol
from pyccl._core import UnlockInstance

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

class EjectedGasProfile(ccl.halos.profiles.profile_base.HaloProfile): 
    """Creating a class for the ejected gas density profile
    where: """  # could put in the equations used

    def __init__(self, cosmo, mass_def): 
        super(EjectedGasProfile, self).__init__(mass_def=mass_def)
        self.cosmo = cosmo

    def _real(self, r, M, delta=200, eta_b = 0.5, scale_a=1): 
        r_use = np.atleast_1d(r) 
        M_use = np.atleast_1d(M)
        r_vir = self.mass_def.get_radius(self.cosmo, M_use, scale_a) / scale_a # halo virial radius
        r_e = 0.375*r_vir*np.sqrt(delta)*eta_b # eta_b = a free parameter
        
        prefix = M_use * (1/(scale_a*np.sqrt(2*np.pi*r_e)))**3
        x = r_use[None, :] / r_e[:, None]
        prof = prefix[:, None] * np.exp(-(x**2)/2)

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
                                                          
        return prof

    def _fourier(self, k, M, delta=200, eta_b = 0.5, scale_a=1):
        k_use = np.atleast_1d(k)
        M_use = np.atleast_1d(M)
        r_vir = self.mass_def.get_radius(self.cosmo, M_use, scale_a) / scale_a # halo virial radius
        r_e = 0.375*r_vir*np.sqrt(delta)*eta_b # eta_b = a free parameter

        prefix = M_use / scale_a**3
        x = k_use[None, :] * r_e[:, None]
        prof = prefix[:, None] * np.exp(-(x**2)/2)  

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)

        return prof

class BoundGasProfile(ccl.halos.profiles.profile_base.HaloProfile): 
    """Creating a class for the bound gas density profile
    where: """  # could put in the equations used

    def __init__(self, cosmo, mass_def, concentration, gamma, GammaRange = (1.01, 10), nGamma=64, qrange=(1e-4, 1e2), nq=64): 
        self.gamma = gamma
        super(BoundGasProfile, self).__init__(mass_def=mass_def, concentration=concentration)
        self.cosmo = cosmo

        self.GammaRange = GammaRange
        self.nGamma = nGamma

        self.qrange = qrange
        self.nq = nq
        
        self._func_normQ0 = None   # General normalised profile (for q=0, over Gamma)
        self._func_normQany = None

    def _shape(self, x, gam):
        gam_use = np.atleast_1d(gam)
        return (np.log(1+x)/x)**gam_use

    def _innerInt(self, x, gam): 
        return x**2 * self._shape(x, gam)   

    def _Vb_prefix(self, r_s=1):
        vB1 = integrate.quad(self._innerInt, 0, np.inf, args = (1/(self.gamma-1)))  
        vB2 = 4*np.pi*(r_s**3)*vB1[0]
        return vB2

    def _norm_interpol1(self):  # interpol1 = for q = 0
        gamma_list = np.linspace(self.GammaRange[0], self.GammaRange[1], self.nGamma) 
        I0_array = np.zeros(self.nGamma)
        k=0
        for i in gamma_list:
            I0_array[k] =  integrate.quad(self._innerInt, 0, np.inf, args = 1/(i-1))[0] 
            k+=1
        func_normQ0 = interpol.interp1d(gamma_list, I0_array) 
        return func_normQ0
        
    def _real(self, r, M, call_interp=True, scale_a=1): 
        r_use = np.atleast_1d(r) 
        M_use = np.atleast_1d(M)
        
        R_M = self.mass_def.get_radius(self.cosmo, M_use, scale_a) / scale_a # halo virial radius
        c_M = self.concentration(self.cosmo, M_use, scale_a) # concentration-mass relation c(M)
        r_s = R_M / c_M # characteristic scale r_s

        if call_interp==False:
            vB_prefix = self._Vb_prefix(r_s)
        else:
            if self._func_normQ0 is None: # is instead of == here
                with UnlockInstance(self):
                    self._func_normQ0 = self._norm_interpol1() 
            vB_prefix = 4*np.pi*(r_s**3)*self._func_normQ0(self.gamma)  
        prefix = M_use * (1/scale_a**3) * (1/vB_prefix)

        x = r_use[None, :] / r_s[:, None]
        prof = prefix[:, None] * self._shape(x, 1/(self.gamma-1)) 

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
                                                          
        return prof

    def _norm_interpol2(self):  # interpol1 for q = any
        gamma_list = np.linspace(self.GammaRange[0], self.GammaRange[1], self.nGamma) 
        q_array = np.geomspace(self.qrange[0], self.qrange[1], self.nq) #
        I0_array =  np.zeros((self.nGamma, self.nq))

        def integralQany(x, gam): 
            return x * self._shape(x, gam) 
        k=0
        for i in gamma_list: 
            l=0
            for j in q_array: 
                I0_array[k, l] =  integrate.quad(integralQany, 0, np.inf, args = 1/(i-1), weight = "sin", wvar=j)[0] / j
                l+=1
            k+=1
            print(f'k = {100*k/self.nGamma:.3g}% through')
        func_normQany = interpol.RegularGridInterpolator((gamma_list, np.log(q_array)), I0_array)
        return func_normQany
    
    def _fourier(self, k, M, scale_a=1):
        k_use = np.atleast_1d(k)
        M_use = np.atleast_1d(M)

        R_M = self.mass_def.get_radius(self.cosmo, M_use, scale_a) / scale_a # halo virial radius
        c_M = self.concentration(cosmo, M_use, scale_a) # concentration-mass relation c(M)
        r_s = R_M / c_M # characteristic scale r_s

        if self._func_normQ0 is None: # is instead of == here
            with UnlockInstance(self):
                self._func_normQ0 = self._norm_interpol1() 
        if self._func_normQany is None:
            with UnlockInstance(self):
                self._func_normQany = self._norm_interpol2()

        q_use = k_use[None, :]*r_s[:, None]
        g_k = self._func_normQany((self.gamma, np.log(q_use))) / self._func_normQ0(self.gamma) # = Ib_qAny / Ib_q0

        prefix = M_use / scale_a**3
        prof = prefix[:, None] * g_k[None,:] 

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)

        return prof
    

class CombinedGasProfile(ccl.halos.profiles.profile_base.HaloProfile): 
    """ Combined profile of ejected & bound gas, assuming $f_{bd} + f_{ej} = 1$
    """

    def __init__(self, cosmo, mass_def, concentration, gamma, GammaRange = (1.01, 10), nGamma=64, qrange=(1e-4, 1e2), nq=64):
        self.gamma = gamma
        super(CombinedGasProfile, self).__init__(mass_def=mass_def, concentration=concentration)
        self.boundProfile = BoundGasProfile(cosmo=cosmo, mass_def=mass_def, concentration=concentration, gamma=gamma)
        self.ejProfile = EjectedGasProfile(cosmo=cosmo, mass_def=mass_def)

        self.GammaRange = GammaRange
        self.nGamma = nGamma
        self.qrange = qrange
        self.nq = nq
        self._func_normQ0 = None   # General normalised bound profile (for q=0, over Gamma)
        self._func_normQany = None

    def _real(self, r, M, f_bd=1, call_interp=True, scale_a=1):
        f_ej = 1 - f_bd
        prof_ej = self.ejProfile._real(r, M, scale_a) 
        prof_bd = self.boundProfile._real(r, M, call_interp, scale_a) 
        profile = f_ej*prof_ej + f_bd*prof_bd
        return profile

    def _fourier(self, k, M, f_bd = 1, call_interp=True, scale_a=1):
        f_ej = 1 - f_bd
        prof_ej = self.ejProfile._fourier(k, M, scale_a)
        prof_bd = self.boundProfile._fourier(k, M, scale_a)
        profile = f_ej*prof_ej + f_bd*prof_bd[0]
        return profile