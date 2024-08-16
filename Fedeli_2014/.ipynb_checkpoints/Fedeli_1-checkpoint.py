__all__ = ("Initialiser", "StellarProfile", "GasProfile", "CDMProfile" , "SAMProfile")

import numpy as np
import pyccl as ccl
import matplotlib.pyplot as plt
from scipy import integrate
import scipy.interpolate as interpol
from scipy.special import erf, gamma, expn, hyp2f1, exp1
from pyccl._core import UnlockInstance

class Initialiser(ccl.halos.profiles.profile_base.HaloProfile):
    """ Contains the __init__ , update_parameters, 
## _f_stell & _f_bd 
    methods to be inherited.
    """

    XXX

class StellarProfile(ccl.halos.profiles.profile_base.HaloProfile):
    """ Stellar halo density profile. Fedeli (2014) arXiv:1401.2997
    """
    def __init__(self, cosmo, mass_def, alpha=1, m_0s=5E12/cosmo['h'], sigma_s=1.2, rho_avg_star=7E8*cosmo['h']**2, limInt_mStell=(1E10 , 1E15)):
        super(StellarProfile, self).__init__(mass_def=mass_def)
        self.cosmo = cosmo
        self.alpha = alpha
        self.m_0s = m_0s = 5E12/cosmo['h']
        self.sigma_s = sigma_s
        self.rho_avg_star = rho_avg_star
        self.limInt_mStell = limInt_mStell

    def _f_stell_noA(self, m):
        return np.exp( (-1/2) * ( np.log10(m/self.m_0s) /self.sigma_s )**2 )
    
    def _f_stell_integrand(self, m):
        # integrand = m * f_star(m) * n(m), where n(m,z) is the standard DM-only halo mass function
      #  DM_mass_func = hmf_200m(cosmo,m,a_sf)/(m*np.log(10)) # ? have as a self. ? (can't with scale_a, but-)
        DM_mass_func = hmf_200m(self.cosmo, np.atleast_1d(m), 1) / (np.atleast_1d(m)*np.log(10))
        return m* self._f_stell_noA(m) * DM_mass_func 
    
    # once we have A:   # f_star(m) = A*np.exp( (-1/2) * ( np.log10(m/m_0s) /omega_s )**2 )
    def _f_stell(self, m):
        # integrating over m (dm)
        integrad = integrate.quad(self._f_stell_integrand, self.limInt_mStell[0], self.limInt_mStell[1])
        A = self.rho_avg_star / integrad[0] 
        return A * self._f_stell_noA(m)
    
    def _real(self, cosmo, r, M, scale_a=1, r_t = 1, xDel_ratio = 1/0.03):
        """ X
        """
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        # r_t = 
        r_vir = self.mass_def.get_radius(self.cosmo, M_use, scale_a) / scale_a    # R_delta = the halo virial radius r_vir
        f_stell = self._f_stell(M_use)
        
        if xDel_ratio is None:
            x_delta = r_vir / r_t # use the inputted value of r_t
        else:
            # default: x_delta = 1/0.03, as in Fedeli 2014 paper 
            x_delta = xDel_ratio  # reassign r_c in order to give the specific r_c/r_Del ratio desired 
            r_t = r_vir / x_delta
            
        nu_alpha = 1 - (2 / self.alpha)
        # Using E_1 = int^infty_1 e^{-xt} * t dt = (e^{-x}*(x+1))/(x^2), assuming x = alpha here
        rho_t_bracket = gamma(1 - nu_alpha) - (x_delta**2)*(x_delta**self.alpha)*(np.exp(-nu_alpha)*(nu_alpha+1))/(nu_alpha**2)
        rho_t = M_use*f_stell*self.alpha / (4*np.pi*(r_t**3) * rho_t_bracket)

        x = r_use[None, :] / r_t[:, None]
        prefix = rho_t * f_stell 
        prof = prefix[:, None] * np.exp(-x**self.alpha)/x 

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

class GasProfile(ccl.halos.profiles.profile_base.HaloProfile):
    """ Gas halo density profile. Fedeli (2014) arXiv:1401.2997
    """
    def __init__(self, cosmo, mass_def, fourier_numerical=True, beta=2/3, limInt=(0,1), nk=64, krange=(5E-3, 5E2), # (1E-2,9E1)
                 m_0g = 5E12/cosmo['h'], sigma_g = 1.2,
                 truncate_param=1):
        super(GasProfile, self).__init__(mass_def=mass_def)
        self.fourier_numerical = fourier_numerical
        if fourier_numerical is True:
            self._fourier = self._fourier_numerical

        self.cosmo = cosmo
        self.beta=beta
        self.truncate_param = truncate_param # if truncate=True in real, truncate at r > (r_vir * truncate_param)
        self.m_0g = m_0g
        self.sigma_g = sigma_g

        self.limInt = limInt
        self.krange = krange
        self.nk = nk
        self._func_fourier = None   # [Normalised] profile from the Fourier interpolator (for Fedeli's Fourier integral)

    def _f_gas(self, m):
        m_use = np.atleast_1d(m)
        f_array = np.zeros(np.shape(m_use))
        for i, mass in enumerate(m_use):
            if (mass < self.m_0g):
                f_array[i] = 0
            else:
                f_array[i] = (self.cosmo['Omega_b']/self.cosmo['Omega_m']) * erf(np.log10(mass/self.m_0g) / self.sigma_g)
        return f_array
    
    def _real(self, cosmo, r, M, scale_a=1, r_c = 1, xDel_ratio=1/0.05, truncate=True, # for inbuilt FFT, need truncation to be default
              no_prefix=False): 
        """ X
        """
        
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        # r_c = 
        r_vir = self.mass_def.get_radius(cosmo, M_use, scale_a) / scale_a    # R_delta = the halo virial radius r_vir
        f_gas = self._f_gas(M_use)

        if xDel_ratio is None:
            x_delta = r_vir / r_c # use the inputted value of r_c
        else:
            # default: x_delta = 1/0.05, as in Fedeli 2014 paper 
            x_delta = xDel_ratio  # reassign r_c in order to give the specific r_c/r_Del ratio desired 
            r_c = r_vir / x_delta

        rho_bracket = (x_delta**3) * hyp2f1(3/2, 3*self.beta/2, 5/2, -(x_delta**2))
        rho_c = 3 * M_use * f_gas / (4 * np.pi * (r_c**3) * rho_bracket)
        
        x = r_use[None, :] / r_c[:, None]

        if no_prefix is True:
            prof = 1/( (1 + x**2)**(3*self.beta/2) )
        else:
            prefix = rho_c * f_gas 
            prof = prefix[:, None] / ((1 + x**2 )**(3 * self.beta / 2) )

        if truncate is True:
            RVIR, R = np.meshgrid(self.truncate_param*r_vir, r_use)
            prof = np.where((R<RVIR).T, prof, 0)
            
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _fourier_integral(self, x, M, r_vir, cosmo=cosmo, scale_a=1, xDel_ratio=1/0.05, no_prefix=False):
        """ Function to integrate for the Fedeli Fourier transform: 
        prof_fourier(k) = prof_real * x, integrated over x from 0 to 1, weighted by sin(k * R_delta * x) 
        & multiplied by prefix of 4*pi*R_delta^2 / k"""
        prof_real = self._real(cosmo=cosmo, r=x*r_vir, M=M, scale_a=scale_a, xDel_ratio=xDel_ratio, no_prefix=no_prefix)
        integral = prof_real * x
        return integral

    def _interpol_fourier(self, M_array, cosmo=cosmo, scale_a=1, xDel_ratio=1/0.05, no_prefix=False):  
        # interpolator for Fedeli's Fourier interpolator, over k
        k_list = np.geomspace(self.krange[0], self.krange[1], self.nk) 
        I0_array = np.zeros((len(M_array), self.nk))
        M_array = np.atleast_1d(M_array)
        
        for i, mass in enumerate(M_array):
            r_vir = self.mass_def.get_radius(cosmo, mass, scale_a) / scale_a  
            for j, k_value in enumerate(k_list):
                int_bob = integrate.quad(self._fourier_integral, self.limInt[0], self.limInt[1], # 0, 1, 
                                         args=(mass, r_vir, cosmo, scale_a, xDel_ratio, no_prefix), weight="sin", wvar=r_vir*k_value)[0] 
                I0_array[i, j] = int_bob * 4 * np.pi * (r_vir **2) / k_value
            func_fourier = interpol.interp1d(k_list, I0_array) 
        return func_fourier
    
    def _fourier_numerical(self, cosmo, k, M, scale_a=1, interpol_true=True, k2=np.geomspace(1E-2,9E1, 100),
                           r_c = 1, xDel_ratio=1/0.05, no_prefix=False): 
        """ X
        """
        k_use = np.atleast_1d(k)
        M_use = np.atleast_1d(M)

        # r_c = 
        r_vir = self.mass_def.get_radius(cosmo, M_use, scale_a) / scale_a    # R_delta = the halo virial radius r_vir
        f_gas = self._f_gas(M_use)

        if interpol_true is True:
            if self._func_fourier is None:
                with UnlockInstance(self):
                    self._func_fourier = self._interpol_fourier(M_use, cosmo, scale_a, xDel_ratio, no_prefix) 
        # giving it the Masses above when setting up the profile
            prof = self._func_fourier(k_use)
            
        else:
            k2_use = np.atleast_1d(k2)  # need to do numerical FT over smaller range of k
            prof_array = np.zeros((len(M_use), len(k2_use))) 
            for i, mass in enumerate(M_use):
                for j, k_value in enumerate(k_use):
                    integrated = integrate.quad(self._fourier_integral, self.limInt[0], self.limInt[1], # 0, 1, 
                                                args=(mass, r_vir[i]), weight="sin", wvar=r_vir[i]*k_value)[0]
                    prof_array[i,j] = integrated * 4 * np.pi * (r_vir[i] **2) / k_value
            prof = prof_array
            
        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof