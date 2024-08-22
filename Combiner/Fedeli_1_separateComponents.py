__all__ = ("StellarProfileSAM", "GasProfileSAM", "CDMProfile")

import numpy as np
import pyccl as ccl
import matplotlib.pyplot as plt
from scipy import integrate
import scipy.interpolate as interpol
from scipy.special import erf, gamma, hyp2f1
from pyccl._core import UnlockInstance

class CDMProfile(ccl.halos.profiles.profile_base.HaloProfile): #ccl.halos.profiles.nfw.HaloProfileNFW): 
    """Density profile for the cold dark matter (cdm), using the Navarro-Frenk-White, multiplied by the cdm's mass fraction (unless no_fraction is set to True in the real & analytical Fourier methods below).
    
    """

    def __init__(self, mass_def, concentration, fourier_analytic=True):
        super().__init__(mass_def=mass_def, concentration=concentration)
        self.fourier_analytic = fourier_analytic
        if fourier_analytic is True:
            self._fourier = self._fourier_analytic
            
        self.cdmProfile = ccl.halos.profiles.nfw.HaloProfileNFW(mass_def=mass_def, concentration=concentration, fourier_analytic=fourier_analytic)         # have truncated nfw profile initialised for cdm 
        # (so can leave out putting in truncated=True par, as should always be true)

    
    def update_parameters(self, mass_def=None, concentration=None, fourier_analytic=None):
        re_nfw = False # Check if we need to re-compute the [truncated] nfw profile for the cdm
        if mass_def is not None and mass_def != self.mass_def:
            self.mass_def = mass_def
            re_nfw = True
        if concentration is not None and concentration != self.concentration:
            self.concentration = concentration
            re_nfw = True
        if fourier_analytic is not None and fourier_analytic is True: 
            self._fourier = self._fourier_analytic    
            re_nfw = True

        if re_nfw is True:
            self.cdmProfile = ccl.halos.profiles.nfw.HaloProfileNFW(mass_def=self.mass_def, concentration=self.concentration, fourier_analytic=self.fourier_analytic) 

    def _f_cdm(self, cosmo):
        f_c = 1 - cosmo['Omega_b']/cosmo['Omega_m']   # f_bar_b = 1 - f_c = cosmo['Omega_b']/cosmo['Omega_m']
        return f_c

    def _real(self, cosmo, r, M, scale_a=1, no_fraction=False):
        if no_fraction is True:
            f = 1
        else:
            f = self._f_cdm(cosmo)
        prof = f * self.cdmProfile._real(cosmo, r, M, scale_a) 
        return prof

    def _fourier_analytic(self, cosmo, k, M, scale_a=1, no_fraction=False):
        if no_fraction is True:
            f = 1
        else:
            f = self._f_cdm(cosmo)
        prof = f * self.cdmProfile._fourier(cosmo, k, M, scale_a) 
        return prof


class StellarProfileSAM(ccl.halos.profiles.profile_base.HaloProfile):
    """ Stellar halo density profile. Fedeli (2014) arXiv:1401.2997
    """

    def __init__(self, mass_def, mass_func, alpha=1, r_t=1, xDelta_stel = 1/0.03, m_0s_prefix=5E12, sigma_s=1.2, rho_avg_star_prefix=7E8, limInt_mStell=(1E10, 1E15), m_0s=None, rho_avg_star=None):
              #  fourier_analytic=True, fourier_numerical=True, truncate_param=1):
        super().__init__(mass_def=mass_def)
        self.mass_func = mass_func

  #     self.fourier_analytic = fourier_analytic
   #     if fourier_analytic is True:
    #        self._fourier = self._fourier_analytic
        
        self.alpha = alpha
        self.r_t = r_t
        self.xDelta_stel = xDelta_stel
        self.sigma_s = sigma_s
        self.limInt_mStell = limInt_mStell

        self.m_0s = m_0s
        self.m_0s_prefix = m_0s_prefix
        self.rho_avg_star = rho_avg_star
        self.rho_avg_star_prefix = rho_avg_star_prefix
        
    #    self.truncate_param = truncate_param # if truncate=True in real, truncate at r > (r_vir * truncate_param)

    #    self.limInt = limInt
     #   self.krange = krange
      #  self.nk = nk
       # self._func_fourier = None   # [Normalised] profile from the Fourier interpolator (for Fedeli's Fourier integral)

    def _f_stell_noA(self, cosmo, M):
        if self.m_0s is None:
            m_0s = self.m_0s_prefix/cosmo['h']
        else:
            m_0s = self.m_0s
        return np.exp( (-1/2) * ( np.log10(M/m_0s) /self.sigma_s )**2 )
    
    def _f_stell_integrand(self, M, cosmo):
        # integrand = m * f_star(m) * n(m), where n(m,z) is the standard DM-only halo mass function
        #  DM_mass_func = hmf_200m(self.cosmo, np.atleast_1d(M), 1) / (np.atleast_1d(M)*np.log(10))
        DM_mass_func = self.mass_func(cosmo, np.atleast_1d(M), 1) / (np.atleast_1d(M)*np.log(10))
        return M * self._f_stell_noA(cosmo, M) * DM_mass_func 
     
    def _f_stell(self, cosmo, M):
        if self.rho_avg_star is None:
            rho_avg_star = self.rho_avg_star_prefix**cosmo['h']**2 
        else:
            rho_avg_star = self.rho_avg_star
        # f_star(m) = A*np.exp( (-1/2) * ( np.log10(m/m_0s) /omega_s )**2 )
        integrad = integrate.quad(self._f_stell_integrand, self.limInt_mStell[0], self.limInt_mStell[1], args=cosmo)  # integrating over m (dm)
        A = rho_avg_star / integrad[0] 
        return A * self._f_stell_noA(cosmo, M)

    def update_parameters(self, mass_def=None, mass_func=None, alpha=None, r_t=None, xDelta_stel=None, sigma_s=None, limInt_mStell=None, m_0s_prefix=None, rho_avg_star_prefix=None, m_0s=None, rho_avg_star=None, ):
        """Update any of the parameters associated with this profile.
        Any parameter set to ``None`` won't be updated.
        """
        if mass_def is not None and mass_def != self.mass_def:
            self.mass_def = mass_def
        if mass_func is not None and mass_func != self.mass_func:
            self.mass_func = mass_func

        if alpha is not None and alpha != self.alpha:
            self.alpha = alpha
        if r_t is not None and r_t != self.r_t:
            self.r_t = r_t
        if xDelta_stel is not None and xDelta_stel != self.xDelta_stel:
            self.xDelta_stel = xDelta_stel
        if sigma_s is not None and sigma_s != self.sigma_s:
            self.sigma_s = sigma_s
        if limInt_mStell is not None and limInt_mStell != self.limInt_mStell:
            self.limInt_mStell = limInt_mStell

        if m_0s is not None and m_0s != self.m_0s:
            self.m_0s = m_0s
        if m_0s_prefix is not None and m_0s_prefix != self.m_0s_prefix:
            self.m_0s_prefix = m_0s_prefix
        if rho_avg_star is not None and rho_avg_star != self.rho_avg_star:
            self.rho_avg_star = rho_avg_star
        if rho_avg_star_prefix is not None and rho_avg_star_prefix != self.rho_avg_star_prefix:
            self.rho_avg_star_prefix = rho_avg_star_prefix
            
        #######
        
    def _real(self, cosmo, r, M, scale_a=1, no_fraction=False):
        """ X
        """
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        r_vir = self.mass_def.get_radius(cosmo, M_use, scale_a) / scale_a    # R_delta = the halo virial radius r_vir
        if no_fraction is True:
            f = 1
        else:
            f = self._f_stell(cosmo, M_use)
        
        if self.xDelta_stel is None:
            r_t = self.r_t
            x_delta = r_vir / self.r_t # use the inputted value of r_t
        else:
            # default: x_delta = 1/0.03, as in Fedeli 2014 paper 
            x_delta = self.xDelta_stel  # reassign r_c in order to give the specific r_c/r_Del ratio desired 
            r_t = r_vir / x_delta
            
        nu_alpha = 1 - (2 / self.alpha)
        # Using E_1 = int^infty_1 e^{-xt} * t dt = (e^{-x}*(x+1))/(x^2), assuming x = alpha here
        rho_t_bracket = gamma(1 - nu_alpha) - (x_delta**2)*(x_delta**self.alpha)*(np.exp(-nu_alpha)*(nu_alpha+1))/(nu_alpha**2)
        rho_t = M_use*f*self.alpha / (4*np.pi*(r_t**3) * rho_t_bracket)

        x = r_use[None, :] / r_t[:, None]
       # prefix = rho_t
        prof = rho_t[:, None] * np.exp(-x**self.alpha)/x # prefix[:, None] * np.exp(-x**self.alpha)/x 

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

class GasProfileSAM(ccl.halos.profiles.profile_base.HaloProfile):
    """ Gas halo density profile. Fedeli (2014) arXiv:1401.2997
    """

    def __init__(self, mass_def, fourier_numerical=True, beta=2/3, r_c = 1, xDelta_gas = 1/0.05, limInt=(0,1), nk=64, krange=(5E-3, 5E2), m_0g=None, m_0g_prefix = 5E12, sigma_g = 1.2, truncate_param=1):
        super().__init__(mass_def=mass_def)
        
        self.fourier_numerical = fourier_numerical
        if fourier_numerical is True:
            self._fourier = self._fourier_numerical

        self.m_0g = m_0g
        self.m_0g_prefix = m_0g_prefix

        self.beta=beta
        self.r_c = r_c
        self.xDelta_gas = xDelta_gas
        self.truncate_param = truncate_param # if truncate=True in real, truncate at r > (r_vir * truncate_param)
        self.sigma_g = sigma_g

        self.limInt = limInt
        self.krange = krange
        self.nk = nk
        self._func_fourier = None   # [Normalised] profile from the Fourier interpolator (for Fedeli's Fourier integral)

    def _f_gas(self, cosmo, M):
        M_use = np.atleast_1d(M)
        f_array = np.zeros(np.shape(M_use))
        if self.m_0g is None:
            m_0g = self.m_0g_prefix/cosmo['h']
        else:
            m_0g = self.m_0g

        for i, mass in enumerate(M_use):
            if (mass < m_0g):
                f_array[i] = 0
            else:
                f_array[i] = (cosmo['Omega_b']/cosmo['Omega_m']) * erf(np.log10(mass/m_0g) / self.sigma_g)
        return f_array

    def update_parameters(self, mass_def=None, m_0g_prefix=None, m_0g=None, beta=None, r_c=None, xDelta_gas=None, sigma_g=None, fourier_numerical=None, limInt=None, nk=None, krange=None, truncate_param=None):
        """Update any of the parameters associated with this profile.
        Any parameter set to ``None`` won't be updated.
        """
        if mass_def is not None and mass_def != self.mass_def:
            self.mass_def = mass_def
            
        if m_0g is not None and m_0g != self.m_0g:
            self.m_0g = m_0g
        if m_0g_prefix is not None and m_0g_prefix != self.m_0g_prefix:
            self.m_0g_prefix = m_0g_prefix
        
        if beta is not None and beta != self.beta:
            self.beta = beta
        if r_c is not None and r_c != self.r_c:
            self.r_c = M_c
        if xDelta_gas is not None and xDelta_gas != self.xDelta_gas:
            self.xDelta_gas = xDelta_gas
        if sigma_g is not None and sigma_g != self.sigma_g:
            self.sigma_g = sigma_g

        if fourier_numerical is not None and fourier_numerical is True: 
            self._fourier = self._fourier_numerical   

        # Check if we need to recompute the (optional) interpolator for the gas
        re_func_fourier = False   
        if limInt is not None and limInt != self.limInt:
            re_normQ0 = True
            self.limInt = limInt #nq
        if kRange is not None and kRange != self.kRange:
            re_func_fourier = True  
            self.kRange = kRange
        if nk is not None and nk != self.nk:  
            re_func_fourier = True
            self.nk = nk
        if truncate_param is not None and truncate_param != self.truncate_param:
            re_func_fourier = True
            self.truncate_param = truncate_param
            
# need to recall the interpolator function for the gas Fourier profile 
## BUT this relies on calling the real profile (with the given masses, & so on)
## so instead of recalling the interpolator here: set it to None, so it will be recalculated when the Fourier method is next called
        if re_func_fourier is True and (self._func_fourier is not None):  
            self._func_fourier = None
        
    def _real(self, cosmo, r, M, scale_a=1, truncate=True, # for inbuilt FFT, need truncation to be default
              no_prefix=False, no_fraction=False): 
        """ X
        """
        
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        r_vir = self.mass_def.get_radius(cosmo, M_use, scale_a) / scale_a    # R_delta = the halo virial radius r_vir
        if no_fraction is True:
            f = 1
        else:
            f = self._f_gas(cosmo, M_use)

        if self.xDelta_gas is None:
            x_delta = r_vir / self.r_c # use the inputted value of r_c
            r_c = self.r_c
        else:
            # default: x_delta = 1/0.05, as in Fedeli 2014 paper 
            x_delta = self.xDelta_gas  # reassign r_c in order to give the specific r_c/r_Del ratio desired 
            r_c = r_vir / x_delta

        rho_bracket = (x_delta**3) * hyp2f1(3/2, 3*self.beta/2, 5/2, -(x_delta**2))
        rho_c = 3 * M_use * f / (4 * np.pi * (r_c**3) * rho_bracket)
        
        x = r_use[None, :] / r_c[:, None]

        if no_prefix is True:
            prof = 1/( (1 + x**2)**(3*self.beta/2) )
        else:
          #  prefix = rho_c 
            prof = rho_c[:, None] / ((1 + x**2 )**(3 * self.beta / 2) ) # prefix[:, None] / ((1 + x**2 )**(3 * self.beta / 2) )

        if truncate is True:
            RVIR, R = np.meshgrid(self.truncate_param*r_vir, r_use)
            prof = np.where((R<RVIR).T, prof, 0)
            
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _fourier_integral(self, x, M, r_vir, cosmo, scale_a=1, no_prefix=False, no_fraction=False):
        """ Function to integrate for the Fedeli Fourier transform: 
        prof_fourier(k) = prof_real * x, integrated over x from 0 to 1, weighted by sin(k * R_delta * x) 
        & multiplied by prefix of 4*pi*R_delta^2 / k"""
        prof_real = self._real(cosmo=cosmo, r=x*r_vir, M=M, scale_a=scale_a, no_prefix=no_prefix, no_fraction=no_fraction)
        integral = prof_real * x
        return integral

    def _interpol_fourier(self, M_array, cosmo, scale_a=1, no_prefix=False, no_fraction=False):  
        # interpolator for Fedeli's Fourier interpolator, over k
        k_list = np.geomspace(self.krange[0], self.krange[1], self.nk) 
        I0_array = np.zeros((len(M_array), self.nk))
        M_array = np.atleast_1d(M_array)
        
        for i, mass in enumerate(M_array):
            r_vir = self.mass_def.get_radius(cosmo, mass, scale_a) / scale_a  
            for j, k_value in enumerate(k_list):
                int_bob = integrate.quad(self._fourier_integral, self.limInt[0], self.limInt[1], 
                                    args=(mass, r_vir, cosmo, scale_a, no_prefix, no_fraction), weight="sin", wvar=r_vir*k_value)[0] 
                I0_array[i, j] = int_bob * 4 * np.pi * (r_vir **2) / k_value
            func_fourier = interpol.interp1d(k_list, I0_array) 
        return func_fourier
    
    def _fourier_numerical(self, cosmo, k, M, scale_a=1, interpol_true=True, k2=np.geomspace(1E-2,9E1, 100), no_prefix=False, no_fraction=False): 
        """ X
        """
        k_use = np.atleast_1d(k)
        M_use = np.atleast_1d(M)

        r_vir = self.mass_def.get_radius(cosmo, M_use, scale_a) / scale_a    # R_delta = the halo virial radius r_vir
        f_gas = self._f_gas(cosmo, M_use)

        if interpol_true is True:
            if self._func_fourier is None:
                with UnlockInstance(self):
                    self._func_fourier = self._interpol_fourier(M_use, cosmo, scale_a, no_prefix, no_fraction) 
        # giving it the Masses above when setting up the profile
            prof = self._func_fourier(k_use)
            
        else:
            k2_use = np.atleast_1d(k2)  # need to do numerical FT over smaller range of k
            prof_array = np.zeros((len(M_use), len(k2_use))) 
            for i, mass in enumerate(M_use):
                for j, k_value in enumerate(k_use):
                    integrated = integrate.quad(self._fourier_integral, self.limInt[0], self.limInt[1], 
                            args=(mass, r_vir[i], cosmo, scale_a, no_prefix, no_fraction), weight="sin", wvar=r_vir[i]*k_value)[0]
                    prof_array[i,j] = integrated * 4 * np.pi * (r_vir[i] **2) / k_value
            prof = prof_array
            
        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof
