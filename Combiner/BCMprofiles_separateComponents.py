__all__ = ("BCM_StellarProfile", "BCM_EjectedGasProfile", "BCM_BoundGasProfile", "BCM_CDMProfile")

import numpy as np
import pyccl as ccl
import matplotlib.pyplot as plt
from scipy import signal
import scipy.integrate as integrate
import scipy.interpolate as interpol
from pyccl._core import UnlockInstance

class StellarProfileBCM(ccl.halos.profiles.profile_base.HaloProfile):
    def __init__(self, *, mass_def, A_star=0.03, sigma_star=1.2, M_star=10**(12.5)):
        self.A_star = A_star
        self.sigma_star = sigma_star
        self.M_star = M_star
        super().__init__(mass_def=mass_def)

    def update_parameters(self, M_star=None, A_star=None, sigma_star=None):
        if M_star is not None:
            self.M_star = M_star
        if A_star is not None:
            self.A_star = A_star
        if sigma_star is not None:
            self.sigma_star = sigma_star

    def _real(self, cosmo, r, M, a):
        pass

class BoundGasProfileBCM(ccl.halos.profiles.profile_base.HaloProfile):
    def __init__(self, *, mass_def, Gamma=1.2, M_c=10**(13.5), beta=0.6,
                 gammaRange = (3, 20), ngamma=64, qrange=(1e-4, 1e2), nq=64,
                 limInt=(1E-3, 5E3), ):
        self.Gamma = Gamma
        self.M_c = M_c
        self.beta = beta
        super().__init__(mass_def=mass_def)

    def update_parameters(self, Gamma=None, M_c=None, beta=None):
        if M_c is not None:
            self.M_c = M_c
        if Gamma is not None:
            self.Gamma = Gamma
        if beta is not None:
            self.beta = beta

    def _real(self, cosmo, r, M, a):
        pass

class BCM_Initialiser(ccl.halos.profiles.profile_base.HaloProfile):
    """ Contains the __init__ , update_parameters, _f_stell & _f_bd methods to be inherited.
    """
    
    def __init__(self, cosmo, mass_def, concentration, Gamma, fourier_analytic = True, delta=200, eta_b=0.5, # for r_e in prof_ej
                 gammaRange = (3, 20), ngamma=64, qrange=(1e-4, 1e2), nq=64, limInt=(1E-3, 5E3), 
                 beta=0.6, M_c = 10**(13.5), M_star = 10**(12.5), A_star = 0.03, sigma_star = 1.2, 
                 projected_analytic=False, cumul2d_analytic=False, truncated=True):
        
        super(BCM_Initialiser, self).__init__(mass_def=mass_def, concentration=concentration)

        self.fourier_analytic = fourier_analytic
        if fourier_analytic is True:
            self._fourier = self._fourier_analytic
        self.truncated = truncated
        self.Gamma = Gamma

        self.delta = delta 
        self.eta_b = eta_b # for r_e in prof_ej

        self.gammaRange = gammaRange
        self.ngamma = ngamma
        self.limInt = limInt
        self.qrange = qrange
        self.nq = nq
        
        self._func_normQ0 = None   # General normalised profile (for q=0, over Gamma)
        self._func_normQany = None
        
        self.cosmo = cosmo
        self.beta = beta
        self.M_c = M_c
        self.M_star = M_star
        self.A_star = A_star
        self.sigma_star = sigma_star

        self.f_bar_b = self.cosmo['Omega_b']/self.cosmo['Omega_m']
        self.f_c = 1 - self.f_bar_b

        self.cdmProfile = ccl.halos.profiles.nfw.HaloProfileNFW(mass_def=mass_def, concentration=concentration, fourier_analytic=fourier_analytic, truncated=truncated)         # have [truncated] nfw profile initialised for cdm here, for inheritence later
        
    def _f_stell(self, M):
        f_stell = self.A_star * np.exp( (-1/2)* (np.log10(M / self.M_star) / self.sigma_star)**2 )
        return f_stell

    def _f_bd(self, M): 
        """ f_bd, f_stell = self._f_bd(M)
            f_ej = self.f_bar_b - f_stell - f_bd """
        f_stell = self._f_stell(M)
        f_b = (self.f_bar_b - f_stell) / (1 + (self.M_c / M)**self.beta )
        return f_b, f_stell

    def update_parameters(self, cosmo=None, mass_def=None, concentration=None, Gamma=None, fourier_analytic=None, delta=None, eta_b = None, gammaRange=None, ngamma=None, qrange=None, nq=None, limInt=None, beta=None, M_c=None, M_star=None, A_star=None, sigma_star=None, truncated=None):
        """Update any of the parameters associated with this profile.
        Any parameter set to ``None`` won't be updated.
        """
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
        if truncated is not None and truncated is True:
            self.truncated = truncated
            re_nfw = True
        if Gamma is not None and Gamma != self.Gamma:
            self.Gamma = Gamma
        if delta is not None and delta != self.delta:
            self.delta = delta
        if eta_b is not None and eta_b != self.eta_b:
            self.eta_b = eta_b

        if re_nfw is True:
            self.cdmProfile = ccl.halos.profiles.nfw.HaloProfileNFW(mass_def=mass_def, concentration=concentration, fourier_analytic=fourier_analytic, truncated=truncated) 
        
        if beta is not None and beta != self.beta:
            self.beta = beta
        if M_c is not None and M_c != self.M_c:
            self.M_c = M_c
        if M_star is not None and M_star != self.M_star:
            self.M_star = M_star
        if A_star is not None and A_star != self.A_star:
            self.A_star = A_star
        if sigma_star is not None and sigma_star != self.sigma_star:
            self.sigma_star = sigma_star

        if cosmo is not None and cosmo != self.cosmo:
            self.cosmo = cosmo
            self.f_bar_b = self.cosmo['Omega_b']/self.cosmo['Omega_m']
            self.f_c = 1 - self.f_bar_b

        # Check if we need to recompute the interpolators
        re_normQ0 = False   
        re_normQany = False
        if limInt is not None and limInt != self.limInt:
            re_normQ0 = True
            re_normQany = True
            self.limInt = limInt
        if gammaRange is not None and gammaRange != self.gammaRange:
            re_normQ0 = True  
            re_normQany = True
            self.gammaRange = gammaRange
        if ngamma is not None and gamma != self.ngamma:
            re_normQ0 = True   
            re_normQany = True
            self.ngamma = ngamma
        if qrange is not None and qrange != self.qrange:
            re_normQany = True
            self.qrange = qrange
        if nq is not None and nq != self.nq:
            re_normQany = True
            self.nq = nq

        if re_normQ0 is True and (self._func_normQ0 is not None):
            self._func_normQ0 = self._norm_interpol1() 
        if re_normQany is True and (self._func_normQany is not None):
            self._func_normQany = self._norm_interpol2()
            

class CDMProfile(BCM_Initialiser): #ccl.halos.profiles.nfw.HaloProfileNFW): 
    """Density profile for the cold dark matter (cdm), using the Navarro-Frenk-White, multiplied by the cdm's mass fraction (unless no_fraction is set to True in the real & analytical Fourier methods below).
    
    """

    def __init__(self, ):
        #####
        #####

    def update_parameters(self, ):
        #####
        #####

    def _f_cdm(self, cosmo):
        f_c = 1 - cosmo['Omega_b']/cosmo['Omega_m']   # f_bar_b = 1 - f_c = cosmo['Omega_b']/cosmo['Omega_m']
        return f_c

    def _real(self, cosmo, r, M, scale_a=1, no_fraction=False):
        if no_fraction is True:
            f = 1
        else:
            f = self._f_cdm(cosmo)
        prof = f * self.cdmProfile._real(self.cosmo, r, M, scale_a) 
        return prof

    def _fourier_analytic(self, k, M, scale_a=1, no_fraction=False):
        if no_fraction is True:
            f = 1
        else:
            f = self._f_cdm(cosmo)
        prof = f * self.cdmProfile._fourier(self.cosmo, k, M, scale_a) 
        return prof

class StellarProfile(BCM_Initialiser): 
    """Creating a class for the stellar density profile \
    where: 
    .. math::
    
        \\rho_*(r)\ = Ma ^{-3} g_*(r)\  & g_*(r)\ \\equiv \\delta^D(\\mathbf{x}) (\\text{a Dirac delta function centred at }r=0). 
    The normalised Fourier profile is then given by: 
    .. math::
        \\tilde{g}_*(k)\ = 1.

    Inherits __init__ , update_parameters, & _f_stell methods from Initialiser parent class.

    To do later: Change real profile of stellar from delta function to (...)
    """ 

    def __init__(self, ):
        ####
        ####

    def update_parameters(self, ):
        ####
        ####

    def _f_stell(self, M):
        f_stell = self.A_star * np.exp( (-1/2)* (np.log10(M / self.M_star) / self.sigma_star)**2 )
        return f_stell

    # To do later: Change real profile of stellar from delta function to (...)
    def _real(self, cosmo, r, M, scale_a=1, centre_pt=None, no_fraction=False): 
        # want delta centred at r=0 (& since log scale, can't do negative or zero values in array)
        r_use = np.atleast_1d(r) 
        M_use = np.atleast_1d(M)
        len_r = len(r_use) 

        if no_fraction is True:
            f = 1
        else:
            f = self._f_stell(M_use) # f = f_stell
        prefix = f * M_use / scale_a**3
        prof = prefix[:, None] * signal.unit_impulse(len_r, centre_pt)[None,:] # If centre_pt=None, defaults to index at the 0th element.

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
                                                          
        return prof

    def _fourier_analytic(self, k, M, scale_a=1, no_fraction=False):
        k_use = np.atleast_1d(k)
        M_use = np.atleast_1d(M)

        if no_fraction is True:
            f = 1
        else:
            f = self._f_stell(M_use) # f = f_stell
        prefix = f * M_use / scale_a**3
        prof = np.ones_like(k_use)[None,:] * prefix[:, None] # k_use[None,:] + prefix[:, None] * 1 # as g(k) = 1

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)

        return prof

class EjectedGasProfile(BCM_Initialiser): 
    """Creating a class for the ejected gas density profile
    where: 
    """  

    

    def _f_ej(self, cosmo, M):
        """ f_ej = self.f_bar_b - f_stell - f_bd. f_bar_b = cosmo['Omega_b']/cosmo['Omega_m'"""
        f_stell = self.A_star * np.exp( (-1/2)* (np.log10(M / self.M_star) / self.sigma_star)**2 )
        f_bd = (cosmo['Omega_b']/cosmo['Omega_m'] - f_stell) / (1 + (self.M_c / M)**self.beta )
        f_ej = cosmo['Omega_b']/cosmo['Omega_m'] - f_stell - f_bd
        return f_ej

    def _real(self, cosmo, r, M, scale_a=1, no_fraction=False): 
        r_use = np.atleast_1d(r) 
        M_use = np.atleast_1d(M)
        r_vir = self.mass_def.get_radius(self.cosmo, M_use, scale_a) / scale_a    # halo virial radius
        r_e = 0.375*r_vir*np.sqrt(self.delta)*self.eta_b                                    # eta_b = a free parameter

        if no_fraction is True:
            f = 1
        else:
            f = self._f_ej(cosmo, M_use)
        prefix = f * M_use / (scale_a*r_e*np.sqrt(2*np.pi))**3  
        x = r_use[None, :] / r_e[:, None]
        prof = prefix[:, None] * np.exp(-(x**2)/2)

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
                                                          
        return prof

    def _fourier_analytic(self, k, M, scale_a=1, no_fraction=False):
        k_use = np.atleast_1d(k)
        M_use = np.atleast_1d(M)
        r_vir = self.mass_def.get_radius(self.cosmo, M_use, scale_a) / scale_a    # halo virial radius
        r_e = 0.375*r_vir*np.sqrt(self.delta)*self.eta_b                                    # eta_b = a free parameter

        if no_fraction is True:
            f = 1
        else:
            f = self._f_ej(cosmo, M_use)
        prefix = f * M_use / scale_a**3
        x = k_use[None, :] * r_e[:, None]
        prof = prefix[:, None] * np.exp(-(x**2)/2)  

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)

        return prof

class BoundGasProfile(BCM_Initialiser): 
    """Creating a class for the bound gas density profile where: 
    
    .. math::
    
        \\rho_b(r)\ = Ma ^{-3} & g_b(r)\ = \\frac{1}{V_b} \\left( \\frac{log(1 + \\frac{r}{r_s})}{\\frac{r}{r_s}} \\right)^{\\frac{1}{\\Gamma - 1}}     \\text{, where log} \equiv \\text{ ln.}    
        V_b \\equiv 4\\pi r_s^3 I_b(\\frac{1}{\\Gamma - 1}, 0)\ .   
        I_b(\\gamma, q)\ = \\int^{\\infty}_0 dx\ x^2 \\left( \\frac{log(1+x)}{x} \\right)^{\\gamma} j_0(qx)\, with q = kr_s \\text{[in Fourier space].}  
        \\to I_b(\\frac{1}{\\Gamma - 1}, 0)\ = \\int^{\\infty}_0 dx\ x^2 \\left( \\frac{log(1+x)}{x} \\right)^{\\frac{1}{\\Gamma - 1}} j_0(0)\  = \int^{\infty}_0 dx\ x^2 \left( \frac{log(1+x)}{x} \right)^{\frac{1}{\Gamma - 1}} 
        \\text{As } j_0 \\text{ is a Besel function, & } j_0(0)\ = 1 .

    Therefore: 
    
    .. math::
    
        \\rho_x(r)\ = \\frac{M f_x\ }{4\\pi r_s^3 a^{3}} \\frac{1}{\\int^{\\infty}_0 dx\ x^2 \\left( \\frac{log(1+x)}{x} \\right)^{\\frac{1}{\\Gamma - 1}}} \\left( \\frac{log(1 + \\frac{r}{r_s})}{\\frac{r}{r_s}} \\right)^{\\frac{1}{\\Gamma - 1}}.
    
    The normalised Fourier profile is then given by: 
    
    .. math::
    
        \\tilde{g}_b(k)\ = \\frac{I_b(1/(\\Gamma - 1),q)\ }{I_b(1/(\\Gamma - 1),0)\ } , with q = kr_s.

    Inherits __init__ , update_parameters, _f_stell & _f_bd methods from Initialiser parent class.    
    """  

    def _f_bd(self, cosmo, M): 
        """f_bd requires f_stell, hence the self.X_star parameters, f_bar_b = cosmo['Omega_b']/cosmo['Omega_m'"""
        f_stell = self.A_star * np.exp( (-1/2)* (np.log10(M / self.M_star) / self.sigma_star)**2 )
        f_bd = (cosmo['Omega_b']/cosmo['Omega_m'] - f_stell) / (1 + (self.M_c / M)**self.beta )
        return f_bd
    
    def _shape(self, x, gam):
        gam_use = np.atleast_1d(gam)
        return (np.log(1+x)/x)**gam_use

    def _innerInt(self, x, gam): 
        return x**2 * self._shape(x, gam)   

    def _Vb_prefix(self, gam, r_s=1):
        vB1 = integrate.quad(self._innerInt, self.limInt[0], self.limInt[1], args = gam)  
        vB2 = 4*np.pi*(r_s**3)*vB1[0]
        return vB2

    def _norm_interpol1(self):  # interpol1 = for q = 0
        gamma_list = np.linspace(self.gammaRange[0], self.gammaRange[1], self.ngamma) 
        I0_array = np.zeros(self.ngamma)
        for i, g in enumerate(gamma_list):
            I0_array[i] =  integrate.quad(self._innerInt, self.limInt[0], self.limInt[1], args = g)[0] 
        func_normQ0 = interpol.interp1d(gamma_list, I0_array) 
        return func_normQ0
        
    def _real(self, cosmo, r, M, scale_a=1, call_interp=True, no_fraction=False): 
        r_use = np.atleast_1d(r) 
        M_use = np.atleast_1d(M)
        
        R_M = self.mass_def.get_radius(self.cosmo, M_use, scale_a) / scale_a       # halo virial radius
        c_M = self.concentration(self.cosmo, M_use, scale_a)                       # concentration-mass relation c(M)
        r_s = R_M / c_M                                                            # characteristic scale r_s

        if call_interp is False:
            print(call_interp)
            vB_prefix = self._Vb_prefix(1/(self.Gamma-1), r_s)
        else:
            if self._func_normQ0 is None: # is instead of == here
                with UnlockInstance(self):
                    self._func_normQ0 = self._norm_interpol1() 
            vB_prefix = 4*np.pi*(r_s**3)*self._func_normQ0(1/(self.Gamma-1))

        if no_fraction is True:
            f = 1
        else:
            f = self._f_bd(cosmo, M_use)
        prefix = f * M_use * (1/scale_a**3) * (1/vB_prefix)

        x = r_use[None, :] / r_s[:, None]
        prof = prefix[:, None] * self._shape(x, 1/(self.Gamma-1)) 

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
                                                          
        return prof

    def _norm_interpol2(self):  # interpol1 for q = any
        gamma_list = np.linspace(self.gammaRange[0], self.gammaRange[1], self.ngamma) 
        q_array = np.geomspace(self.qrange[0], self.qrange[1], self.nq) #
        I0_array =  np.zeros((self.ngamma, self.nq))

        def integralQany(x, gam): 
            return x * self._shape(x, gam) 
            
        for i, g in enumerate(gamma_list): 
            for j, q in enumerate(q_array): 
                I0_array[i, j] =  integrate.quad(integralQany, self.limInt[0], self.limInt[1], args = g, weight = "sin", wvar=q)[0] / q
            print(f'Qany = {100*(i+1)/self.ngamma:.3g}% through')
        func_normQany = interpol.RegularGridInterpolator((gamma_list, np.log(q_array)), I0_array)
        return func_normQany
    
    def _fourier_analytic(self, k, M, scale_a=1, no_fraction=False):
        k_use = np.atleast_1d(k)
        M_use = np.atleast_1d(M)

        R_M = self.mass_def.get_radius(self.cosmo, M_use, scale_a) / scale_a       # halo virial radius
        c_M = self.concentration(self.cosmo, M_use, scale_a)                       # concentration-mass relation c(M)
        r_s = R_M / c_M                                                            # characteristic scale r_s

        if self._func_normQ0 is None:
            with UnlockInstance(self):
                self._func_normQ0 = self._norm_interpol1() 
        if self._func_normQany is None:
            with UnlockInstance(self):
                self._func_normQany = self._norm_interpol2()

        q_use = k_use[None, :]*r_s[:, None]
        g0 = self._func_normQ0(1/(self.Gamma-1))
        gAny = self._func_normQany((1/(self.Gamma-1), np.log(q_use)))
        g_k = gAny/g0 

        if no_fraction is True:
            f = 1
        else:
            f = self._f_bd(cosmo, M_use)
        prefix = f * M_use / scale_a**3
        prof = prefix[:, None] * g_k[None,:] 

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)

        return prof
