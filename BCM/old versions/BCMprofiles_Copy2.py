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
    where: $\\rho_*(r)\ = Ma ^{-3} g_*(r)\ $ & $g_*(r)\ \\equiv \\delta^D$(**x**) (a Dirac delta funciton centred at $r=0$). 
    The normalised Fourier profile is then given by: $\\tilde{g}_*(k)\ = 1$.
    
    """  

    def __init__(self, cosmo, mass_def, fourier_analytic=True):
        super(StellarProfile, self).__init__(mass_def=mass_def)
        self.fourier_analytic = fourier_analytic
        self.cosmo = cosmo
        if fourier_analytic == True:
            self._fourier = self._fourier_analytic

    def _real(self, cosmo, r, M, scale_a=1, centre_pt=None): 
        # want delta centred at r=0 (& since log scale, can't do negative or zero values in array)
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

    def _fourier_analytic(self, k, M, scale_a=1):
        k_use = np.atleast_1d(k)
        M_use = np.atleast_1d(M)

        prefix = M_use / scale_a**3
        prof = np.ones_like(k_use)[None,:] * prefix[:, None] # k_use[None,:] + prefix[:, None] * 1 # as g(k) = 1

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)

        return prof

class EjectedGasProfile(ccl.halos.profiles.profile_base.HaloProfile): 
    """Creating a class for the ejected gas density profile
    where: """  # could put in the equations used

    def __init__(self, cosmo, mass_def, fourier_analytic = True): 
        super(EjectedGasProfile, self).__init__(mass_def=mass_def)
        self.cosmo = cosmo
        self.fourier_analytic = fourier_analytic
        if fourier_analytic == True:
            self._fourier = self._fourier_analytic

    def _real(self, cosmo, r, M, scale_a=1, delta=200, eta_b = 0.5): 
        r_use = np.atleast_1d(r) 
        M_use = np.atleast_1d(M)
        r_vir = self.mass_def.get_radius(self.cosmo, M_use, scale_a) / scale_a # halo virial radius
        r_e = 0.375*r_vir*np.sqrt(delta)*eta_b # eta_b = a free parameter
        
        prefix = M_use / (scale_a*r_e*np.sqrt(2*np.pi))**3  
        x = r_use[None, :] / r_e[:, None]
        prof = prefix[:, None] * np.exp(-(x**2)/2)

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
                                                          
        return prof

    def _fourier_analytic(self, k, M, scale_a=1, delta=200, eta_b = 0.5):
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
    """Creating a class for the bound gas density profile where: 
    .. math::
        \\rho_b(r)\ = Ma ^{-3} & g_b(r)\ = \\frac{1}{V_b} \\left( \\frac{log(1 + \\frac{r}{r_s})}{\\frac{r}{r_s}} \\right)^{\\frac{1}{\\Gamma - 1}}     , where log \equiv ln.    
        V_b \\equiv 4\\pi r_s^3 I_b(\\frac{1}{\\Gamma - 1}, 0)\ .   
        I_b(\\gamma, q)\ = \\int^{\\infty}_0 dx\ x^2 \\left( \\frac{log(1+x)}{x} \\right)^{\\gamma} j_0(qx)\, with q = kr_s [in Fourier space].  
        \\to I_b(\\frac{1}{\\Gamma - 1}, 0)\ = \\int^{\\infty}_0 dx\ x^2 \\left( \\frac{log(1+x)}{x} \\right)^{\\frac{1}{\\Gamma - 1}} j_0(0)\  = \int^{\infty}_0 dx\ x^2 \left( \frac{log(1+x)}{x} \right)^{\frac{1}{\Gamma - 1}} 
        As j_0 is a Besel function, & j_0(0)\ = 1 .

    Therefore: 
    .. math::
        \\rho_x(r)\ = \\frac{M f_x\ }{4\\pi r_s^3 a^{3}} \\frac{1}{\\int^{\\infty}_0 dx\ x^2 \\left( \\frac{log(1+x)}{x} \\right)^{\\frac{1}{\\Gamma - 1}}} \\left( \\frac{log(1 + \\frac{r}{r_s})}{\\frac{r}{r_s}} \\right)^{\\frac{1}{\\Gamma - 1}}.
    
    The normalised Fourier profile is then given by: 
    .. math::
    \\tilde{g}_b(k)\ = \\frac{I_b(1/(\\Gamma - 1),q)\ }{I_b(1/(\\Gamma - 1),0)\ } , with q = kr_s.
    
    """  

    def __init__(self, cosmo, mass_def, concentration, fourier_analytic = True, gammaRange = (3, 20), ngamma=64, qrange=(1e-4, 1e2), nq=64, limInt=(1E-4, 1E4)): 
        super(BoundGasProfile, self).__init__(mass_def=mass_def, concentration=concentration)
        self.fourier_analytic = fourier_analytic
        if fourier_analytic is True:
            self._fourier = self._fourier_analytic
        self.cosmo = cosmo

        self.gammaRange = gammaRange
        self.ngamma = ngamma

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
        vB1 = integrate.quad(self._innerInt,limInt[0], limInt[1], args = (1/(self.Gamma-1)))  
        vB2 = 4*np.pi*(r_s**3)*vB1[0]
        return vB2

    def _norm_interpol1(self):  # interpol1 = for q = 0
        gamma_list = np.linspace(self.gammaRange[0], self.gammaRange[1], self.ngamma) 
        I0_array = np.zeros(self.ngamma)
        for i, g in enumerate(gamma_list):
            I0_array[i] =  integrate.quad(self._innerInt, limInt[0], limInt[1], args = g)[0] 
        func_normQ0 = interpol.interp1d(gamma_list, I0_array) 
        return func_normQ0
        
    def _real(self, cosmo, r, M, scale_a=1, Gamma=1.1, call_interp=True): 
        r_use = np.atleast_1d(r) 
        M_use = np.atleast_1d(M)
        
        R_M = self.mass_def.get_radius(self.cosmo, M_use, scale_a) / scale_a # halo virial radius
        c_M = self.concentration(self.cosmo, M_use, scale_a) # concentration-mass relation c(M)
        r_s = R_M / c_M # characteristic scale r_s

        if call_interp is False:
            print(call_interp)
            vB_prefix = self._Vb_prefix(r_s)
        else:
            if self._func_normQ0 is None: # is instead of == here
                with UnlockInstance(self):
                    self._func_normQ0 = self._norm_interpol1() 
            vB_prefix = 4*np.pi*(r_s**3)*self._func_normQ0(1/(Gamma-1))  
        prefix = M_use * (1/scale_a**3) * (1/vB_prefix)

        x = r_use[None, :] / r_s[:, None]
        prof = prefix[:, None] * self._shape(x, 1/(Gamma-1)) 

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
                I0_array[i, j] =  integrate.quad(integralQany, limInt[0], limInt[1], args = g, weight = "sin", wvar=q)[0] / q
            print(f'Qany = {100*(i+1)/self.nGamma:.3g}% through')
        func_normQany = interpol.RegularGridInterpolator((gamma_list, np.log(q_array)), I0_array)
        return func_normQany
    
    def _fourier_analytic(self, k, M, scale_a=1, Gamma=1.1):
        k_use = np.atleast_1d(k)
        M_use = np.atleast_1d(M)

        R_M = self.mass_def.get_radius(self.cosmo, M_use, scale_a) / scale_a # halo virial radius
        c_M = self.concentration(self.cosmo, M_use, scale_a) # concentration-mass relation c(M)
        r_s = R_M / c_M # characteristic scale r_s

        if self._func_normQ0 is None:
            with UnlockInstance(self):
                self._func_normQ0 = self._norm_interpol1() 
        if self._func_normQany is None:
            with UnlockInstance(self):
                self._func_normQany = self._norm_interpol2()

        q_use = k_use[None, :]*r_s[:, None]
        g_k = self._func_normQany((Gamma, np.log(q_use))) / self._func_normQ0(Gamma) # = Ib_qAny / Ib_q0

        prefix = M_use / scale_a**3
        prof = prefix[:, None] * g_k[None,:] 

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)

        return prof
    

class CombinedGasProfile(ccl.halos.profiles.profile_base.HaloProfile): 
    """ Combined profile of ejected & bound gas, assuming $f_{bd} + f_{ej} = 1$.

    
    """

    def __init__(self, cosmo, mass_def, concentration, fourier_analytic = True, gammaRange = (3, 20), ngamma=64, qrange=(1e-4, 1e2), nq=64):
        super(CombinedGasProfile, self).__init__(mass_def=mass_def, concentration=concentration)
        self.boundProfile = BoundGasProfile(cosmo=cosmo, mass_def=mass_def, concentration=concentration)
        self.ejProfile = EjectedGasProfile(cosmo=cosmo, mass_def=mass_def)
        self.fourier_analytic = fourier_analytic
        if fourier_analytic is True:
            self._fourier = self._fourier_analytic

        self.gammaRange = gammaRange
        self.ngamma = ngamma
        self.qrange = qrange
        self.nq = nq
        self._func_normQ0 = None   # General normalised bound profile (for q=0, over Gamma)
        self._func_normQany = None

    def _real(self, cosmo, r, M, scale_a=1, Gamma=1.1, f_bd=1, call_interp=True):
        f_ej = 1 - f_bd
        prof_ej = self.ejProfile._real(cosmo, r, M, scale_a) 
        prof_bd = self.boundProfile._real(cosmo, r, M, scale_a, Gamma, call_interp) 
        profile = f_ej*prof_ej + f_bd*prof_bd
        return profile

    def _fourier_analytic(self, k, M, scale_a=1, Gamma, f_bd = 1):
        f_ej = 1 - f_bd
        prof_ej = self.ejProfile._fourier(k, M, scale_a)
        prof_bd = self.boundProfile._fourier(k, M, scale_a, Gamma)
        profile = f_ej*prof_ej + f_bd*prof_bd[0]
        return profile

class CombinedStellarGasProfile(ccl.halos.profiles.profile_base.HaloProfile): 
    """Combined profile for the stellar & ejected & bound gas components, with the assumption that $f_c = 0$ & $f_{ej} + f_* + f_{bd} = 1$.      (even though $\bar{f}_b does not actually = 1).

    
    """

    def __init__(self, cosmo, mass_def, concentration, fourier_analytic = True, gammaRange = (3, 20), ngamma=64, qrange=(1e-4, 1e2), nq=64, beta=0.6, M_c = 10**(13.5), M_star = 10**(12.5), A_star = 0.03, sigma_star = 1.2):
        super(CombinedStellarGasProfile, self).__init__(mass_def=mass_def, concentration=concentration)
        self.boundProfile = BoundGasProfile(cosmo=cosmo, mass_def=mass_def, concentration=concentration)
        self.ejProfile = EjectedGasProfile(cosmo=cosmo, mass_def=mass_def)
        self.stellProfile = StellarProfile(cosmo=cosmo, mass_def=mass_def)
        self.fourier_analytic = fourier_analytic
        if fourier_analytic is True:
            self._fourier = self._fourier_analytic

        self.gammaRange = gammaRange
        self.ngamma = ngamma
        self.qrange = qrange
        self.nq = nq
        self._func_normQ0 = None   # General normalised bound profile (for q=0, over Gamma)
        self._func_normQany = None

        self.cosmo = cosmo
        self.beta = beta
        self.M_c = M_c
        self.M_star = M_star
        self.A_star = A_star
        self.sigma_star = sigma_star
        self.f_bar_b = self.cosmo['Omega_b']/self.cosmo['Omega_m']
        
    def _f_stell(self, M):
        f_stell = self.A_star * np.exp( (-1/2)* (np.log10(M / self.M_star) / self.sigma_star)**2 )
        return f_stell

    def _f_bd(self, M):     #$f_b(M)\ = \frac{\bar{f}_b - f_*(M)}{1 + (M_c/M)^{\beta}} $
        f_stell = self._f_stell(M)
        f_b = (self.f_bar_b - f_stell) / (1 + (self.M_c / M)**self.beta )
        return f_b, f_stell

    def _real(self, cosmo, r, M, scale_a=1, Gamma=1.1, call_interp=True, centre_pt=None):
        f_bd, f_stell = self._f_bd(M)
        f_ej = 1 - f_stell - f_bd  
        
        prof_ej = self.ejProfile._real(cosmo, r, M, scale_a) 
        prof_bd = self.boundProfile._real(cosmo, r, M, scale_a, Gamma, call_interp)
        prof_stell = self.stellProfile._real(cosmo, r, M, scale_a, centre_pt) 

        if np.shape(M) == ():
            prof_array = f_ej*prof_ej + f_bd*prof_bd + f_stell*prof_stell
        else:
            prof_array = np.zeros(len(M), dtype=object)
            i = 0
            for e, b, s in zip(f_ej, f_bd, f_stell): # should be same as: for mass in M
                profile = e*prof_ej[i] + b*prof_bd[i] + s*prof_stell[i]
                prof_array[i] = profile
                i+=1
        return prof_array 

    def _fourier_analytic(self, k, M, scale_a=1, Gamma=1.1, call_interp=True, centre_pt=None):
        f_bd, f_stell = self._f_bd(M)
        f_ej = 1 - f_stell - f_bd  
        
        prof_ej = self.ejProfile._fourier(k, M, scale_a)
        prof_bd = self.boundProfile._fourier(k, M, scale_a, Gamma)
        prof_stell = self.stellProfile._fourier(k, M, scale_a) 

        if np.shape(M) == ():
            prof_array = f_ej*prof_ej + f_bd*prof_bd + f_stell*prof_stell
        else:
            prof_array = np.zeros(len(M), dtype=object)
            i = 0
            for e, b, s in zip(f_ej, f_bd, f_stell): 
                profile = e*prof_ej[i] 
                profile += b*prof_bd[0,i] 
                profile += s*prof_stell[i]
                prof_array[i] = profile
                i+=1
        return prof_array

class CombinedAllBCMProfile(ccl.halos.profiles.profile_base.HaloProfile): 
    """Combined profile for the stellar & ejected & bound gas & cdm components (ie- The BCM Model), with the truncated Navarro-Frenk-White (NFW) profile used to calculate the density profiles of the cold dark matter (cdm) component.

    $f_c + f_b + f_e + f_* = 1$ 
    & (assuming adiabaticity) $f_b + f_e + f_* = \bar{f}_b \equiv \frac{\Omega_b}{\Omega_M}$

    For cold dark matter: $f_c\ = 1 - \bar{f}_b$.
    For the stellar component: $f_*(M)\ = A_*\ \exp{\left[ -\frac{1}{2} \left( \frac{\log_{10}(M/M_*)}{\sigma_*} \right)^2 \right]}$,
    with default parameters of: $A_* = 0.03$, $M_* = 10^{12.5}M_{\odot} $, & $\sigma_* = 1.2$.   
    For the bound gas: $f_b(M)\ = \frac{\bar{f}_b - f_*(M)}{1 + (M_c/M)^{\beta}} $,      
    with default parameter of: $M_c \simeq 10^{13.5 - 14} M_{\odot}$ & $\beta \sim 0.6$. 
    For the ejected gas: $f_e(M)\ = \bar{f}_b\ - f_b(M)\ - f_*(M)\ $.
    
    """

    def __init__(self, cosmo, mass_def, concentration, fourier_analytic = True, gammaRange = (3, 20), ngamma=64, qrange=(1e-4, 1e2), nq=64, beta=0.6, M_c = 10**(13.5), M_star = 10**(12.5), A_star = 0.03, sigma_star = 1.2):
        super(CombinedAllBCMProfile, self).__init__(mass_def=mass_def, concentration=concentration)
        self.boundProfile = BoundGasProfile(cosmo=cosmo, mass_def=mass_def, concentration=concentration)
        self.ejProfile = EjectedGasProfile(cosmo=cosmo, mass_def=mass_def)
        self.stellProfile = StellarProfile(cosmo=cosmo, mass_def=mass_def)
        self.cdmProfile = ccl.halos.profiles.nfw.HaloProfileNFW(mass_def=mass_def, concentration=concentration)
        self.fourier_analytic = fourier_analytic
        if fourier_analytic is True:
            self._fourier = self._fourier_analytic

        self.gammaRange = gammaRange
        self.ngamma = ngamma
        self.qrange = qrange
        self.nq = nq
        self._func_normQ0 = None   # General normalised bound profile (for q=0, over Gamma)
        self._func_normQany = None

        self.cosmo = cosmo
        self.beta = beta
        self.M_c = M_c
        self.M_star = M_star
        self.A_star = A_star
        self.sigma_star = sigma_star
        
        self.f_bar_b = self.cosmo['Omega_b']/self.cosmo['Omega_m']
        self.f_c = 1 - self.f_bar_b
        
    def _f_stell(self, M):
        f_stell = self.A_star * np.exp( (-1/2)* (np.log10(M / self.M_star) / self.sigma_star)**2 )
        return f_stell

    def _f_bd(self, M):     #$f_b(M)\ = \frac{\bar{f}_b - f_*(M)}{1 + (M_c/M)^{\beta}} $
        f_stell = self._f_stell(M)
        f_b = (self.f_bar_b - f_stell) / (1 + (self.M_c / M)**self.beta )
        return f_b, f_stell

    def _real(self, cosmo, r, M, scale_a=1, Gamma=1.1, call_interp=True, centre_pt=None):
        f_bd, f_stell = self._f_bd(M)
        f_ej = self.f_bar_b - f_stell - f_bd
        
        prof_ej = self.ejProfile._real(cosmo, r, M, scale_a) 
        prof_bd = self.boundProfile._real(cosmo, r, M, scale_a, Gamma, call_interp)
        prof_stell = self.stellProfile._real(cosmo, r, M, scale_a, centre_pt)
        prof_cdm = self.cdmProfile._real(self.cosmo, r, M, scale_a) 

        if np.shape(M) == ():
            prof_array = f_ej*prof_ej + f_bd*prof_bd + f_stell*prof_stell + self.f_c*prof_cdm 
        else:
            prof_array = np.zeros(len(M), dtype=object)
            i = 0
            for e, b, s in zip(f_ej, f_bd, f_stell): # should be same as: for mass in M
                profile = e*prof_ej[i] + b*prof_bd[i] + s*prof_stell[i] + self.f_c*prof_cdm[i] 
                prof_array[i] = profile
                i+=1
        return prof_array

    def _fourier_analytic(self, k, M, scale_a=1, Gamma):
        f_bd, f_stell = self._f_bd(M)
        f_ej = self.f_bar_b - f_stell - f_bd
        
        prof_ej = self.ejProfile._fourier(k, M, scale_a)
        prof_bd = self.boundProfile._fourier(k, M, scale_a, Gamma)
        prof_stell = self.stellProfile._fourier(k, M, scale_a)  
        prof_cdm = self.cdmProfile._fourier(self.cosmo, k, M, scale_a) 

        if np.shape(M) == ():
            prof_array = f_ej*prof_ej + f_bd*prof_bd + f_stell*prof_stell + self.f_c*prof_cdm 
        else:
            prof_array = np.zeros(len(M), dtype=object)
            i = 0
            for e, b, s in zip(f_ej, f_bd, f_stell): # should be same as: for mass in M
                profile = e*prof_ej[i] + b*prof_bd[0,i] + s*prof_stell[i] + self.f_c*prof_cdm[i]
                prof_array[i] = profile
                i+=1
        return prof_array