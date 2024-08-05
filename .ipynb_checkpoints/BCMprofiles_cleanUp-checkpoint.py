__all__ = ("StellarProfile", "EjectedGasProfile", "BoundGasProfile", "CDMProfile" , "BCMProfile", "CombinedAllBCMProfile")

import numpy as np
import pyccl as ccl
import matplotlib.pyplot as plt
from scipy import signal
import scipy.integrate as integrate
import scipy.interpolate as interpol
from pyccl._core import UnlockInstance

class CDMProfile(ccl.halos.profiles.nfw.HaloProfileNFW): 
    """Density profile for the cold dark matter (cdm), using the Navarro-Frenk-White, multiplied by the cdm's mass fraction.
    
    """

    def __init__(self, cosmo, mass_def, concentration, fourier_analytic=True, projected_analytic=False, cumul2d_analytic=False, truncated=True):
        
        super(CDMProfile, self).__init__(mass_def=mass_def, concentration=concentration, fourier_analytic=fourier_analytic, projected_analytic=projected_analytic, cumul2d_analytic=cumul2d_analytic, truncated=truncated)
        self.cdmProfile = ccl.halos.profiles.nfw.HaloProfileNFW(mass_def=mass_def, concentration=concentration, fourier_analytic=fourier_analytic, projected_analytic=projected_analytic, cumul2d_analytic=cumul2d_analytic, truncated=truncated)
            
        self.cosmo = cosmo
        self.f_bar_b = self.cosmo['Omega_b']/self.cosmo['Omega_m']
        self.f_c = 1 - self.f_bar_b

    def update_parameters(self, cosmo=None, mass_def=None, concentration=None, fourier_analytic=None, projected_analytic=None, cumul2d_analytic=None, truncated=None):
        """Update any of the parameters associated with this profile.
        Any parameter set to ``None`` won't be updated.
        """
        if mass_def is not None:
            self.mass_def = mass_def
        if concentration is not None:
            self.concentration = concentration
        if fourier_analytic is not None and fourier_analytic is True:                   
            self._fourier = self._fourier_analytic

        # COULD ADD IN THESE FOR UPDATE PARS, BUT instead of self. have ccl.halos.profiles.nfw.HaloProfileNFW.
        
 #       if projected_analytic is not None and projected_analytic is True and truncated is False:
  #          self._projected = self._projected_analytic
  #      if cumul2d_analytic is not None and cumul2d_analytic is True and truncated is False:
   #         self._cumul2d = self._cumul2d_analytic
  #      if truncated is not None and if truncated is True:
   #         self.truncated = truncated

        if cosmo is not None and cosmo != self.cosmo:
            self.cosmo = cosmo
            self.f_bar_b = self.cosmo['Omega_b']/self.cosmo['Omega_m']
            self.f_c = 1 - self.f_bar_b

    def _real(self, cosmo, r, M, scale_a=1):
        prof = self.f_c * self.cdmProfile._real(self.cosmo, r, M, scale_a) 
        # ccl.halos.profiles.nfw.HaloProfileNFW._real(cosmo=self.cosmo, r=r, M=M, a=scale_a)  
                          # * self.cdmProfile._real(self.cosmo, r, M, scale_a) 
        return prof

    def _fourier_analytic(self, k, M, scale_a=1):
        prof = self.f_c * self.cdmProfile._fourier(self.cosmo, k, M, scale_a) 
        #ccl.halos.profiles.nfw.HaloProfileNFW._fourier(cosmo=self.cosmo, k=k, M=M, a=scale_a)  
                          # * self.cdmProfile._fourier(self.cosmo, k, M, scale_a) 
        return prof

class StellarProfile(ccl.halos.profiles.profile_base.HaloProfile): 
    """Creating a class for the stellar density profile
    where: $\\rho_*(r)\ = Ma ^{-3} g_*(r)\ $ & $g_*(r)\ \\equiv \\delta^D$(**x**) (a Dirac delta funciton centred at $r=0$). 
    The normalised Fourier profile is then given by: $\\tilde{g}_*(k)\ = 1$.
    
    """  

    def __init__(self, cosmo, mass_def, fourier_analytic=True, M_star = 10**(12.5), A_star = 0.03, sigma_star = 1.2):
        super(StellarProfile, self).__init__(mass_def=mass_def)
        self.cosmo = cosmo
        self.fourier_analytic = fourier_analytic
        if fourier_analytic is True:
            self._fourier = self._fourier_analytic

        self.M_star = M_star
        self.A_star = A_star
        self.sigma_star = sigma_star
        
    def _f_stell(self, M):
        f_stell = self.A_star * np.exp( (-1/2)* (np.log10(M / self.M_star) / self.sigma_star)**2 )
        return f_stell

    def update_parameters(self, cosmo=None, mass_def=None, fourier_analytic=None, M_star=None, A_star=None, sigma_star=None):
        """Update any of the parameters associated with this profile.
        Any parameter set to ``None`` won't be updated.
        """
        if cosmo is not None:
            self.cosmo = cosmo
        if mass_def is not None:
            self.mass_def = mass_def
        if fourier_analytic is not None and fourier_analytic is True:                  
            self._fourier = self._fourier_analytic

        if M_star is not None:
            self.M_star = M_star
        if A_star is not None:
            self.A_star = A_star
        if sigma_star is not None:
            self.sigma_star = sigma_star
########
            
    def _real(self, cosmo, r, M, scale_a=1, centre_pt=None): 
        # want delta centred at r=0 (& since log scale, can't do negative or zero values in array)
        r_use = np.atleast_1d(r) 
        M_use = np.atleast_1d(M)
        len_r = len(r_use) 

        f_stell = self._f_stell(M_use)
        prefix = f_stell * M_use / scale_a**3
        prof = prefix[:, None] * signal.unit_impulse(len_r, centre_pt)[None,:] # If centre_pt=None, defaults to index at the 0th element.

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
                                                          
        return prof

    def _fourier_analytic(self, k, M, scale_a=1):
        k_use = np.atleast_1d(k)
        M_use = np.atleast_1d(M)

        f_stell = self._f_stell(M_use)
        prefix = f_stell * M_use / scale_a**3
        prof = np.ones_like(k_use)[None,:] * prefix[:, None] # k_use[None,:] + prefix[:, None] * 1 # as g(k) = 1

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)

        return prof

class EjectedGasProfile(ccl.halos.profiles.profile_base.HaloProfile): 
    """Creating a class for the ejected gas density profile
    where: """  

    def __init__(self, cosmo, mass_def, fourier_analytic = True, beta=0.6, M_c = 10**(13.5), M_star = 10**(12.5), A_star = 0.03, sigma_star = 1.2): 
        super(EjectedGasProfile, self).__init__(mass_def=mass_def)
        self.fourier_analytic = fourier_analytic
        if fourier_analytic is not None and True:
            self._fourier = self._fourier_analytic

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

    def _f_bd(self, M):     
        f_stell = self._f_stell(M)
        f_b = (self.f_bar_b - f_stell) / (1 + (self.M_c / M)**self.beta )
        return f_b, f_stell

    def update_parameters(self, cosmo=None, mass_def=None, fourier_analytic=None, beta=None, M_c=None, M_star=None, A_star=None, sigma_star=None):
        """Update any of the parameters associated with this profile.
        Any parameter set to ``None`` won't be updated.
        """
        if mass_def is not None:
            self.mass_def = mass_def
        if fourier_analytic is not None and fourier_analytic is True:                  
            self._fourier = self._fourier_analytic
        
        if beta is not None:
            self.beta = beta
        if M_c is not None:
            self.M_c = M_c
        if M_star is not None:
            self.M_star = M_star
        if A_star is not None:
            self.A_star = A_star
        if sigma_star is not None:
            self.sigma_star = sigma_star

        if cosmo is not None and cosmo != self.cosmo:
            self.cosmo = cosmo
            self.f_bar_b = self.cosmo['Omega_b']/self.cosmo['Omega_m']
            self.f_c = 1 - self.f_bar_b

########
            
    def _real(self, cosmo, r, M, scale_a=1, delta=200, eta_b = 0.5): 
        r_use = np.atleast_1d(r) 
        M_use = np.atleast_1d(M)
        r_vir = self.mass_def.get_radius(self.cosmo, M_use, scale_a) / scale_a    # halo virial radius
        r_e = 0.375*r_vir*np.sqrt(delta)*eta_b                                    # eta_b = a free parameter

        f_bd, f_stell = self._f_bd(M)
        f_ej = self.f_bar_b - f_stell - f_bd
        prefix = f_ej * M_use / (scale_a*r_e*np.sqrt(2*np.pi))**3  
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
        r_vir = self.mass_def.get_radius(self.cosmo, M_use, scale_a) / scale_a    # halo virial radius
        r_e = 0.375*r_vir*np.sqrt(delta)*eta_b                                    # eta_b = a free parameter

        f_bd, f_stell = self._f_bd(M)
        f_ej = self.f_bar_b - f_stell - f_bd
        prefix = f_ej * M_use / scale_a**3
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

    def __init__(self, cosmo, mass_def, concentration, Gamma, fourier_analytic = True, gammaRange = (3, 20), ngamma=64, qrange=(1e-4, 1e2), nq=64, limInt=(1E-3, 5E3), beta=0.6, M_c = 10**(13.5), M_star = 10**(12.5), A_star = 0.03, sigma_star = 1.2): 
        super(BoundGasProfile, self).__init__(mass_def=mass_def, concentration=concentration)
        self.Gamma = Gamma
        
        self.fourier_analytic = fourier_analytic
        if fourier_analytic is not None and True:
            self._fourier = self._fourier_analytic
            
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
        
    def _f_stell(self, M):
        f_stell = self.A_star * np.exp( (-1/2)* (np.log10(M / self.M_star) / self.sigma_star)**2 )
        return f_stell

    def _f_bd(self, M):     
        f_stell = self._f_stell(M)
        f_b = (self.f_bar_b - f_stell) / (1 + (self.M_c / M)**self.beta )
        return f_b, f_stell
    
    def update_parameters(self, cosmo=None, mass_def=None, Gamma=None, fourier_analytic=None, gammaRange=None, ngamma=None, qrange=None, nq=None, limInt=None, beta=None, M_c=None, M_star=None, A_star=None, sigma_star=None):
        """Update any of the parameters associated with this profile.
        Any parameter set to ``None`` won't be updated.
        """
        if mass_def is not None:
            self.mass_def = mass_def
        if fourier_analytic is not None and fourier_analytic is True: 
            self._fourier = self._fourier_analytic    
        if Gamma is None:
            self.Gamma = Gamma
        
        if beta is not None:
            self.beta = beta
        if M_c is not None:
            self.M_c = M_c
        if M_star is not None:
            self.M_star = M_star
        if A_star is not None:
            self.A_star = A_star
        if sigma_star is not None:
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
            self.limInt = nq
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

        if re_normQ0 and (self._func_normQ0 is not None):
            self._func_normQ0 = self._norm_interpol1() 
        if re_normQany and (self._func_normQany is not None):
            self._func_normQany = self._norm_interpol2()

########

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
        
    def _real(self, cosmo, r, M, scale_a=1, call_interp=True): 
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

        f_bd, f_stell = self._f_bd(M)
        prefix = f_bd * M_use * (1/scale_a**3) * (1/vB_prefix)

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
    
    def _fourier_analytic(self, k, M, scale_a=1):
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

        f_bd, f_stell = self._f_bd(M)
        prefix = f_bd * M_use / scale_a**3
        prof = prefix[:, None] * g_k[None,:] 

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)

        return prof

class BCMProfile(ccl.halos.profiles.profile_base.HaloProfile): 
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

    def __init__(self, cosmo, mass_def, concentration, Gamma, fourier_analytic = True, gammaRange = (3, 20), ngamma=64, qrange=(1e-4, 1e2), nq=64, limInt=(1E-3, 5E3), beta=0.6, M_c = 10**(13.5), M_star = 10**(12.5), A_star = 0.03, sigma_star = 1.2, projected_analytic=False, cumul2d_analytic=False, truncated=True):
        super(BCMProfile, self).__init__(mass_def=mass_def, concentration=concentration)
        self.boundProfile = BoundGasProfile(cosmo=cosmo, mass_def=mass_def, concentration=concentration, Gamma=Gamma, gammaRange=gammaRange, ngamma=ngamma, qrange=qrange, nq=nq, limInt=limInt, beta=beta, M_c=M_c, M_star=M_star, A_star=A_star, sigma_star=sigma_star)
        self.ejProfile = EjectedGasProfile(cosmo=cosmo, mass_def=mass_def, beta=beta, M_c=M_c, M_star=M_star, A_star=A_star, sigma_star=sigma_star)
        self.stellProfile = StellarProfile(cosmo=cosmo, mass_def=mass_def, M_star=M_star, A_star=A_star, sigma_star=sigma_star)
        # self.cdmProfile = ccl.halos.profiles.nfw.HaloProfileNFW(mass_def=mass_def, concentration=concentration)
        self.cdmProfile = CDMProfile(cosmo=cosmo, mass_def=mass_def, concentration=concentration, fourier_analytic=fourier_analytic, projected_analytic=projected_analytic, cumul2d_analytic=cumul2d_analytic, truncated=truncated)

        self.cosmo=cosmo
        # do I need these?
        self.fourier_analytic = fourier_analytic
        if fourier_analytic is True:
            self._fourier = self._fourier_analytic
            
        self._func_normQ0 = None   # General normalised bound profile (for q=0, over Gamma)
        self._func_normQany = None
        
    def _real(self, cosmo, r, M, scale_a=1, call_interp=True, centre_pt=None):

        # the mass fractions are now included in the individual profiles
        prof_ej = self.ejProfile._real(cosmo, r, M, scale_a) 
        prof_bd = self.boundProfile._real(cosmo, r, M, scale_a, call_interp)
        prof_stell = self.stellProfile._real(cosmo, r, M, scale_a, centre_pt)
        prof_cdm = self.cdmProfile._real(cosmo, r, M, scale_a) 

        if np.shape(M) == ():
            prof_array = prof_ej + prof_bd + prof_stell + prof_cdm 
        else:
            prof_array = np.zeros(len(M), dtype=object)
            i = 0
            for e, b, s, c in zip(prof_ej, prof_bd, prof_stell, prof_cdm): # should be same as: for mass in M
                profile = e + b + s + c
                prof_array[i] = profile
                i+=1
        return prof_array

    def _fourier_analytic(self, k, M, scale_a=1, delta=200, eta_b = 0.5):
        
        # the mass fractions are now included in the individual profiles
        prof_ej = self.ejProfile._fourier(k, M, scale_a, delta, eta_b)
        prof_bd = self.boundProfile._fourier(k, M, scale_a)
        prof_stell = self.stellProfile._fourier(k, M, scale_a)  
        prof_cdm = self.cdmProfile._fourier(k, M, scale_a) 

        if np.shape(M) == ():
            prof_array = prof_ej + prof_bd[0] + prof_stell + prof_cdm 
        else:
            prof_array = np.zeros(len(M), dtype=object)
            i = 0
            for e, b, s, c in zip(prof_ej, prof_bd[0], prof_stell, prof_cdm): # should be same as: for mass in M
                profile = e + b + s + c
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

    def __init__(self, cosmo, mass_def, concentration, Gamma, fourier_analytic = True, gammaRange = (3, 20), ngamma=64, qrange=(1e-4, 1e2), nq=64, limInt=(1E-3, 5E3), beta=0.6, M_c = 10**(13.5), M_star = 10**(12.5), A_star = 0.03, sigma_star = 1.2):
        super(CombinedAllBCMProfile, self).__init__(mass_def=mass_def, concentration=concentration, Gamma=Gamma)
        self.boundProfile = BoundGasProfile(cosmo=cosmo, mass_def=mass_def, concentration=concentration, Gamma=Gamma, gammaRange=gammaRange, ngamma=ngamma, qrange=qrange, nq=nq, limInt=limInt, beta=beta, M_c=M_c, M_star=M_star, A_star=A_star, sigma_star=sigma_star)
        self.ejProfile = EjectedGasProfile(cosmo=cosmo, mass_def=mass_def, beta=beta, M_c=M_c, M_star=M_star, A_star=A_star, sigma_star=sigma_star)
        self.stellProfile = StellarProfile(cosmo=cosmo, mass_def=mass_def, M_star=M_star, A_star=A_star, sigma_star=sigma_star)
        self.cdmProfile = ccl.halos.profiles.nfw.HaloProfileNFW(mass_def=mass_def, concentration=concentration)

        # do I need these?
        self.fourier_analytic = fourier_analytic
        if fourier_analytic is True:
            self._fourier = self._fourier_analytic
            
        self._func_normQ0 = None   # General normalised bound profile (for q=0, over Gamma)
        self._func_normQany = None
        
        self.f_bar_b = self.cosmo['Omega_b']/self.cosmo['Omega_m']
        self.f_c = 1 - self.f_bar_b
        
    def _f_stell(self, M):
        f_stell = self.A_star * np.exp( (-1/2)* (np.log10(M / self.M_star) / self.sigma_star)**2 )
        return f_stell

    def _f_bd(self, M):     #$f_b(M)\ = \frac{\bar{f}_b - f_*(M)}{1 + (M_c/M)^{\beta}} $
        f_stell = self._f_stell(M)
        f_b = (self.f_bar_b - f_stell) / (1 + (self.M_c / M)**self.beta )
        return f_b, f_stell

    def _real(self, cosmo, r, M, scale_a=1, call_interp=True, centre_pt=None):
        f_bd, f_stell = self._f_bd(M)
        f_ej = self.f_bar_b - f_stell - f_bd
        
        prof_ej = self.ejProfile._real(cosmo, r, M, scale_a) 
        prof_bd = self.boundProfile._real(cosmo, r, M, scale_a, call_interp)
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

    def _fourier_analytic(self, k, M, scale_a=1):
        f_bd, f_stell = self._f_bd(M)
        f_ej = self.f_bar_b - f_stell - f_bd
        
        prof_ej = self.ejProfile._fourier(k, M, scale_a)
        prof_bd = self.boundProfile._fourier(k, M, scale_a)
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