__all__ = ("CombinerClass")

class CombinerClass(ccl.halos.profiles.profile_base.HaloProfile):

    def __init__(self, prof_list):
        # prof_list = [p1, p2, p3, ...] // = [stel_prof, cdm_prof, gas_prof, ej_prof]
        self.prof_list = prof_list
        ####
    def divide_param

    def update_parameters(self, **kwargs):

        params = # self.divide_params(**kwargs)
        # what function is this, in Inspect ?

        for prof, pars in zip(self.prof_list, params)
            prof.update_parameters(**pars)


    def _real(self, cosmo, r, M, a): # scale_a = 1
        real_list = [prof._real(cosmo, r, M, a) for prof in self.prof_list]
        # apply np.array or np.atleast_1d for this
        profile = np.sum(np.atleast_1d(real_list))
    return profile
    

    def _fourier(self, cosmo, k, M, a): # scale_a = 1
        # use ._fourier OR .fourier (not all have options other than inbuilt)
        # could add in to the component profiles that, if fourier_analytic (etc) is called, but there is none, Then: self._fourier = self.fourier (so call CCL)
         fourier_list = [prof._fourier(cosmo, k, M, a) for prof in self.prof_list]
        # apply np.array or np.atleast_1d for this
        profile = np.sum(np.atleast_1d(fourier_list))
    return profile



class CombinerClassDavid(ccl.halos.profiles.profile_base.HaloProfile):
    def __init__(self, prof_list):
        self.prof_list = prof_list
        self.param_list = []
        for prof in self.prof_list:
            arglist = list(prof.update_parameters.__code__.co_varnames)
            if 'self' in arglist:
                arglist.remove('self')
            self.param_list.append(arglist)

    def update_parameters(self, **kwargs):
        for prof, param_names in zip(self.prof_list, self.param_list):
            param_dict = {}
            for name in param_names:
                if name in kwargs:
                    param_dict[name] = kwargs[name]
            prof.update_parameters(**param_dict)

    def _real(self, cosmo, r, M, a):
        pass

    def _fourier(self, cosmo, k, M, a):
        pass