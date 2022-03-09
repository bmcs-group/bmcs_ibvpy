
import copy

from ibvpy.tmodel.mats_eval import \
    MATSEval
from traits.api import \
    Constant, Property, cached_property, Array,\
    Dict, Str, Callable

from bmcs_utils.api import Float, Int
import numpy as np


class MATSXDEval(MATSEval):
    '''Base class for elastic models.
    '''

    n_dims = Constant(Float)
    '''Number of spatial dimensions of an integration 
    cell for the material model
    '''

    E = Float(34e+3,
              label="E",
              desc="Young's Modulus",
              MAT=True)

    nu = Float(0.2,
               label='nu',
               desc="Poison's ratio",
               MAT=True)

    eps_max = Float(0.03, ALG=True)

    n_eps = Int(100, ALG=True)

    def subplots(self, fig):
        ax_sig = fig.subplots(1,1)
        ax_d_sig = ax_sig.twinx()
        return ax_sig, ax_d_sig

    def update_plot(self, axes):
        ax_sig, ax_d_sig = axes
        eps_max = self.eps_max
        n_eps = 100
        eps11_range = np.linspace(1e-9,eps_max,n_eps)
        eps_range = np.zeros((n_eps, self.n_dims, self.n_dims))
        eps_range[:,0,0] = eps11_range
        state_vars = { var : np.zeros( (n_eps,) + shape )
            for var, shape in self.state_var_shapes.items()
        }
        sig_range, D_range = self.get_corr_pred(eps_range, 1, **state_vars)
        sig11_range = sig_range[:,0,0]
        ax_sig.plot(eps11_range, sig11_range,color='blue')
        d_sig1111_range = D_range[...,0,0,0,0]
        ax_d_sig.plot(eps11_range, d_sig1111_range, linestyle='dashed', color='gray')
        ax_sig.set_xlabel(r'$\varepsilon_{11}$ [-]')
        ax_sig.set_ylabel(r'$\sigma_{11}$ [MPa]')
        ax_d_sig.set_ylabel(r'$\mathrm{d} \sigma_{11} / \mathrm{d} \varepsilon_{11}$ [MPa]')
        ax_d_sig.plot(eps11_range[:-1],
                    (sig11_range[:-1]-sig11_range[1:])/(eps11_range[:-1]-eps11_range[1:]),
                    color='orange', linestyle='dashed')


    def _get_lame_params(self):
        # First Lame parameter (bulk modulus)
        la = self.E * self.nu / ((1. + self.nu) * (1. - 2. * self.nu))
        # second Lame parameter (shear modulus)
        mu = self.E / (2. + 2. * self.nu)
        return la, mu

    D_abef = Property(Array, depends_on='E, nu')
    '''Material stiffness - rank 4 tensor
    '''
    @cached_property
    def _get_D_abef(self):
        la, mu = self._get_lame_params()
        delta = np.identity(self.n_dims)
        return (
            np.einsum(',ij,kl->ijkl', la, delta, delta) +
            np.einsum(',ik,jl->ijkl', mu, delta, delta) +
            np.einsum(',il,jk->ijkl', mu, delta, delta)
        )

    def get_corr_pred(self, eps_ab, tn1, **Eps):
        '''
        Corrector predictor computation.
        @param eps_Emab input variable - strain tensor
        '''
        sigma_ab = np.einsum(
            'abcd,...cd->...ab', self.D_abef, eps_ab
        )
        Em_len = len(eps_ab.shape) - 2
        new_shape = tuple([1 for _ in range(Em_len)]) + self.D_abef.shape
        D_abef = self.D_abef.reshape(*new_shape)
        return sigma_ab, D_abef

    #=========================================================================
    # Response variables
    #=========================================================================
    def get_eps_ab(self, eps_ab, tn1, **Eps):
        return eps_ab

    def get_sig_ab(self, eps_ab, tn1, **Eps):
        '''
        Get the stress tensor
        :param eps_ab: strain tensor with ab indexes
        :param tn1: time - can be used for time dependent properties
        :param Eps: dictionary of state variables corresponding to the
         specification in state_var_shapes
        :return: stress vector with with the shape of input and state variables + ab
         spatial indexes.
        '''
        return self.get_sig(eps_ab, tn1, **Eps)

    def get_state_var(self, var_name, eps_ab, tn1, **Eps):
        '''
        Acess to state variables used for visualization
        :param var_name: string with the name of the state variable
        :param eps_ab: strain tensor (not used)
        :param tn1: time variable (not used)
        :param Eps: dictionary of state variables corresponding to the
         specification in state_var_shapes
        :return: Array with the state variable with the shape corresponding
         to the level of the problem
        '''
        return Eps[var_name]

    var_dict = Property(Dict(Str, Callable))
    '''Dictionary of functions that deliver the response/state variables 
    of a material model
    '''
    @cached_property
    def _get_var_dict(self):
        var_dict = super(MATSXDEval, self)._get_var_dict()
        var_dict.update(eps_ab=self.get_eps_ab,
                        sig_ab=self.get_sig_ab)
        var_dict.update({
            var_name : lambda eps_ab, tn1, **Eps : self.get_state_var(
                var_name, eps_ab, tn1, **Eps
            )
            for var_name in self.state_var_shapes.keys()
        })
        return var_dict
