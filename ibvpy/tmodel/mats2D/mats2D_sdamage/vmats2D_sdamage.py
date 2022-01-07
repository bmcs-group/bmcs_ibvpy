
from ibvpy.tmodel.mats2D.mats2D_eval import MATS2DEval
from bmcs_utils.api import Float, Item, View, Enum, EitherType, FloatRangeEditor
import numpy as np
import traits.api as tr
from ibvpy.tmodel.mats_damage_fn import \
    IDamageFn, GfDamageFn, ExpSlopeDamageFn, AbaqusDamageFn, \
    LinearDamageFn, FRPDamageFn, WeibullDamageFn

from .vstrain_norm2d import StrainNorm2D, SN2DRankine, SN2DMasars, SN2DEnergy

class MATS2DScalarDamage(MATS2DEval):
    r'''Isotropic damage model.
    '''

    name = 'isotropic damage model'
    node_name = 'isotropic damage model'

    tree = ['omega_fn','strain_norm']

    omega_fn = EitherType(
        options=[('exp-slope', ExpSlopeDamageFn),
                 ('linear', LinearDamageFn),
                 ('abaqus', AbaqusDamageFn),
                 ('fracture-energy', GfDamageFn),
                 ('weibull-CDF', WeibullDamageFn),
                 ],
        MAT=True,
        on_option_change='link_omega_to_mats'
    )

    D_alg = Float(0)
    r'''Selector of the stiffness calculation.
    '''

    eps_max = Float(0.03, ALG=True)
    # upon change of the type attribute set the link to the material model
    def link_omega_to_mats(self):
        self.omega_fn_.trait_set(mats=self,
                                 E_name='E',
                                 x_max_name='eps_max')

    #=========================================================================
    # Material model
    #=========================================================================
    strain_norm = EitherType(
        options=[('Rankine', SN2DRankine),
                 ('Masars', SN2DMasars),
                 ('Energy', SN2DEnergy)],
        on_option_change='link_strain_norm_to_mats'
    )

    def link_strain_norm_to_mats(self):
        self.strain_norm_.trait_set(mats=self)

    state_var_shapes = {'kappa': (),
                        'omega': ()}
    r'''
    Shapes of the state variables
    to be stored in the global array at the level 
    of the domain.
    '''

    def get_corr_pred(self, eps_ab_n1, tn1, kappa, omega):
        r'''
        Corrector predictor computation.
        @param eps_app_eng input variable - engineering strain
        '''
        eps_eq = self.strain_norm_.get_eps_eq(eps_ab_n1, kappa)
        I = self.omega_fn_.get_f_trial(eps_eq, kappa)
        eps_eq_I = eps_eq[I]
        kappa[I] = eps_eq_I
        omega[I] = self._omega(eps_eq_I)
        phi = (1.0 - omega)
        D_abcd = np.einsum(
            '...,abcd->...abcd',
            phi, self.D_abcd
        )
        sig_ab = np.einsum(
            '...abcd,...cd->...ab',
            D_abcd, eps_ab_n1
        )
        if self.D_alg > 0:
            domega_ds_I = self._omega_derivative(eps_eq_I)
            deps_eq_I = self.strain_norm_.get_deps_eq(eps_ab_n1[I])
            D_red_I = np.einsum('...,...ef,cdef,...ef->...cdef', domega_ds_I,
                                deps_eq_I, self.D_abcd, eps_ab_n1[I]) * self.D_alg
            D_abcd[I] -= D_red_I

        return sig_ab, D_abcd

    def _omega(self, kappa):
        return self.omega_fn_(kappa)

    def _omega_derivative(self, kappa):
        return self.omega_fn_.diff(kappa)

    ipw_view = View(
        Item('E'),
        Item('nu'),
        Item('strain_norm'),
        Item('omega_fn'),
        Item('stress_state'),
        Item('D_alg', latex=r'\theta_\mathrm{alg. stiff.}',
                editor=FloatRangeEditor(low=0,high=1)),
        Item('eps_max'),
        Item('G_f', latex=r'G_\mathrm{f}^{\mathrm{estimate}}', readonly=True),
    )

    G_f = tr.Property(Float, depends_on='state_changed')
    @tr.cached_property
    def _get_G_f(self):
        eps_max = self.eps_max
        n_eps = 1000
        eps11_range = np.linspace(1e-9,eps_max,n_eps)
        eps_range = np.zeros((len(eps11_range), 2, 2))
        eps_range[:,1,1] = eps11_range
        state_vars = { var : np.zeros( (len(eps11_range),) + shape )
            for var, shape in self.state_var_shapes.items()
        }
        sig_range, D = self.get_corr_pred(eps_range, 1, **state_vars)
        sig11_range = sig_range[:,1,1]
        return np.trapz(sig11_range, eps11_range)

    def subplots(self, fig):
        ax_sig = fig.subplots(1,1)
        ax_d_sig = ax_sig.twinx()
        return ax_sig, ax_d_sig

    def update_plot(self, axes):
        ax_sig, ax_d_sig = axes
        eps_max = self.eps_max
        n_eps = 100
        eps11_range = np.linspace(1e-9,eps_max,n_eps)
        eps_range = np.zeros((n_eps, 2, 2))
        eps_range[:,0,0] = eps11_range
        state_vars = { var : np.zeros( (n_eps,) + shape )
            for var, shape in self.state_var_shapes.items()
        }
        sig_range, D_range = self.get_corr_pred(eps_range, 1, **state_vars)
        sig11_range = sig_range[:,0,0]
        ax_sig.plot(eps11_range, sig11_range,color='blue')
        d_sig1111_range = D_range[...,0,0,0,0]
        ax_d_sig.plot(eps11_range, d_sig1111_range,
                      linestyle='dashed', color='gray')
        ax_sig.set_xlabel(r'$\varepsilon_{11}$ [-]')
        ax_sig.set_ylabel(r'$\sigma_{11}$ [MPa]')
        ax_d_sig.set_ylabel(r'$\mathrm{d} \sigma_{11} / \mathrm{d} \varepsilon_{11}$ [MPa]')

        ax_d_sig.plot(eps11_range[:-1],
                    (sig11_range[:-1]-sig11_range[1:])/(eps11_range[:-1]-eps11_range[1:]),
                    color='orange', linestyle='dashed')

    def get_omega(self, eps_ab, tn1, **Eps):
        return Eps['omega']

    var_dict = tr.Property(tr.Dict(tr.Str, tr.Callable))
    '''Dictionary of response variables
    '''
    @tr.cached_property
    def _get_var_dict(self):
        var_dict = dict(omega=self.get_omega)
        var_dict.update(super()._get_var_dict())
        return var_dict
