
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

    # upon change of the type attribute set the link to the material model
    def link_omega_to_mats(self):
        self.omega_fn_.trait_set(mats=self,
                                 E_name='E',
                                 x_max_name='eps_max')

    D_alg = Float(0)
    r'''Selector of the stiffness calculation.
    '''

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
        *MATS2DEval.ipw_view.content,
        Item('strain_norm'),
        Item('omega_fn'),
        Item('stress_state'),
        Item('D_alg', latex=r'\theta_\mathrm{alg. stiff.}',
                editor=FloatRangeEditor(low=0,high=1)),
    )

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
