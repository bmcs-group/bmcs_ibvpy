
from math import sqrt as scalar_sqrt
from ibvpy.tmodel.mats2D.mats2D_eval import MATS2DEval
from numpy import \
    array, zeros, dot, \
    sqrt, vdot, \
    float_, diag
from scipy.linalg import inv, norm
from traits.api import \
    Array, Enum, \
    Int, Trait, \
    Dict, Property, cached_property
from bmcs_utils.api import \
    Item, View, Float
from ibvpy.util.traits.either_type import EitherType
from .yield_face2D import IYieldFace2D, J2, DruckerPrager, Gurson, CamClay

#---------------------------------------------------------------------------
# Material time-step-evaluator for Scalar-Damage-Model
#---------------------------------------------------------------------------
class MATS2DPlastic(MATS2DEval):
    '''
    Elastic Model.
    '''

    # implements(IMATSEval)

    #-------------------------------------------------------------------------
    # Parameters of the numerical algorithm (integration)
    #-------------------------------------------------------------------------

    stress_state = Enum("plane strain", "plane stress",)
    algorithm = Enum("closest point", "cutting plane")

    #-------------------------------------------------------------------------
    # Material parameters
    #-------------------------------------------------------------------------
    yf = EitherType(klasses=[J2, DruckerPrager, Gurson, CamClay],
                    label="Yield Face",
                    desc="Yield Face Definition"
                    )
    E = Float(210.0e+3,
              label="E",
              desc="Young's Modulus")
    nu = Float(0.2,
               label='nu',
               desc="Poison's ratio")
    K_bar = Float(0.,
                  label='K',
                  desc="isotropic softening parameter")
    H_bar = Float(0.,
                  label='H',
                  desc="kinematic softening parameter")
    tolerance = Float(1.0e-4,
                      label='TOL',
                      desc="tolerance of return mapping")

    max_iter = Int(20,
                   label='Iterations',
                   desc="maximal number of iterations")

    D_el = Property(Array(float), depends_on='E, nu')

    @cached_property
    def _get_D_el(self):
        if self.stress_state == "plane_stress":
            return self._get_D_plane_stress()
        else:
            return self._get_D_plane_strain()

    H_mtx = Property(Array(float), depends_on='K_bar, H_bar')

    @cached_property
    def _get_H_mtx(self):
        H_mtx = diag([self.K_bar, self.H_bar, self.H_bar, self.H_bar])
        return H_mtx

    # This event can be used by the clients to trigger an action upon
    # the completed reconfiguration of the material model
    #

    #-------------------------------------------------------------------------
    # View specification
    #-------------------------------------------------------------------------

    view_traits = View(
        Item('yf'),
        Item('E'),
        Item('nu'),
        Item('K_bar'),
        Item('H_bar'),
        Item('tolerance'),
        Item('max_iter'),
        Item('algorithm'),
    )

    #-------------------------------------------------------------------------
    # Evaluation - get the corrector and predictor
    #-------------------------------------------------------------------------

    def get_corr_pred(self, sctx, eps_app_eng, d_eps, tn, tn1):
        '''
        Corrector predictor computation.
        @param eps_app_eng input variable - engineering strain
        '''
        delta_gamma = 0.
        if sctx.update_state_on:
            # print "in us"
            eps_n = eps_app_eng - d_eps
            sigma, f_trial, epsilon_p, q_1, q_2 = self._get_state_variables(
                sctx, eps_n)

            sctx.mats_state_array[:3] = epsilon_p
            sctx.mats_state_array[3] = q_1
            sctx.mats_state_array[4:] = q_2

        diff1s = zeros([3])
        sigma, f_trial, epsilon_p, q_1, q_2 = self._get_state_variables(
            sctx, eps_app_eng)
        # Note: the state variables are not needed here, just gamma
        diff2ss = self.yf.get_diff2ss(eps_app_eng, self.E, self.nu, sctx)
        Xi_mtx = inv(inv(self.D_el) + delta_gamma * diff2ss * f_trial)
        N_mtx_denom = sqrt(dot(dot(diff1s, Xi_mtx), diff1s))
        if N_mtx_denom == 0.:
            N_mtx = zeros(3)
        else:
            N_mtx = dot(Xi_mtx, self.diff1s) / N_mtx_denom
        D_mtx = Xi_mtx - vdot(N_mtx, N_mtx)

        # print "sigma ",sigma
        # print "D_mtx ",D_mtx
        return sigma, D_mtx


    def _get_D_plane_stress(self):
        E = self.E
        nu = self.nu
        D_stress = zeros([3, 3])
        D_stress[0][0] = E / (1.0 - nu * nu)
        D_stress[0][1] = E / (1.0 - nu * nu) * nu
        D_stress[1][0] = E / (1.0 - nu * nu) * nu
        D_stress[1][1] = E / (1.0 - nu * nu)
        D_stress[2][2] = E / (1.0 - nu * nu) * (1.0 / 2.0 - nu / 2.0)
        return D_stress

    def _get_D_plane_strain(self):
        E = self.E
        nu = self.nu
        D_strain = zeros([3, 3])
        D_strain[0][0] = E * (1.0 - nu) / (1.0 + nu) / (1.0 - 2.0 * nu)
        D_strain[0][1] = E / (1.0 + nu) / (1.0 - 2.0 * nu) * nu
        D_strain[1][0] = E / (1.0 + nu) / (1.0 - 2.0 * nu) * nu
        D_strain[1][1] = E * (1.0 - nu) / (1.0 + nu) / (1.0 - 2.0 * nu)
        D_strain[2][2] = E * (1.0 - nu) / (1.0 + nu) / (2.0 - 2.0 * nu)
        return D_strain

    #-------------------------------------------------------------------------
    # Response trace evaluators
    #-------------------------------------------------------------------------

    def get_sig_norm(self, sctx, eps_app_eng):
        sig_eng, D_mtx = self.get_corr_pred(sctx, eps_app_eng, 0, 0, 0)
        return array([scalar_sqrt(sig_eng[0] ** 2 + sig_eng[1] ** 2)])

    def get_eps_p(self, sctx, eps_app_eng):
        # print "eps tracer ", sctx.mats_state_array[:3]
        return sctx.mats_state_array[:3]

    # Declare and fill-in the rte_dict - it is used by the clients to
    # assemble all the available time-steppers.
    #
    rte_dict = Trait(Dict)

    def _rte_dict_default(self):
        return {'sig_app': self.get_sig_app,
                'eps_app': self.get_eps_app,
                'sig_norm': self.get_sig_norm,
                'eps_p': self.get_eps_p}
