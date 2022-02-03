'''
'''
from math import sqrt as scalar_sqrt

from ibvpy.tmodel.mats2D.mats2D_eval import MATS2DEval
from ibvpy.tmodel.mats_eval import IMATSEval
from numpy import \
     array, zeros, dot, float_
from traits.api import \
     Array, Enum, \
     Trait, Event, provides, \
     Dict, Property, cached_property
import bmcs_utils.api as bu

#---------------------------------------------------------------------------
# Material time-step-evaluator for Scalar-Damage-Model
#---------------------------------------------------------------------------
@provides( IMATSEval )
class MATS2D5Bond(MATS2DEval):
    '''
    Elastic Model.
    '''


    #---------------------------------------------------------------------------
    # Parameters of the numerical algorithm (integration)
    #---------------------------------------------------------------------------

    stress_state = Enum("plane_stress", "plane_strain")

    #---------------------------------------------------------------------------
    # Material parameters 
    #---------------------------------------------------------------------------

    E_m = bu.Float(1.,  # 34e+3,
                 label="E_m",
                 desc="Young's Modulus",
                 auto_set=False)
    nu_m = bu.Float(0.2,
                 label='nu_m',
                 desc="Poison's ratio",
                 auto_set=False)

    E_f = bu.Float(1.,  # 34e+3,
                 label="E_f",
                 desc="Young's Modulus",
                 auto_set=False)
    nu_f = bu.Float(0.2,
                 label='nu_f',
                 desc="Poison's ratio",
                 auto_set=False)

    G = bu.Float(1.,  # 34e+3,
                 label="G",
                 desc="Shear Modulus",
                 auto_set=False)

    D_el = Property(Array(float), depends_on='E_f, nu_f,E_m,nu_f,G, stress_state')

    @cached_property
    def _get_D_el(self):
        if self.stress_state == "plane_stress":
            return self._get_D_plane_stress()
        else:
            return self._get_D_plane_strain()

    # This event can be used by the clients to trigger an action upon
    # the completed reconfiguration of the material model
    #
    changed = Event

    #---------------------------------------------------------------------------------------------
    # View specification
    #---------------------------------------------------------------------------------------------

    view_traits = bu.View(
        bu.Item('E_m'),
        bu.Item('nu_m'),
        bu.Item('E_f'),
        bu.Item('nu_f'),
        bu.Item('G')
    )

    def get_corr_pred(self, eps_app_eng, tn1):
        '''
        Corrector predictor computation.
        @param eps_app_eng input variable - engineering strain
        '''

        sigma = dot(self.D_el[:], eps_app_eng)

        # You print the stress you just computed and the value of the apparent E

        return  sigma, self.D_el

    #---------------------------------------------------------------------------------------------
    # Subsidiary methods realizing configurable features
    #---------------------------------------------------------------------------------------------

    def _get_D_plane_stress(self):
        E_m = self.E_m
        nu_m = self.nu_m
        E_f = self.E_f
        nu_f = self.nu_f
        G = self.G
        D_stress = zeros([8, 8])
        D_stress[0, 0] = E_m / (1.0 - nu_m * nu_m)
        D_stress[0, 1] = E_m / (1.0 - nu_m * nu_m) * nu_m
        D_stress[1, 0] = E_m / (1.0 - nu_m * nu_m) * nu_m
        D_stress[1, 1] = E_m / (1.0 - nu_m * nu_m)
        D_stress[2, 2] = E_m / (1.0 - nu_m * nu_m) * (1.0 / 2.0 - nu_m / 2.0)

        D_stress[3, 3] = E_f / (1.0 - nu_f * nu_f)
        D_stress[3, 4] = E_f / (1.0 - nu_f * nu_f) * nu_f
        D_stress[4, 3] = E_f / (1.0 - nu_f * nu_f) * nu_f
        D_stress[4, 4] = E_f / (1.0 - nu_f * nu_f)
        D_stress[5, 5] = E_f / (1.0 - nu_f * nu_f) * (1.0 / 2.0 - nu_f / 2.0)

        D_stress[6, 6] = G
        D_stress[7, 7] = G
        return D_stress

    def _get_D_plane_strain(self):
        # TODO: adapt to use arbitrary 2d model following the 1d5 bond
        E_m = self.E_m
        nu_m = self.nu_m
        E_f = self.E_f
        nu_f = self.nu_f
        G = self.G
        D_strain = zeros([8, 8])
        D_strain[0, 0] = E_m * (1.0 - nu_m) / (1.0 + nu_m) / (1.0 - 2.0 * nu_m)
        D_strain[0, 1] = E_m / (1.0 + nu_m) / (1.0 - 2.0 * nu_m) * nu_m
        D_strain[1, 0] = E_m / (1.0 + nu_m) / (1.0 - 2.0 * nu_m) * nu_m
        D_strain[1, 1] = E_m * (1.0 - nu_m) / (1.0 + nu_m) / (1.0 - 2.0 * nu_m)
        D_strain[2, 2] = E_m * (1.0 - nu_m) / (1.0 + nu_m) / (2.0 - 2.0 * nu_m)

        D_strain[3, 3] = E_f * (1.0 - nu_f) / (1.0 + nu_f) / (1.0 - 2.0 * nu_f)
        D_strain[3, 4] = E_f / (1.0 + nu_f) / (1.0 - 2.0 * nu_f) * nu_f
        D_strain[4, 3] = E_f / (1.0 + nu_f) / (1.0 - 2.0 * nu_f) * nu_f
        D_strain[4, 4] = E_f * (1.0 - nu_f) / (1.0 + nu_f) / (1.0 - 2.0 * nu_f)
        D_strain[5, 5] = E_f * (1.0 - nu_f) / (1.0 + nu_f) / (2.0 - 2.0 * nu_f)

        D_strain[6, 6] = G
        D_strain[7, 7] = G
        return D_strain

    #---------------------------------------------------------------------------------------------
    # Response trace evaluators
    #---------------------------------------------------------------------------------------------

    def get_sig_norm(self, sctx, eps_app_eng):
        sig_eng, D_mtx = self.get_corr_pred(sctx, eps_app_eng, 0, 0, 0)
        return array([ scalar_sqrt(sig_eng[0] ** 2 + sig_eng[1] ** 2) ])

    def get_eps_app_m(self, sctx, eps_app_eng):
        return self.map_eps_eng_to_mtx((eps_app_eng[:3]))

    def get_eps_app_f(self, sctx, eps_app_eng):
        return self.map_eps_eng_to_mtx((eps_app_eng[3:6]))

    def get_sig_app_m(self, sctx, eps_app_eng):
        sig_eng, D_mtx = self.get_corr_pred(sctx, eps_app_eng, 0, 0, 0)
        return self.map_sig_eng_to_mtx((sig_eng[:3]))

    def get_sig_app_f(self, sctx, eps_app_eng):
        sig_eng, D_mtx = self.get_corr_pred(sctx, eps_app_eng, 0, 0, 0)
        return self.map_sig_eng_to_mtx((sig_eng[3:6]))

    # Declare and fill-in the rte_dict - it is used by the clients to
    # assemble all the available time-steppers.
    #
    rte_dict = Trait(Dict)

    def _rte_dict_default(self):
        return {
                'eps_app_f'  : self.get_eps_app_f,
                'eps_app_m'  : self.get_eps_app_m,
                'sig_app_f'  : self.get_sig_app_f,
                'sig_app_m'  : self.get_sig_app_m,
                'sig_norm'   : self.get_sig_norm}

