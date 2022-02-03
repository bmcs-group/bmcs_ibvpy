
from math import sqrt as scalar_sqrt

from ibvpy.tmodel.mats2D.mats2D_eval import MATS2DEval
from ibvpy.tmodel.mats_eval import IMATSEval
from numpy import \
    array, zeros, dot, \
    float_, \
    eye
from traits.api import \
    Array, Enum, Float, \
    Trait, Event, provides, \
    Dict, Property, cached_property
from bmcs_utils.api import \
    Item, View


@provides(IMATSEval)
class MATS2DConduction(MATS2DEval):
    '''
    Elastic Model.
    '''

    #-------------------------------------------------------------------------
    # Parameters of the numerical algorithm (integration)
    #-------------------------------------------------------------------------

    stress_state = Enum("plane_stress", "plane_strain", "rotational_symetry")

    #-------------------------------------------------------------------------
    # Material parameters
    #-------------------------------------------------------------------------

    k = Float(1.,
              label="k",
              desc="conduction",
              auto_set=False)

    D_mtx = Property(Array, depends_on='k')

    @cached_property
    def _get_D_mtx(self):
        return self.k * eye(2)

    # This event can be used by the clients to trigger an action upon
    # the completed reconfiguration of the material model
    #
    changed = Event

    #-------------------------------------------------------------------------
    # View specification
    #-------------------------------------------------------------------------

    view_traits = View(Item('k'),
                              ),

    #-------------------------------------------------------------------------
    # Private initialization methods
    #-------------------------------------------------------------------------

    #-------------------------------------------------------------------------
    # Setup for computation within a supplied spatial context
    #-------------------------------------------------------------------------

    def new_cntl_var(self):
        return zeros(3, float_)

    def new_resp_var(self):
        return zeros(3, float_)

    #-------------------------------------------------------------------------
    # Evaluation - get the corrector and predictor
    #-------------------------------------------------------------------------

    def get_corr_pred(self, eps_app_eng, tn1):
        '''
        Corrector predictor computation.
        @param eps_app_eng input variable - engineering strain
        '''

        # You print the stress you just computed and the value of the apparent
        # E

        return dot(self.D_mtx, eps_app_eng), self.D_mtx

    #-------------------------------------------------------------------------
    # Subsidiary methods realizing configurable features
    #-------------------------------------------------------------------------

    #-------------------------------------------------------------------------
    # Response trace evaluators
    #-------------------------------------------------------------------------

    def get_sig_norm(self, sctx, eps_app_eng):
        sig_eng, D_mtx = self.get_corr_pred(sctx, eps_app_eng, 0, 0, 0)
        return array([scalar_sqrt(sig_eng[0] ** 2 + sig_eng[1] ** 2)])

    # Declare and fill-in the rte_dict - it is used by the clients to
    # assemble all the available time-steppers.
    #
    rte_dict = Trait(Dict)

    def _rte_dict_default(self):
        return {'sig_app': self.get_sig_app,
                'eps_app': self.get_eps_app,
                'sig_norm': self.get_sig_norm,
                'strain_energy': self.get_strain_energy}

