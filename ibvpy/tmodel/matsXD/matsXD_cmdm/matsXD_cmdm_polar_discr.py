
from math import pi as Pi
from numpy import \
    array,  frompyfunc
from traits.api import \
    Array, Bool, Callable,  Float, HasTraits, \
    Instance,   Range,  on_trait_change,  \
    Dict, Property, cached_property,   \
    WeakRef, String, List, Constant, Str, Type, TraitError

from ibvpy.mathkit.mfn.mfn_polar.mfn_polar import MFnPolar
from .matsXD_cmdm_phi_fn import \
    IPhiFn, PhiFnStrainSoftening, PhiFnStrainHardening, PhiFnStrainHardeningLinear, \
    PhiFnGeneral, PhiFnGeneralExtended, PhiFnGeneralExtendedExp, PhiFnStrainHardeningBezier
from ibvpy.util.traits.either_type import EitherType


# Traits UI imports
# @todo change to mfn_polar
#-------------------------------------------------------------------------
#                                     VariedParam
#-------------------------------------------------------------------------
class VariedParam(HasTraits):

    """
    Association between the spatial function and material parameter.
    """
    mats_eval = WeakRef
    phi_fn = WeakRef(IPhiFn)

    varname = String

    reference_value = Property(Float)

    def _get_reference_value(self):
        return getattr(self.phi_fn, self.varname)

    switched_on = Bool(False)

    polar_fn = Instance(MFnPolar)

    @on_trait_change('switched_on')
    def reset_polar_fn(self):
        if self.switched_on:
            if self.mats_eval.mfn_class == None:
                raise TraitError('No class for function representation specified')
            self.polar_fn = self.mats_eval.mfn_class()
        else:
            self.polar_fn = None

    # vectorized form of polar_fn to returning an array of coefficients
    polar_fn_vectorized = Property(Callable, depends_on='polar_fn')

    @cached_property
    def _get_polar_fn_vectorized(self):
        if self.switched_on:
            return frompyfunc(self.polar_fn, 1, 1)
        else:
            return frompyfunc(lambda angle: 1., 1, 1)


class PolarDiscr(HasTraits):

    '''
    Manager of the microplane arrays.

    This class is responsible for the generation and initialization
    and state management of an array of microplanes. Additionally, it
    can perform the setup of damage function parameters using the
    value of the microplane integrator object.
    '''

    mfn_class = Type(None)
    #-------------------------------------------------------------------------
    # Common parameters for for isotropic and anisotropic damage function specifications
    #-------------------------------------------------------------------------
    n_mp = Range(0, 50, 6,
                 label='Number of microplanes',
                 auto_set=False)

    E = Float(34e+3,
              label="E",
              desc="Young's Modulus",

              auto_set=False, enter_set=True)
    nu = Float(0.2,
               label='nu',
               desc="Poison's ratio",
               auto_set=False, enter_set=True)

    c_T = Float(0.0,
                label='c_T',
                desc='fraction of tangential stress accounted on each microplane',
                auto_set=False, enter_set=True)

    #-------------------------------------------------------------------------
    # list of angles
    #-------------------------------------------------------------------------
    alpha_list = Property(Array, depends_on='n_mp')

    @cached_property
    def _get_alpha_list(self):
        return array([Pi / self.n_mp * (i - 0.5) for i in range(1, self.n_mp + 1)])

    #-------------------------------------------------------------------------
    # Damage function specification
    #-------------------------------------------------------------------------

    phi_fn = EitherType(klasses=[PhiFnGeneral,
                                 PhiFnGeneralExtended,
                                 PhiFnGeneralExtendedExp,
                                 PhiFnStrainSoftening,
                                 PhiFnStrainHardening,
                                 PhiFnStrainHardeningLinear,
                                 PhiFnStrainHardeningBezier])

    def _phi_fn_default(self):
        print('setting phi_fn default')
        return PhiFnStrainSoftening(polar_discr=self)

    def _phi_fn_changed(self):
        print('setting phi_fn changed')
        self.phi_fn.polar_discr = self

    varied_params = List(Str, [])

    #-------------------------------------------------------------------------
    # Management of spatially varying parameters depending on the value of mats_eval
    #-------------------------------------------------------------------------
    varpars = Dict

    def _varpars_default(self):
        return self._get_varpars()

    @on_trait_change('phi_fn,varied_params')
    def _update_varpars(self):
        self.varpars = self._get_varpars()

    def _get_varpars(self):
        '''
        reset the varpar list according to the current phi_fn object.
        '''
        params = self.phi_fn.identify_parameters()
        varset = {}
        for key in params:
            varset[key] = VariedParam(phi_fn=self.phi_fn,
                                      mats_eval=self,
                                      varname=key)
            if key in self.varied_params:
                varset[key].switched_on = True
        return varset

    varpar_list = Property(List(VariedParam), depends_on='varpars')

    @cached_property
    def _get_varpar_list(self):
        return [self.varpars[key]
                for key in self.phi_fn.identify_parameters()]

    # variable selectable in the table of varied params (just for viewing)
    current_varpar = Instance(VariedParam)

    def _current_varpar_default(self):
        if len(self.varpar_list) > 0:
            return self.varpar_list[0]
        return None

    @on_trait_change('phi_fn')
    def set_current_varpar(self):
        if len(self.varpar_list) > 0:
            self.current_varpar = self.varpar_list[0]

    #-------------------------------------------------------------------------
    # Get the damage state for all microplanes
    #-------------------------------------------------------------------------
    def get_phi_arr(self, sctx, e_max_arr):
        '''
        Return the damage coefficients
        '''
        # gather the coefficients for parameters depending on the orientation
        carr_list = [self.varpars[key].polar_fn_vectorized(self.alpha_list)
                     for key in self.phi_fn.identify_parameters()]
        # vectorize the damage function evaluation
        n_arr = 1 + len(carr_list)
        phi_fn_vectorized = frompyfunc(self.phi_fn.get_value, n_arr, 1)
        # damage parameter for each microplane
        return phi_fn_vectorized(e_max_arr, *carr_list)

    def get_polar_fn_fracture_energy_arr(self, sctx, e_max_arr):
        '''
        Return the fracture energy contributions
        '''
        carr_list = [self.varpars[key].polar_fn_vectorized(self.alpha_list)
                     for key in self.phi_fn.identify_parameters()]
        # vectorize the damage function evaluation
        n_arr = 1 + len(carr_list)
        integ_phi_fn_vectorized = frompyfunc(self.phi_fn.get_integ, n_arr, 1)
        return self.E * integ_phi_fn_vectorized(e_max_arr, *carr_list)

