
import copy

from traits.api import \
    HasStrictTraits, Bool, \
    Property, Instance, cached_property, \
    Dict, DelegatesTo, WeakRef, Array, Str

import numpy as np

from .i_tmodel import ITModel
from .i_xmodel import IXModel


class DomainState(HasStrictTraits):
    '''
    Container of spatial domains represented according to XModel.
    and corresponding state derived usi ng the TModel instance
    '''

    tstep = WeakRef

    hist = DelegatesTo('tstep')

    xmodel = Instance(IXModel)

    tmodel = Instance(ITModel)

    state_n = Property(Dict(Str, Array),
                       depends_on='model_structure_changed')
    '''Dictionary of state arrays.
    The entry names and shapes are defined by the material
    model.
    '''
    @cached_property
    def _get_state_n(self):
        xmodel_shape = self.xmodel.state_var_shape
        tmodel_shapes = self.tmodel.state_var_shapes
        return {
            name: np.zeros(xmodel_shape + mats_sa_shape, dtype=np.float64)
            for name, mats_sa_shape
            in list(tmodel_shapes.items())
        }

    hidden = Bool(False)

    state_k = Dict
    '''State variables within the current iteration step
    '''

    def get_corr_pred(self, U_k, t_n, t_n1):
        U_k_field = self.xmodel.map_U_to_field(U_k)
        self.state_k = copy.deepcopy(self.state_n) # todo: optimize memory management.
        sig_k, D_k = self.tmodel.get_corr_pred(
            U_k_field, t_n1, **self.state_k
        )
        K_k = self.xmodel.map_field_to_K(D_k)
        dof_E, f_Ei = self.xmodel.map_field_to_F(sig_k)
        return f_Ei, K_k, dof_E

    def record_state(self):
        '''The trial state k becomes a fundamental state n.
        '''
        for name, s_k in self.state_k.items():
            self.state_n[name] = s_k
        return self.state_n