
from ibvpy.tmodel.mats_eval import \
    IMATSEval, MATSEval
from traits.api import  \
    provides, Property, cached_property
from bmcs_utils.api import \
    Float, View, Item

import numpy as np

@provides(IMATSEval)
class MATS3DIfcElastic(MATSEval):

    node_name = "Elastic interface"

    E_n = Float(1e+6, tooltip='Normal stiffness of the interface [MPa]',
                MAT=True, unit='MPa/m', symbol='E_\mathrm{n}',
                desc='Normal stiffness of the interface',
                auto_set=False, enter_set=True)

    E_s = Float(1.0, tooltip='Shear stiffness of the interface [MPa]',
                MAT=True, unit='MPa/m', symbol='E_\mathrm{s}',
                desc='Shear-modulus of the interface',
                auto_set=True, enter_set=True)

    state_var_shapes = {}

    D_rs = Property(depends_on='E_n,E_s')

    @cached_property
    def _get_D_rs(self):
        return np.array([[self.E_n, 0, 0],
                         [0, self.E_s, 0],
                         [0, 0, self.E_s]], dtype=np.float64)

    def get_corr_pred(self, u_r, tn1):
        tau = np.einsum(
            'rs,...s->...r',
            self.D_rs, u_r)
        grid_shape = tuple([1 for _ in range(len(u_r.shape[:-1]))])
        D = self.D_rs.reshape(grid_shape + self.D_rs.shape)
        return tau, D

    ipw_view = View(
        Item('E_s'),
        Item('E_n')
    )
