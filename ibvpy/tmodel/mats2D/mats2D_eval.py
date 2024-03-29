'''
Created on Sep 3, 2009

@author: RChudoba, ABaktheer
'''

from ibvpy.tmodel.mats2D.mats2D_tensor import \
    map2d_sig_eng_to_mtx, map2d_eps_mtx_to_eng, map2d_sig_mtx_to_eng, \
    map2d_tns4_to_tns2, compliance_mapping2d
from ibvpy.tmodel.mats2D.mats2D_tensor import map2d_eps_eng_to_mtx
from traits.api import \
    Array, Property, cached_property, Callable, Constant
from bmcs_utils.api import Enum, Float, View, Item

from ibvpy.tmodel.matsXD.vmatsXD_eval import MATSXDEval
import numpy as np


class MATS2DEval(MATSXDEval):

    n_dims = Constant(2)

    ipw_view = View(
        Item('E'),
        Item('nu'),
        Item('eps_max'),
        Item('G_f', latex=r'G_\mathrm{f}^{\mathrm{estimate}}', readonly=True),
    )

    G_f = Property(Float, depends_on='state_changed')
    @cached_property
    def _get_G_f(self):
        eps_max = self.eps_max
        n_eps = 1000
        eps11_range = np.linspace(1e-9,eps_max,n_eps)
        eps_range = np.zeros((len(eps11_range), 2, 2))
        eps_range[:, 1, 1] = eps11_range
        state_vars = { var : np.zeros( (len(eps11_range),) + shape )
            for var, shape in self.state_var_shapes.items()
        }
        sig_range, _ = self.get_corr_pred(eps_range, 1, **state_vars)
        sig11_range = sig_range[:,1,1]
        return np.trapz(sig11_range, eps11_range)

    stress_state = Enum(
        options=['plane_stress', 'plane_strain'], MAT=True
    )

    D_ab = Property(Array, depends_on='MAT')
    '''Elasticity matrix (shape: (3,3))
    '''
    @cached_property
    def _get_D_ab(self):
        if self.stress_state == 'plane_stress':
            return self._get_D_ab_plane_stress()
        elif self.stress_state == 'plane_strain':
            return self._get_D_ab_plane_strain()

    def _get_D_ab_plane_stress(self):
        '''
        Elastic Matrix - Plane Stress
        '''
        E = self.E
        nu = self.nu
        D_stress = np.zeros([3, 3])
        D_stress[0, 0] = E / (1.0 - nu * nu)
        D_stress[0, 1] = E / (1.0 - nu * nu) * nu
        D_stress[1, 0] = E / (1.0 - nu * nu) * nu
        D_stress[1, 1] = E / (1.0 - nu * nu)
        D_stress[2, 2] = E / (1.0 - nu * nu) * (1.0 / 2.0 - nu / 2.0)
        return D_stress

    def _get_D_ab_plane_strain(self):
        '''
        Elastic Matrix - Plane Strain
        '''
        E = self.E
        nu = self.nu
        D_strain = np.zeros([3, 3])
        D_strain[0, 0] = E * (1.0 - nu) / (1.0 + nu) / (1.0 - 2.0 * nu)
        D_strain[0, 1] = E / (1.0 + nu) / (1.0 - 2.0 * nu) * nu
        D_strain[1, 0] = E / (1.0 + nu) / (1.0 - 2.0 * nu) * nu
        D_strain[1, 1] = E * (1.0 - nu) / (1.0 + nu) / (1.0 - 2.0 * nu)
        D_strain[2, 2] = E * (1.0 - nu) / (1.0 + nu) / (2.0 - 2.0 * nu)
        return D_strain

    map2d_ijkl2a = Array(np.int_,
                         value=[[[[0, 0],
                                  [0, 0]],
                                 [[2, 2],
                                  [2, 2]]],
                                [[[2, 2],
                                  [2, 2]],
                                 [[1, 1],
                                  [1, 1]]]])
    map2d_ijkl2b = Array(np.int_,
                         value=[[[[0, 2],
                                  [2, 1]],
                                 [[0, 2],
                                  [2, 1]]],
                                [[[0, 2],
                                  [2, 1]],
                                 [[0, 2],
                                  [2, 1]]]])

    D_abcd = Property(Array, depends_on='MAT')

    @cached_property
    def _get_D_abcd(self):
        return self.D_ab[self.map2d_ijkl2a, self.map2d_ijkl2b]


class NotUsed:
    # dimension-dependent mappings
    #
    map_tns4_to_tns2 = Callable(map2d_tns4_to_tns2, transient=True)
    map_eps_eng_to_mtx = Callable(map2d_eps_eng_to_mtx, transient=True)
    map_sig_eng_to_mtx = Callable(map2d_sig_eng_to_mtx, transient=True)
    compliance_mapping = Callable(compliance_mapping2d, transient=True)
    map_sig_mtx_to_eng = Callable(map2d_sig_mtx_to_eng, transient=True)
    map_eps_mtx_to_eng = Callable(map2d_eps_mtx_to_eng, transient=True)

    def _get_explorer_config(self):
        '''Get the specific configuration of this material model in the explorer
        '''
        c = super(MATS2DEval, self)._get_explorer_config()

        from ibvpy.api import TLine
        from ibvpy.tmodel.mats2D.mats2D_explorer_bcond import BCDofProportional

        # overload the default configuration
        c['bcond_list'] = [
            BCDofProportional(max_strain=0.00016, alpha_rad=np.pi / 8.0)]
        c['tline'] = TLine(step=0.05, max=1)
        return c

