'''
Created on 15.02.2018

@author: abaktheer
'''

from .fets3D import FETS3D
import numpy as np
import sympy as sp
import traits.api as tr
from bmcs_utils.api import InjectSymbExpr, SymbExpr
import k3d

xi_1, xi_2, xi_3 = sp.symbols('xi_1, xi_2, xi_3')

xi_i = sp.Matrix([xi_1, xi_2, xi_3])

#=================================================
# 8 nodes isoparametric volume element (3D)
#=================================================
import sympy as sp

class FETS3D8HSymbExpr(SymbExpr):

    xi_i = xi_i

    N_xi_i = sp.Matrix([(1 - xi_1) * (1 - xi_2) * (1 - xi_3) / 8,
                        (1 + xi_1) * (-1 + xi_2) * (-1 + xi_3) / 8,  # 2
                        (-1 + xi_1) * (1 + xi_2) * (-1 + xi_3) / 8,
                        (-1 - xi_1) * (-1 - xi_2) * (1 - xi_3) / 8,
                        (-1 + xi_1) * (-1 + xi_2) * (1 + xi_3) / 8,
                        (-1 - xi_1) * (1 - xi_2) * (-1 - xi_3) / 8,  # 6
                        (1 - xi_1) * (-1 - xi_2) * (-1 - xi_3) / 8,
                        (1 + xi_1) * (1 + xi_2) * (1 + xi_3) / 8, ]).T

    dN_xi_ai_ = [N_xi.diff(xi_i) for N_xi in N_xi_i]
    dN_xi_ai = sp.Matrix.hstack(*dN_xi_ai_)

    symb_model_params = []

    symb_expressions = [
        ('N_xi_i', ('xi_i',)),
        ('dN_xi_ai', ('xi_i',))
    ]


class FETS3D8H(FETS3D, InjectSymbExpr):

    symb_class = FETS3D8HSymbExpr

    dof_r = tr.Array(np.float_,
                     value=[[-1, -1, -1], [1, -1, -1],
                            [-1, 1, -1], [1, 1, -1],
                            [-1, -1, 1], [1, -1, 1],
                            [-1, 1, 1], [1, 1, 1], ])

    geo_r = tr.Array(np.float_,
                     value=[[-1, -1, -1], [1, -1, -1],
                            [-1, 1, -1], [1, 1, -1],
                            [-1, -1, 1], [1, -1, 1],
                            [-1, 1, 1], [1, 1, 1], ])
    vtk_r = tr.Array(np.float_,
                     value=[[-1, -1, -1], [1, -1, -1],
                            [-1, 1, -1], [1, 1, -1],
                            [-1, -1, 1], [1, -1, 1],
                            [-1, 1, 1], [1, 1, 1], ])
    n_nodal_dofs = 3

    delta = np.identity(n_nodal_dofs)

    vtk_cells = [[0, 1, 3, 2, 4, 5, 7, 6]]
    vtk_cell_types = 'Hexahedron'
    vtk_cell = [0, 1, 3, 2, 4, 5, 7, 6]
    vtk_cell_type = 'Hexahedron'

    vtk_expand_operator = tr.Array(np.float_, value=np.identity(3))

    # numerical integration points (IP) and weights
    xi_m = tr.Array(np.float_,
                    value=[[-1.0 / np.sqrt(3.0), -1.0 / np.sqrt(3.0), -1.0 / np.sqrt(3.0)],
                           [1.0 / np.sqrt(3.0), -1.0 /
                            np.sqrt(3.0), -1.0 / np.sqrt(3.0)],
                           [-1.0 / np.sqrt(3.0), 1.0 /
                            np.sqrt(3.0), -1.0 / np.sqrt(3.0)],
                           [1.0 / np.sqrt(3.0), 1.0 /
                            np.sqrt(3.0), -1.0 / np.sqrt(3.0)],
                           [-1.0 / np.sqrt(3.0), -1.0 /
                            np.sqrt(3.0), 1.0 / np.sqrt(3.0)],
                           [1.0 / np.sqrt(3.0), -1.0 /
                            np.sqrt(3.0), 1.0 / np.sqrt(3.0)],
                           [-1.0 / np.sqrt(3.0), 1.0 /
                            np.sqrt(3.0), 1.0 / np.sqrt(3.0)],
                           [1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0),
                            1.0 / np.sqrt(3.0)],
                           ])

    w_m = tr.Array(value=[1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float_)

    n_m = tr.Property

    def _get_n_m(self):
        return len(self.w_m)

    N_im = tr.Property()
    '''Shape function values in integration poindots.
    '''
    @tr.cached_property
    def _get_N_im(self):
        N_im = self.symb.get_N_xi_i(self.xi_m.T)
        return N_im

    dN_imr = tr.Property()
    '''Shape function derivatives in integration points.
    '''
    @tr.cached_property
    def _get_dN_imr(self):
        N_rim = self.symb.get_dN_xi_ai(self.xi_m.T)
        return np.einsum('rim->imr', N_rim)

    dN_inr = tr.Property()
    '''Shape function derivatives in visualization points.
    '''
    @tr.cached_property
    def _get_dN_inr(self):
        N_rin = self.symb.get_dN_xi_ai(self.dof_r.T)
        return np.einsum('rin->inr', N_rin)

    I_sym_abcd = tr.Array(np.float)

    def _I_sym_abcd_default(self):
        return 0.5 * \
            (np.einsum('ac,bd->abcd', delta, delta) +
             np.einsum('ad,bc->abcd', delta, delta))

    plot_backend = 'k3d'

    def update_plot(self, axes):
        ax = axes
        import numpy as np
        v = np.linspace(-1,1,10)
        x, y, z = np.meshgrid(v,v,v)
        X_aIJK = np.array([x, y, z], dtype=np.float_)
        xmin, xmax, ymin, ymax, zmin, zmax = -1, 1, -1, 1, -1, 1
        N_IJK = fets.symb.get_N_xi_i(X_aIJK)[0,...]
        N_IJK.shape
        plt_iso = k3d.marching_cubes(N_IJK[0,...],compression_level=9,
                                     xmin=xmin, xmax=xmax,
                                 ymin=ymin, ymax=ymax,
                                 zmin=zmin, zmax=zmax, level=0.5,
                                flat_shading=False)
        k3d_plot += plt_iso
        k3d_plot.display()
