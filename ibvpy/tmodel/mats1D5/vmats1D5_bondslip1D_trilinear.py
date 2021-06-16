'''
Created on 05.12.2016

@author: abaktheer
'''

from os.path import join

from ibvpy.tmodel import MATSEval
from ibvpy.mathkit.mfn.mfn_line.mfn_line import MFnLineArray
from traits.api import  \
    Tuple, List, on_trait_change, \
    Instance, Trait, Bool, Str, Button, Property, cached_property

import bmcs_utils.api as bu
import numpy as np
import traitsui.api as ui
from .vmats1D5_bondslip1D import MATSEval1D5

class MATS1D5BondSlipTriLinear(MATSEval1D5):
    """Multilinear bond-slip law
    """
    name = "tri-linear bond law"

    E_m = bu.Float(28000.0, tooltip='Stiffness of the matrix [MPa]',
                MAT=True, unit='MPa', symbol=r'E_\mathrm{m}',
                desc='E-modulus of the matrix')

    E_f = bu.Float(170000.0, tooltip='Stiffness of the fiber [MPa]',
                MAT=True, unit='MPa', symbol=r'E_\mathrm{f}',
                desc='E-modulus of the reinforcement')

    tau_1 = bu.Float(10.0, tooltip='Shear strength [MPa]',
                MAT=True, unit='MPa', symbol=r'\tau_1',
                desc='shear strength')

    tau_2 = bu.Float(1.0, tooltip='Shear at plateau [MPa]',
                MAT=True, unit='MPa', symbol=r'\tau_2',
                desc='shear plateau')

    s_1 = bu.Float(0.1, tooltip='Slip at peak [mm]',
                MAT=True, unit='mm', symbol='s_1',
                desc='slip at peak')

    s_2 = bu.Float(0.5, tooltip='Slip at plateau [mm]',
                MAT=True, unit='mm', symbol='s_2',
                desc='slip at plateau')

    s_tau_table = Property(depends_on='state_changed')
    @cached_property
    def _get_s_tau_table(self):
        s_data = [0, self.s_1, self.s_2, self.s_2 * 3]
        tau_data = [0, self.tau_1, self.tau_2, self.tau_2]
        if len(s_data) != len(tau_data):
            raise ValueError('s array and tau array must have the same size')
        return s_data, tau_data

    #=========================================================================
    # Configurational parameters
    #=========================================================================
    U_var_shape = (1,)
    '''Shape of the primary variable required by the TStepState.
    '''

    state_var_shapes = {}
    r'''
    Shapes of the state variables
    to be stored in the global array at the level 
    of the domain.
    '''

    node_name = 'tri-linear linear bond'

    def get_corr_pred(self, eps_n1, t_n1):
        D_shape = eps_n1.shape[:-1] + (3, 3)
        D = np.zeros(D_shape, dtype=np.float_)

        D[..., 0, 0] = self.E_m
        D[..., 2, 2] = self.E_f

        tau = np.einsum('...st,...t->...s', D, eps_n1)
        s = eps_n1[..., 1]
        shape = s.shape
        signs = np.sign(s.flatten())
        s_pos = np.fabs(s.flatten())
        bs_law = self.bs_law
        tau[..., 1] = (signs * bs_law(s_pos)).reshape(*shape)
        D_tau = self.bs_law.diff(s_pos).reshape(*shape)
        D[..., 1, 1] = D_tau

        return tau, D

    bs_law = Property(depends_on='state_changed')
    @cached_property
    def _get_bs_law(self):
        s_data, tau_data = self.s_tau_table
        return MFnLineArray(
            xdata=s_data,
            ydata=tau_data,
            plot_diff=False
        )

    ipw_view = bu.View(
        bu.Item('E_m'),
        bu.Item('E_f'),
        bu.Item('tau_1'),
        bu.Item('s_1'),
        bu.Item('tau_2'),
        bu.Item('s_2'),
        bu.Item('s_max'),
    )
