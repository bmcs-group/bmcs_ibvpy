#-------------------------------------------------------------------------
#
# Copyright (c) 2021, IMB, RWTH Aachen.
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in bmcs/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.bmcs.com/licenses/BSD.txt
#
# Thanks for using BMCS open source!
#

from .i_bcond import \
    IBCond
from ibvpy.mesh.fe_grid_node_slice import FEGridNodeSlice
from ibvpy.tfunction import TimeFunction, TFMonotonic
from ibvpy.mathkit.mfn import MFnLineArray
import numpy as np
import traits.api as tr
import bmcs_utils.api as bu
from .bc_dof import BCDof


@tr.provides(IBCond)
class BCSliceI(bu.Model):
    '''
    Implements the IBC functionality for a constrained dof.
    '''
    name = tr.Str('<unnamed>')

    var = tr.Enum('u', 'f')

    slice = tr.Instance(FEGridNodeSlice)
    link_slice = tr.Instance(FEGridNodeSlice)

    bcdof_list = tr.List(BCDof)

    def reset(self):
        self.bcdof_list = []

    link_coeffs = tr.List(tr.Float)
    '''
    List of dofs that determine the value of the current dof

    If this list is empty, then the current dof is
    prescribed. Otherwise, the dof value is given by the
    linear combination of DOFs in the list (see the example below)

    link_dofs = List( Int )

    Coefficients of the linear combination of DOFs specified in the
    above list.
    '''

    dims = tr.List(tr.Int)

    _link_dims = tr.List(tr.Int)
    link_dims = tr.Property(tr.List(tr.Int))

    def _get_link_dims(self):
        if len(self._link_dims) == 0:
            return self.dims
        else:
            return self._link_dims

    def _set_link_dims(self, link_dims):
        self._link_dims = link_dims

    value = tr.Float

    time_function = tr.Instance(TimeFunction, ())

    def _time_function_default(self):
        return TFMonotonic()

    space_function = tr.Instance(MFnLineArray, ())

    def _space_function_default(self):
        return MFnLineArray(xdata=[0, 1], ydata=[1, 1], extrapolate='diff')

    def is_essential(self):
        return self.var == 'u'

    def is_linked(self):
        return self.link_dofs != []

    def is_constrained(self):
        '''
        Return true if a DOF is either explicitly prescribed or it depends on other DOFS.
        '''
        return self.is_essential() or self.is_linked()

    def is_natural(self):
        return self.var == 'f'

    def setup(self, sctx):
        '''
        Locate the spatial context.f
        '''
        if self.link_slice == None:
            for node_dofs, dof_X in zip(self.slice.dofs,
                                        self.slice.dof_X):
                for dof in node_dofs[self.dims]:
                    self.bcdof_list.append(BCDof(var=self.var,
                                                 dof=dof,
                                                 value=self.value,
                                                 # link_coeffs=self.link_coeffs,
                                                 time_function=self.time_function))
        else:
            # apply the linked slice
            n_link_nodes = len(self.link_slice.dofs.flatten())
            link_dofs = self.link_dofs
            if n_link_nodes == 1:
                #
                link_dof = self.link_slice.dofs.flatten()[0]
                link_coeffs = self.link_coeffs
                for node_dofs, dof_X in zip(self.slice.dofs, self.slice.dof_X):
                    for dof, link_dof, link_coeff in zip(
                            node_dofs[self.dims], link_dofs, link_coeffs):
                        self.bcdof_list.append(BCDof(var=self.var,
                                                     dof=dof,
                                                     link_dofs=[link_dof],
                                                     value=self.value,
                                                     link_coeffs=[link_coeff],
                                                     time_function=self.time_function))
            else:
                for node_dofs, dof_X, node_link_dofs, link_dof_X in \
                    zip(self.slice.dofs, self.slice.dof_X,
                        self.link_slice.dofs, self.link_slice.dof_X):
                    #print('node', node_dofs, node_link_dofs)
                    #print('node[dims]', node_dofs[self.dims],
                          # node_link_dofs[self.link_dims])
                    for dof, link_dof, link_coeff in zip(node_dofs[self.dims],
                                                         node_link_dofs[self.link_dims],
                                                         self.link_coeffs):
                        #print('dof, link, coeff', dof, link_dof, link_coeff)
                        self.bcdof_list.append(BCDof(var=self.var,
                                                     dof=dof,
                                                     link_dofs=[link_dof],
                                                     value=self.value,
                                                     link_coeffs=[
                                                         link_coeff],
                                                     time_function=self.time_function))

    def register(self, K):
        '''Register the boundary condition in the equation system.
        '''
        for bcond in self.bcdof_list:
            bcond.register(K)

    def apply_essential(self, K):

        for bcond in self.bcdof_list:
            bcond.apply_essential(K)

    def apply(self, step_flag, sctx, K, R, t_n, t_n1):

        for bcond in self.bcdof_list:
            bcond.apply(step_flag, sctx, K, R, t_n, t_n1)

    #-------------------------------------------------------------------------
    # Ccnstrained DOFs
    #-------------------------------------------------------------------------

    dofs = tr.Property

    def _get_dofs(self):
        return np.unique(self.slice.dofs[..., self.dims].flatten())

    dof_X = tr.Property

    def _get_dof_X(self):
        return self.slice.dof_X

    n_dof_nodes = tr.Property

    def _get_n_dof_nodes(self):
        sliceshape = self.dofs.shape
        return sliceshape[0] * sliceshape[1]

    #-------------------------------------------------------------------------
    # Link DOFs
    #-------------------------------------------------------------------------
    link_dofs = tr.Property(tr.List)

    def _get_link_dofs(self):
        if self.link_slice != None:
            return np.unique(self.link_slice.dofs[..., self.link_dims].flatten())
        else:
            return []

    link_dof_X = tr.Property

    def _get_link_dof_X(self):
        return self.link_slice.dof_X

    n_link_dof_nodes = tr.Property

    def _get_n_link_dof_nodes(self):
        sliceshape = self.link_dofs.shape
        return sliceshape[0] * sliceshape[1]
