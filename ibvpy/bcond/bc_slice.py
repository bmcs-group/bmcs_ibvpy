#-------------------------------------------------------------------------
#
# Copyright (c) 2009, IMB, RWTH Aachen.
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in simvisage/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.simvisage.com/licenses/BSD.txt
#
# Thanks for using Simvisage open source!
#
# Created on May 26, 2009 by: rchx

from .i_bcond import \
    IBCond
from ibvpy.mesh.fe_grid_idx_slice import FEGridIdxSlice
# from ibvpy.view.plot3d.mayavi_util.pipelines import \
#     MVPointLabels
from ibvpy.tfunction import TimeFunction, TFMonotonic
from ibvpy.mathkit.mfn import MFnLineArray
from numpy import \
    ix_, dot, repeat, zeros
from scipy.linalg import \
    det, norm
from traits.api import Float, \
    Instance, Int, Trait, Str, Enum, \
    List, cached_property, \
    Button, \
    provides, Property
from ibvpy.view.plot2d import Vis2D, Viz2DTimeFunction
from ibvpy.view.ui import BMCSTreeNode

import numpy as np

from .bc_dof import BCDof


@provides(IBCond)
class BCSlice(BMCSTreeNode, Vis2D):
    '''
    Implements the IBC functionality for a constrained dof.
    '''
    name = Str('<unnamed>')
    node_name = Property

    def _get_node_name(self):
        s = '%s:%s=%s' % (self.var, self.slice, self.value)
        return s

    var = Enum('u', 'f', 'eps', 'sig')

    # slice = Instance(FEGridIdxSlice)
    slice = Trait()

    link_slice = Instance(FEGridIdxSlice)

    bcdof_list = List(BCDof)

    def reset(self):
        self.bcdof_list = []
    # List of dofs that determine the value of the current dof
    #
    # If this list is empty, then the current dof is
    # prescribed. Otherwise, the dof value is given by the
    # linear combination of DOFs in the list (see the example below)
    #
    # link_dofs = List( Int )

    # Coefficients of the linear combination of DOFs specified in the
    # above list.
    #
    link_coeffs = List(Float)

    dims = List(Int)

    _link_dims = List(Int)
    link_dims = Property(List(Int))

    def _get_link_dims(self):
        if len(self._link_dims) == 0:
            return self.dims
        else:
            return self._link_dims

    def _set_link_dims(self, link_dims):
        self._link_dims = link_dims

    value = Float

    integ_domain = Enum(['global', 'local'])

    time_function = Instance(TimeFunction, ())

    def _time_function_default(self):
        return TFMonotonic()

    space_function = Instance(MFnLineArray, ())

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
        return self.var == 'f' or self.var == 'eps' or self.var == 'sig'

    def setup(self, sctx):
        '''
        Locate the spatial context.f
        '''
        if self.link_slice == None:
            for el, el_dofs, el_dofs_X in zip(self.slice.elems,
                                              self.slice.dofs,
                                              self.slice.dof_X):
                # print 'el_dofs', el_dofs
                for node_dofs, dof_X in zip(el_dofs, el_dofs_X):
                    # print 'node_dofs ', node_dofs
                    for dof in node_dofs[self.dims]:
                        self.bcdof_list.append(BCDof(var=self.var,
                                                     dof=dof,
                                                     value=self.value,
                                                     link_coeffs=self.link_coeffs,
                                                     time_function=self.time_function))
        else:
            # apply the linked slice
            n_link_nodes = len(self.link_slice.dof_nodes.flatten())
            link_dofs = self.link_dofs
            if n_link_nodes == 1:
                #
                link_dof = self.link_slice.dofs.flatten()[0]
                link_coeffs = self.link_coeffs
                for el, el_dofs, el_dofs_X in \
                        zip(self.slice.elems, self.slice.dofs, self.slice.dof_X):
                    for node_dofs, dof_X in zip(el_dofs, el_dofs_X):
                        for dof, link_dof, link_coeff in zip(node_dofs[self.dims], link_dofs, link_coeffs):
                            self.bcdof_list.append(BCDof(var=self.var,
                                                         dof=dof,
                                                         link_dofs=[link_dof],
                                                         value=self.value,
                                                         link_coeffs=[
                                                             link_coeff],
                                                         time_function=self.time_function))
            else:
                for el, el_dofs, el_dofs_X, el_link, el_link_dofs, el_link_dofs_X in \
                    zip(self.slice.elems, self.slice.dofs, self.slice.dof_X,
                        self.link_slice.elems, self.link_slice.dofs, self.link_slice.dof_X):
                    # the link slice is compatible with the bc slice
                    #print('el', el, el_dofs, el_link, el_link_dofs)

                    for node_dofs, dof_X, node_link_dofs, link_dof_X in \
                            zip(el_dofs, el_dofs_X, el_link_dofs, el_link_dofs_X):

                        #print('node', node_dofs, node_link_dofs)
                        #print('node[dims]', node_dofs[self.dims],
                              # node_link_dofs[self.link_dims])
                        for dof, link_dof, link_coeff in zip(node_dofs[self.dims],
                                                             node_link_dofs[
                                                                 self.link_dims],
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

        if self.is_essential():
            for bcond in self.bcdof_list:
                bcond.apply(step_flag, sctx, K, R, t_n, t_n1)
        else:
            self.apply_natural(step_flag, sctx, K, R, t_n, t_n1)

    def apply_natural(self, step_flag, sctx, K, R, t_n, t_n1):

        fets_eval = self.slice.fe_grid.fets_eval

        e_idx, n_idx = self.slice.dof_grid_slice.idx_tuple

        r_arr, w_arr, ix = fets_eval.get_sliced_ip_scheme(n_idx)

        slice_geo_X = self.slice.fe_grid.elem_X_map[self.slice.elems]
        slice_dofs = self.slice.fe_grid.elem_dof_map[self.slice.elems]

        p_value = self.value * float(self.time_function(t_n1))

        p_vct = zeros((fets_eval.n_nodal_dofs,), dtype='float64')
        for d in self.dims:
            p_vct[d] = p_value

        for el, el_dofs, el_geo_X in zip(self.slice.elems, slice_dofs, slice_geo_X):
            f_vct = zeros((fets_eval.n_e_dofs,), dtype='float64')
            for r_pnt, w in zip(r_arr, w_arr):
                if len(ix) > 0:
                    J_mtx = fets_eval.get_J_mtx(r_pnt, el_geo_X)

                    if self.integ_domain == 'global':

                        # Integration projected onto the global coordinates
                        # axes - use submatrix of the J_mtx
                        #
                        J_det = det(J_mtx[ix_(ix, ix)])

                    elif self.integ_domain == 'local':

                        # Integration over the parametric coordinates on the
                        # surface
                        #
                        J_det = 1.0
                        for i in ix:
                            J_det *= norm(J_mtx[:, i])
                else:
                    J_det = 1.0

                sctx.r_pnt = r_pnt
                sctx.X = el_geo_X
                X_pnt = fets_eval.get_X_pnt(sctx)
                space_factor = self.space_function(X_pnt)

                # add to the element vector
                N_mtx = fets_eval.get_N_mtx(r_pnt)
                f_vct += dot(N_mtx.T, p_vct * space_factor) * w * J_det

            R[el_dofs] += f_vct

    #-------------------------------------------------------------------------
    # Ccnstrained DOFs
    #-------------------------------------------------------------------------

    dofs = Property

    def _get_dofs(self):
        return np.unique(self.slice.dofs[..., self.dims].flatten())

    dof_X = Property

    def _get_dof_X(self):
        return self.slice.dof_X

    n_dof_nodes = Property

    def _get_n_dof_nodes(self):
        sliceshape = self.dofs.shape
        return sliceshape[0] * sliceshape[1]

    # # register the pipelines for plotting labels and geometry
    # #
    # mvp_dofs = Trait(MVPointLabels)
    #
    # def _mvp_dofs_default(self):
    #     return MVPointLabels(name='Boundary condition',
    #                          points=self._get_mvpoints,
    #                          vectors=self._get_labels,
    #                          color=(0.0, 0.0, 0.882353))

    def _get_mvpoints(self):
        # blow up
        if self.dof_X.shape[2] == 2:
            dof_X = np.hstack([self.dof_X.reshape(self.n_dof_nodes, 2),
                               np.zeros((self.n_dof_nodes, 1), dtype='f')])
        elif self.dof_X.shape[2] == 1:
            dof_X = np.hstack([self.dof_X.reshape(self.n_dof_nodes, 1),
                               np.zeros((self.n_dof_nodes, 2), dtype='f')])
        else:
            dof_X = self.dof_X.reshape(self.n_dof_nodes, 3)
        return dof_X

    def _get_labels(self):
        # blow up
        n_points = self.n_dof_nodes
        dofs = repeat(-1., n_points * 3).reshape(n_points, 3)
        dofs[:, tuple(self.dims)] = self.dofs
        return dofs

    #-------------------------------------------------------------------------
    # Link DOFs
    #-------------------------------------------------------------------------
    link_dofs = Property(List)

    def _get_link_dofs(self):
        if self.link_slice != None:
            return np.unique(self.link_slice.dofs[..., self.link_dims].flatten())
        else:
            return []

    link_dof_X = Property

    def _get_link_dof_X(self):
        return self.link_slice.dof_X

    n_link_dof_nodes = Property

    def _get_n_link_dof_nodes(self):
        sliceshape = self.link_dofs.shape
        return sliceshape[0] * sliceshape[1]

    # # register the pipelines for plotting labels and geometry
    # #
    # mvp_link_dofs = Trait(MVPointLabels)
    #
    # def _mvp_link_dofs_default(self):
    #     return MVPointLabels(name='Link boundary condition',
    #                          points=self._get_link_mvpoints,
    #                          vectors=self._get_link_labels,
    #                          color=(0.0, 0.882353, 0.0))

    def _get_link_mvpoints(self):
        # blow up
        # blow up
        if self.link_slice == None:
            return np.zeros((0, 3), dtype='f')
        if self.dof_X.shape[2] == 2:
            dof_X = np.hstack([self.link_dof_X.reshape(self.n_dof_nodes, 2),
                               np.zeros((self.n_dof_nodes, 1), dtype='f')])
        elif self.dof_X.shape[2] == 1:
            dof_X = np.hstack([self.link_dof_X.reshape(self.n_dof_nodes, 1),
                               np.zeros((self.n_dof_nodes, 2), dtype='f')])
        else:
            dof_X = self.link_dof_X.reshape(self.n_dof_nodes, 3)
        return dof_X

    def _get_link_labels(self):
        # blow up
        if self.link_slice == None:
            return np.zeros((0, 3), dtype='f')
        n_points = self.n_link_dof_nodes
        dofs = repeat(-1., n_points * 3).reshape(n_points, 3)
        dofs[:, tuple(self.dims)] = self.link_dofs
        return dofs

    redraw_button = Button('Redraw')

    def _redraw_button_fired(self):
        self.mvp_dofs.redraw(label_mode='label_vectors')
        self.mvp_link_dofs.redraw(label_mode='label_vectors')

    viz2d_classes = {'time function': Viz2DTimeFunction}
