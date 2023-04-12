
from numpy import zeros, hstack, meshgrid, vstack
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg.dsolve import linsolve
from traits.api import HasTraits, Property, cached_property, Any

import numpy as np


class COOSparseMtx(HasTraits):

    assemb = Any

    ij_map = Property(depends_on='assemb.+')
    @cached_property
    def _get_ij_map(self):
        '''
        Derive the row and column indices of individual values
        in every element matrix.
        '''

        O_list, P_list = [], []
        # loop over the list of matrix arrays
        for sys_mtx_arr in self.assemb.get_sys_mtx_arrays():

            O_Eo = sys_mtx_arr.dof_map_arr
            a = np.arange(2, dtype=np.int_)
            O, P = (np.einsum('a,Lo->aLo', (1 - a), O_Eo)[:, :, :, None] +
                    np.einsum('a,Lp->aLp', a, O_Eo)[:, :, None, :])
            O_list.append(O.flatten())
            P_list.append(P.flatten())

        return np.hstack(O_list), np.hstack(P_list)

    data_l = Property

    def _get_data_l(self):

        return hstack([sm_arr.mtx_arr.ravel()
                       for sm_arr in self.assemb.get_sys_mtx_arrays()])

    def solve(self, rhs, check_pos_dev=False):
        '''Construct the matrix and use the solver to get 
        the solution for the supplied rhs
        pos_dev - test the positive definiteness of the matrix. 
        '''

        # Assemble the system matrix from the flattened data and
        # sparsity map containing two rows - first one are the row
        # indices and second one are the column indices.
        mtx = sparse.coo_array((self.data_l, self.ij_map))
        mtx_csr = mtx.tocsr()

        pos_def = True
        if check_pos_dev:
            evals_small, evecs_small = eigsh(mtx_csr, 3, sigma=0, which='LM')
            min_eval = np.min(evals_small)
            pos_def = min_eval > 1e-10

        u_vct = linsolve.spsolve(mtx_csr, rhs)
        return u_vct, pos_def

    def toarray(self):
        return sparse.coo_array((self.data_l, self.ij_map)).toarray()
