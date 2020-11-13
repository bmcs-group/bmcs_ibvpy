
from numpy import linalg, ix_, zeros
from traits.api import HasTraits, Property, Any


class DenseMtx(HasTraits):
    '''Dense matrix with the interface of a sparse matrix assembly.
    Used for debugging and performance comparison of sparse solvers.
    '''
    assemb = Any

    mtx = Property

    def _get_mtx(self):
        n_dofs = self.assemb.n_dofs
        sys_mtx = zeros([n_dofs, n_dofs], dtype=float)
        # loop over the list of matrix arrays
        for sys_mtx_arr in self.assemb.get_sys_mtx_arrays():
            # loop over the matrix array
            for dof_map, mtx in zip(sys_mtx_arr.dof_map_arr,
                                    sys_mtx_arr.mtx_arr):
                sys_mtx[ix_(dof_map, dof_map)] += mtx
        return sys_mtx

    def solve(self, rhs):
        u_vct = linalg.solve(self.mtx, rhs)
        return u_vct

    def __str__(self):
        '''String representation - delegate to matrix'''
        return str(self.mtx)
