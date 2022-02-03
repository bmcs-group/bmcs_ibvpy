
from traits.api import \
    HasTraits, List, Array, Property, cached_property, \
    Instance, Trait, Button, on_trait_change, \
    Int, Float, DelegatesTo, provides, WeakRef, Bool

from ibvpy.mesh.sdomain import \
    SDomain
# from ibvpy.view.plot3d.mayavi_util.pipelines import \
#     MVPolyData, MVPointLabels
import numpy as np

from .cell_array import CellView, CellArray, ICellArraySource
from .cell_grid import CellGrid
from .cell_grid_slice import CellGridSlice


#--------------------------------------------------------------------------
# DofGrid
#--------------------------------------------------------------------------
@provides(ICellArraySource)
class DofCellGrid(SDomain):

    '''
    Get an array with element Dof numbers
    '''
    cell_grid = Instance(CellGrid)

    get_cell_point_X_arr = DelegatesTo('cell_grid')
    get_cell_mvpoints = DelegatesTo('cell_grid')
    cell_node_map = DelegatesTo('cell_grid')
    get_cell_offset = DelegatesTo('cell_grid')

    # offset of dof within domain list
    #
    dof_offset = Int(0)

    # number of degrees of freedom in a single node
    #
    n_nodal_dofs = Int(3)
    #-------------------------------------------------------------------------
    # Generation methods for geometry and index maps
    #-------------------------------------------------------------------------
    n_dofs = Property(depends_on='cell_grid.shape,n_nodal_dofs,dof_offset')

    def _get_n_dofs(self):
        '''
        Get the total number of DOFs
        '''
        unique_cell_nodes = np.unique(self.cell_node_map.flatten())
        n_unique_nodes = len(unique_cell_nodes)
        return n_unique_nodes * self.n_nodal_dofs

    dofs = Property(depends_on='cell_grid.shape,n_nodal_dofs,dof_offset')

    @cached_property
    def _get_dofs(self):
        '''
        Construct the point grid underlying the mesh grid structure.
        '''
        cell_node_map = self.cell_node_map

        unique_cell_nodes = np.unique(cell_node_map.flatten())
        n_unique_nodes = len(unique_cell_nodes)

        n_nodal_dofs = self.n_nodal_dofs
        n_nodes = self.cell_grid.point_grid_size
        node_dof_array = np.repeat(-1, n_nodes *
                                   n_nodal_dofs).reshape(n_nodes, n_nodal_dofs)

        # Enumerate the DOFs in the mesh. The result is an array with n_nodes rows
        # and n_nodal_dofs columns
        #
        # A = array( [[ 0, 1 ],
        #             [ 2, 3 ],
        #             [ 4, 5 ]] );
        #
        node_dof_array[np.index_exp[unique_cell_nodes]] = \
            np.arange(
                n_unique_nodes * n_nodal_dofs).reshape(n_unique_nodes,
                                                       n_nodal_dofs)

        # add the dof_offset before returning the array
        #
        node_dof_array += self.dof_offset
        return node_dof_array

    dofs_Ia = Property()

    def _get_dofs_Ia(self):
        return self.dofs

    def _get_doffed_nodes(self):
        '''
        Get the indices of nodes containing DOFs. 
        '''
        cell_node_map = self.cell_node_map

        unique_cell_nodes = np.unique(cell_node_map.flatten())

        n_nodes = self.cell_grid.point_grid_size
        doffed_nodes = np.repeat(-1, n_nodes)

        doffed_nodes[np.index_exp[unique_cell_nodes]] = 1
        return np.where(doffed_nodes > 0)[0]

    #-----------------------------------------------------------------
    # Elementwise-representation of dofs
    #-----------------------------------------------------------------

    cell_dof_map = Property(depends_on='cell_grid.shape,n_nodal_dofs')

    def _get_cell_dof_map(self):
        return self.dofs[np.index_exp[self.cell_grid.cell_node_map]]

    dof_Eid = Property
    '''Mapping of Element, Node, Dimension -> DOF 
    '''

    def _get_dof_Eid(self):
        return self.cell_dof_map

    cell_grid_dof_map = Property(depends_on='cell_grid.shape,n_nodal_dofs')

    def _get_cell_grid_dof_map(self):
        return self.dofs[np.index_exp[self.cell_grid.cell_grid_node_map]]

    def get_cell_dofs(self, cell_idx):
        return self.cell_dof_map[cell_idx]

    elem_dof_map = Property(depends_on='cell_grid.shape,n_nodal_dofs')

    @cached_property
    def _get_elem_dof_map(self):
        el_dof_map = np.copy(self.cell_dof_map)
        tot_shape = el_dof_map.shape[0]
        n_entries = el_dof_map.shape[1] * el_dof_map.shape[2]
        elem_dof_map = el_dof_map.reshape(tot_shape, n_entries)
        return elem_dof_map

    def __getitem__(self, idx):
        '''High level access and slicing to the cells within the grid.

        The return value is a tuple with 
        1. array of cell indices
        2. array of nodes for each element
        3. array of coordinates for each node.
        '''
        dgs = DofGridSlice(dof_grid=self, grid_slice=idx)
        return dgs

    #-----------------------------------------------------------------
    # Spatial queries for dofs
    #-----------------------------------------------------------------

    def _get_dofs_for_nodes(self, nodes):
        '''Get the dof numbers and associated coordinates
        given the array of nodes.
        '''
        doffed_nodes = self._get_doffed_nodes()
#         print 'nodes'
#         print nodes
#         print 'doffed_nodes'
#         print doffed_nodes
        intersect_nodes = np.intersect1d(
            nodes, doffed_nodes, assume_unique=False)
        return (self.dofs[np.index_exp[intersect_nodes]],
                self.cell_grid.point_X_arr[np.index_exp[intersect_nodes]])

    def get_boundary_dofs(self):
        '''Get the boundary dofs and the associated coordinates
        '''
        nodes = [self.cell_grid.point_idx_grid[s]
                 for s in self.cell_grid.boundary_slices]
        dofs, coords = [], []
        for n in nodes:
            d, c = self._get_dofs_for_nodes(n)
            dofs.append(d)
            coords.append(c)
        return (np.vstack(dofs), np.vstack(coords))

    def get_all_dofs(self):
        nodes = self.cell_grid.point_idx_grid[...]
        return self._get_dofs_for_nodes(nodes)

    def get_left_dofs(self):
        nodes = self.cell_grid.point_idx_grid[0, ...]
        return self._get_dofs_for_nodes(nodes)

    def get_right_dofs(self):
        nodes = self.cell_grid.point_idx_grid[-1, ...]
        return self._get_dofs_for_nodes(nodes)

    def get_top_dofs(self):
        nodes = self.cell_grid.point_idx_grid[:, -1, ...]
        return self._get_dofs_for_nodes(nodes)

    def get_bottom_dofs(self):
        nodes = self.cell_grid.point_idx_grid[:, 0, ...]
        return self._get_dofs_for_nodes(nodes)

    def get_front_dofs(self):
        nodes = self.cell_grid.point_idx_grid[:, :, -1]
        return self._get_dofs_for_nodes(nodes)

    def get_back_dofs(self):
        nodes = self.cell_grid.point_idx_grid[:, :, 0]
        return self._get_dofs_for_nodes(nodes)

    def get_bottom_left_dofs(self):
        nodes = self.cell_grid.point_idx_grid[0, 0, ...]
        return self._get_dofs_for_nodes(nodes)

    def get_bottom_front_dofs(self):
        nodes = self.cell_grid.point_idx_grid[:, 0, -1]
        return self._get_dofs_for_nodes(nodes)

    def get_bottom_back_dofs(self):
        nodes = self.cell_grid.point_idx_grid[:, 0, 0]
        return self._get_dofs_for_nodes(nodes)

    def get_top_left_dofs(self):
        nodes = self.cell_grid.point_idx_grid[0, -1, ...]
        return self._get_dofs_for_nodes(nodes)

    def get_bottom_right_dofs(self):
        nodes = self.cell_grid.point_idx_grid[-1, 0, ...]
        return self._get_dofs_for_nodes(nodes)

    def get_top_right_dofs(self):
        nodes = self.cell_grid.point_idx_grid[-1, -1, ...]
        return self._get_dofs_for_nodes(nodes)

    def get_bottom_middle_dofs(self):
        if self.cell_grid.point_idx_grid.shape[0] % 2 == 1:
            slice_middle_x = self.cell_grid.point_idx_grid.shape[0] / 2
            nodes = self.cell_grid.point_idx_grid[slice_middle_x, 0, ...]
            return self._get_dofs_for_nodes(nodes)
        else:
            print('Error in get_bottom_middle_dofs:'
                  ' the method is only defined for an odd number of dofs in x-direction')

    def get_top_middle_dofs(self):
        if self.cell_grid.point_idx_grid.shape[0] % 2 == 1:
            slice_middle_x = self.cell_grid.point_idx_grid.shape[0] / 2
            nodes = self.cell_grid.point_idx_grid[slice_middle_x, -1, ...]
            return self._get_dofs_for_nodes(nodes)
        else:
            print('Error in get_top_middle_dofs:'
                  ' the method is only defined for an odd number of dofs in x-direction')

    def get_left_middle_dofs(self):
        if self.cell_grid.point_idx_grid.shape[1] % 2 == 1:
            slice_middle_y = self.cell_grid.point_idx_grid.shape[1] / 2
            nodes = self.cell_grid.point_idx_grid[0, slice_middle_y, ...]
            return self._get_dofs_for_nodes(nodes)
        else:
            print('Error in get_left_middle_dofs:'
                  ' the method is only defined for an odd number of dofs in y-direction')

    def get_right_middle_dofs(self):
        if self.cell_grid.point_idx_grid.shape[1] % 2 == 1:
            slice_middle_y = self.cell_grid.point_idx_grid.shape[1] / 2
            nodes = self.cell_grid.point_idx_grid[-1, slice_middle_y, ...]
            return self._get_dofs_for_nodes(nodes)
        else:
            print('Error in get_right_middle_dofs:'
                  ' the method is only defined for an odd number of dofs in y-direction')

    def get_left_front_bottom_dof(self):
        nodes = self.cell_grid.point_idx_grid[0, 0, -1]
        return self._get_dofs_for_nodes(nodes)

    def get_left_front_middle_dof(self):
        if self.cell_grid.point_idx_grid.shape[1] % 2 == 1:
            slice_middle_y = self.cell_grid.point_idx_grid.shape[1] / 2
            nodes = self.cell_grid.point_idx_grid[0, slice_middle_y, -1]
            return self._get_dofs_for_nodes(nodes)
        else:
            print('Error in get_left_middle_front_dof:'
                  ' the method is only defined for an odd number of dofs in y-direction')

    #-----------------------------------------------------------------
    # Visualization related methods
    #-----------------------------------------------------------------

    refresh_button = Button('Draw')

    @on_trait_change('refresh_button')
    def redraw(self):
        '''Redraw the point grid.
        '''
        self.cell_grid.redraw()

    dof_cell_array = Button

    def _dof_cell_array_fired(self):
        cell_array = self.cell_grid.cell_node_map
        self.show_array = CellArray(data=cell_array,
                                    cell_view=DofCellView(cell_grid=self))
        self.show_array.current_row = 0
        self.show_array.configure_traits(kind='live')

class DofGridSlice(CellGridSlice):

    dof_grid = WeakRef(DofCellGrid)

    def __init__(self, dof_grid, **args):
        self.dof_grid = dof_grid
        super(DofGridSlice, self).__init__(**args)

    cell_grid = Property()

    def _get_cell_grid(self):
        return self.dof_grid.cell_grid

    dofs = Property

    def _get_dofs(self):
        _, idx2 = self.idx_tuple
        return self.dof_grid.cell_dof_map[
            np.ix_(
                self.elems,
                self.cell_grid.grid_cell[idx2]
            )
        ]

#-----------------------------------------------------------------------
# View a single cell instance
#-----------------------------------------------------------------------


class DofCellView(CellView):

    '''View a single cell instance.
    '''
    # implements(ICellView)

    elem_dofs = Array

    def set_cell_traits(self):
        '''Set the trait values for the current cell_idx
        '''
        self.elem_dofs = self.cell_grid.get_cell_dofs(self.cell_idx)


    def _get_cell_mvpoints(self):
        return self.cell_grid.get_cell_mvpoints(self.cell_idx)

    def _get_cell_labels(self):
        cell_dofs = self.cell_grid.get_cell_dofs(self.cell_idx)
        shape = cell_dofs.shape
        if shape[1] < 3:
            cd = np.zeros((shape[0], 3))
            cd[:, :shape[1]] = cell_dofs
            return cd
        else:
            return cell_dofs

    def redraw(self):
        if self.draw_cell:
            self.mvp_elem_labels.redraw(label_mode='label_vectors')

