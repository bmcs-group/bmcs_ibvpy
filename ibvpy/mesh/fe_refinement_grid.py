from traits.api import \
    Instance, Array, Int, on_trait_change, Property, cached_property, \
    List, Button, provides,  \
    Event
from ibvpy.fets.i_fets import IFETSEval
from ibvpy.mesh.cell_grid.cell_array import ICellArraySource
from ibvpy.mesh.cell_grid.cell_spec import CellSpec
import numpy as np

from .fe_grid import FEGrid, MElem
from .fe_refinement_level import FERefinementLevel
from .i_fe_uniform_domain import IFEUniformDomain


@provides(ICellArraySource, IFEUniformDomain)
class FERefinementGrid(FERefinementLevel):

    '''Subgrid derived from another grid domain.
    '''

    changed_structure = Event

    @on_trait_change('+changed_structure')
    def set_changed_structure(self):
        self.changed_structure = True

    #-----------------------------------------------------------------
    # Feature: domain time-stepper
    #-----------------------------------------------------------------

    _fets_eval = Instance(IFETSEval)
    # inherit the fets_eval from the parent. This does not necessarily
    # have to be the case - calls for delegate here - that can be overloaded.
    fets_eval = Property

    def _set_fets_eval(self, value):
        self._fets_eval = value

    def _get_fets_eval(self):
        if self._fets_eval == None:
            return self.parent.fets_eval
        else:
            return self._fets_eval

    dof_r = Property

    def _get_dof_r(self):
        return self.fets_eval.dof_r

    geo_r = Property

    def _get_geo_r(self):
        return self.fets_eval.geo_r

    geo_transform = Property

    def _get_geo_transform(self):
        # geo transform should be part of a FEPatchedGrid
        #
        return self.parent.fe_subgrids[0].geo_transform

    n_nodal_dofs = Property

    def _get_n_nodal_dofs(self):
        return self.fets_eval.n_nodal_dofs

    # @TODO - move this to the base class - FEDomainBase
    #-------------------------------------------------------------------------
    # Derived properties
    #-------------------------------------------------------------------------
    # dof point distribution within the cell converted into the CellSpec format
    # CellSpec can derive the shape of the single grid cell, i.e.
    # the number of points specified in the individual directions.
    #
    dof_grid_spec = Property(Instance(CellSpec), depends_on='dof_r')

    def _get_dof_grid_spec(self):
        return CellSpec(node_coords=self.dof_r)

    # geo point distribution within the cell ... the same as above ...
    #
    geo_grid_spec = Property(Instance(CellSpec), depends_on='geo_r')

    def _get_geo_grid_spec(self):
        return CellSpec(node_coords=self.geo_r)

    fine_cell_shape = Array(int, value=[2, 2, 2])

    def get_fine_ix(self, coarse_ix):
        return np.array(list(coarse_ix), dtype=int) * self.fine_cell_shape

    def get_bounding_box(self, coarse_ix):
        '''Get the corner coordinates of the parent cell
        '''
        # @todo: TEMPORARY this must be done for the patched grid shape
        # FEPatchedGrid
        #
        pgrid = self.parent.fe_subgrids[0].geo_grid.point_x_grid
        print('shape', pgrid.shape)
        coarse_ix = np.array(list(coarse_ix), dtype=int)
        coord_min = pgrid[(slice(0, pgrid.shape[0]),) + tuple(coarse_ix)]
        coord_max = pgrid[(slice(0, pgrid.shape[0]),) + tuple(coarse_ix + 1)]
        return coord_min, coord_max

    def get_fine_fe_domain(self, coarse_ix):
        # get it for the general ix_slice
        # access fe_subgrid using coarse slices? or geometrically?
        # probably using fine slices?
        coord_min, coord_max = self.get_bounding_box(coarse_ix)
        fe_grid = FEGrid(fets_eval=self.fets_eval,
                         # multiply with the coarse ix / slice
                         shape=self.fine_cell_shape,
                         geo_transform=self.geo_transform,
                         coord_min=coord_min,
                         coord_max=coord_max)
        fe_grid.level = self
        return fe_grid

    def get_lset_subdomain(self, lset_function):
        '''@TODO - implement the subdomain selection method
        '''
        pass

    def get_boundary(self, side=None):
        '''@todo: - implement the boundary extraction
        '''
        pass

    def get_interior(self):
        '''@todo: - implement the boundary extraction
        '''
        pass

    #-------------------------------------------------------------
    # FEDomain interface
    #-------------------------------------------------------------

    elem_dof_map = Property(Array)

    def _get_elem_dof_map(self):
        elem_dof_map = np.vstack([fe_subgrid.elem_dof_map
                                  for fe_subgrid in self.fe_subgrids])
        return elem_dof_map

    elem_dof_map_unmasked = Property(Array)

    def _get_elem_dof_map_unmasked(self):
        elem_dof_map = np.vstack([fe_subgrid.elem_dof_map_unmasked
                                  for fe_subgrid in self.fe_subgrids])
        return elem_dof_map

    n_elems = Property

    def _get_n_elems(self):
        return len(self.elem_dof_map_unmasked)

    # get the number of dofs in the subgrids
    #  - consider caching
    n_dofs = Property(Int)

    def _get_n_dofs(self):
        '''Total number of dofs'''
        last_fe_subgrid = self.last_subgrid
        if last_fe_subgrid:
            return last_fe_subgrid.dof_offset + last_fe_subgrid.n_dofs - self.dof_offset
        else:
            return 0

    elem_X_map = Property(Array)

    def _get_elem_X_map(self):
        '''Array with the point coordinates'''
        return np.vstack([fe_subgrid.elem_X_map
                          for fe_subgrid in self.fe_subgrids])

    elem_X_map_unmasked = Property(Array)

    def _get_elem_X_map_unmasked(self):
        '''Array with the point coordinates'''
        return np.vstack([fe_subgrid.elem_X_map_unmasked
                          for fe_subgrid in self.fe_subgrids])

    elem_x_map = Property(Array)

    def _get_elem_x_map(self):
        '''Array with the point coordinates'''
        return np.vstack([fe_subgrid.elem_x_map
                          for fe_subgrid in self.fe_subgrids])

    elem_x_map_unmasked = Property(Array)

    def _get_elem_x_map_unmasked(self):
        '''Array with the point coordinates'''
        return np.vstack([fe_subgrid.elem_x_map_unmasked
                          for fe_subgrid in self.fe_subgrids])

    dof_Eid = Property
    '''Mapping of Element, Node, Dimension -> DOF 
    '''

    def _get_dof_Eid(self):
        return np.vstack([fe_subgrid.dof_Eid
                          for fe_subgrid in self.fe_subgrids])

    dofs = Property
    ''' 
    '''

    def _get_dofs(self):
        return np.vstack([fe_subgrid.dofs
                          for fe_subgrid in self.fe_subgrids])

    I_Ei = Property(Array)
    '''For a given element and its node number return the global index
    of the node'''

    def _get_I_Ei(self):
        return np.vstack([fe_subgrid.I_Ei
                          for fe_subgrid in self.fe_subgrids])

    X_Id = Property(Array)

    def _get_X_Id(self):
        return np.vstack([fe_subgrid.X_Id
                          for fe_subgrid in self.fe_subgrids])

    def deactivate(self, idx):
        '''Deactivate the specified element.

        The idx is an expanded cell index on the fine grid.
        Note, that the refined grid does not really exist
        as array structure, only the subgrids are there.
        Their offsets are set so as to fit into the implicit
        grid with the fineness defined by the current
        refinement level.
        '''
        # identify the correct subgrid to propagate the request to
        #
        # raise an index error if the idx points into non-existing element
        # (non-refined one)
        #
        for fe_grid in self.fe_subgrids:
            fe_grid.deactivate(idx)

    def reactivate(self, idx):
        '''Deactivate the specified element.

        The idx is an expanded cell index on the fine grid.
        Note, that the refined grid does not really exist
        as array structure, only the subgrids are there.
        Their offsets are set so as to fit into the implicit
        grid with the fineness defined by the current
        refinement level.
        '''
        raise NotImplementedError

#    fe_subgrids_params = Property
#    def _get_fe_subgrids_params( self ):
#        return zip( self.elem_dof_enumeration[6],
#                    self.elem_dof_enumeration[7],
#                    self.elem_dof_enumeration[5] )

    _fe_subgrids = List
    fe_subgrids = Property

    def _get_fe_subgrids(self):
        return self._fe_subgrids + self.elem_dof_enumeration[5]

    last_subgrid = Property

    def _get_last_subgrid(self):
        '''Return the last subgrids in order to establish the links.
        '''
        if len(self.fe_subgrids) > 0:
            return self.fe_subgrids[-1]
        else:
            return None

    def subgrids(self):
        #no_parents = [ None for i in range( len( self._fe_subgrids ) ) ]
        return list(zip(list(self.refinement_dict.keys()), self.fe_subgrids))

    elem_dof_enumeration = Property(
        depends_on='changed_structure,+changed_geometry')

    @cached_property
    def _get_elem_dof_enumeration(self):
        '''Array with the dof enumeration
        '''
        #p_list, args_list, fe_domain_list = [], [], []
        for p, refinement_args in list(self.refinement_dict.items()):

            fe_domain = self.get_fine_fe_domain(p)
            #fe_domain_list.append( fe_domain )
            #p_list.append( p )
            #args_list.append( refinement_args )

        prev_grid = None
        for fe_grid in self._fe_subgrids:
            if prev_grid:
                fe_grid.prev_grid = prev_grid
                prev_grid.next_grid = fe_grid
            prev_grid = fe_grid

        return [], -1, \
            [], [], [], \
            [], [], []
        #fe_domain_list, p_list, args_list

    #--------------------------------------------------------------
    # Activation should be implicit (include pending)
    #--------------------------------------------------------------
    # activation map
    #
    # get boolean array with inactive elements indicated by False
    activation_map = Property(depends_on='changed_structure')

    @cached_property
    def _get_activation_map(self):
        '''@TODO - react to changes in parent'''
        return np.hstack([subgrid.activation_map for subgrid in self.fe_subgrids])

    # get indices of all active elements
    idx_active_elems = Property(depends_on='changed_structure')

    @cached_property
    def _get_idx_active_elems(self):
        return np.arange(self.n_grid_elems)[self.activation_map]

    n_grid_elems = Property

    def _get_n_grid_elems(self):
        '''Total number of elements in the subgrid'''
        n_elem_arr = np.array(
            [subgrid.n_grid_elems for subgrid in self.fe_subgrids], dtype='int')
        return sum(n_elem_arr)

    n_active_elems = Property(
        List, depends_on='changed_structure,+changed_geometry,+changed_formulation,+changed_context')

    @cached_property
    def _get_n_active_elems(self):
        n_elem_arr = np.array(
            [subgrid.n_active_elems for subgrid in self.fe_subgrids], dtype='int')
        return sum(n_elem_arr)

    elements = Property(
        List, depends_on='changed_structure,+changed_geometry,+changed_formulation,+changed_context')

    @cached_property
    def _get_elements(self):
        '''The active list of elements to be included in the spatial integration'''
        # only active elements are returned
        return [MElem(dofs=dofs, point_X_arr=point_X_arr, point_x_arr=point_x_arr)
                for dofs, point_X_arr, point_x_arr in zip(self.elem_dof_map,
                                                          self.elem_X_map, self.elem_x_map)]

    def apply_on_ip_grid(self, fn, ip_mask):
        '''
        Apply the function fn over the first dimension of the array.
        @param fn: function to apply for each ip from ip_mask and each element.
        @param ip_mask: specifies the local coordinates within the element.
        '''
        X_el = self.elem_X_map
        # test call to the function with single output - to get the shape of
        # the result.
        out_single = fn(ip_mask[0], X_el[0])
        out_grid_shape = (X_el.shape[0], ip_mask.shape[0],) + out_single.shape
        out_grid = np.zeros(out_grid_shape)

        for el in range(X_el.shape[0]):
            for ip in range(ip_mask.shape[0]):
                out_grid[el, ip, ...] = fn(ip_mask[ip], X_el[el])

        return out_grid

    def apply_on_ip_grid_unmasked(self, fn, ip_mask):
        '''
        Apply the function fn over the first dimension of the array.
        @param fn: function to apply for each ip from ip_mask and each element.
        @param ip_mask: specifies the local coordinates within the element.
        '''
        X_el = self.elem_X_map_unmasked
        # test call to the function with single output - to get the shape of
        # the result.
        out_single = fn(ip_mask[0], X_el[0])
        out_grid_shape = (X_el.shape[0], ip_mask.shape[0],) + out_single.shape
        out_grid = np.zeros(out_grid_shape)

        for el in range(X_el.shape[0]):
            for ip in range(ip_mask.shape[0]):
                out_grid[el, ip, ...] = fn(ip_mask[ip], X_el[el])

        return out_grid

    #-------------------------------------------------------------
    # Visual introspection
    #-------------------------------------------------------------
    refresh_button = Button('Draw')

    @on_trait_change('refresh_button')
    def redraw(self):
        '''Redraw the point grid.
        @TODO
        '''
        for fe_grid in self.fe_subgrids:
            fe_grid.redraw()

