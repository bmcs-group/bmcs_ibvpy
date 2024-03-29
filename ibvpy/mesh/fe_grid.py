
from functools import reduce

from ibvpy.fets.i_fets import IFETSEval
from ibvpy.mesh.cell_grid.cell_array import ICellView, CellView, CellArray, ICellArraySource
from ibvpy.mesh.cell_grid.cell_grid import CellGrid
from ibvpy.mesh.cell_grid.cell_spec import CellSpec
from ibvpy.mesh.cell_grid.dof_grid import DofCellGrid, DofCellView
from ibvpy.mesh.cell_grid.geo_grid import GeoCellGrid, GeoCellView
from numpy import copy, zeros, array_equal
from traits.api import \
    Instance, Array, Int, on_trait_change, Property, cached_property, \
    List, Button, HasTraits, provides, WeakRef, Float,  \
    Callable, Str, Event
from .fe_grid_activation_map import FEGridActivationMap
from .fe_grid_idx_slice import FEGridIdxSlice
from .fe_grid_ls_slice import FEGridLevelSetSlice
from .fe_grid_node_slice import ISliceProperty
from .i_fe_uniform_domain import IFEUniformDomain



class MElem(HasTraits):

    '''
    Finite element spatial representation.
    '''
    point_X_arr = Array
    point_x_arr = Array
    dofs = Array

    def get_X_mtx(self):
        '''
        Index mapping from the global array of coordinates.
        '''
        return self.point_X_arr

    def get_x_mtx(self):
        '''
        Index mapping from the global array of coordinates.
        '''
        return self.point_x_arr

    def get_dof_map(self):
        '''
        Return the dof map for the current element as a list
        '''
        return self.dofs

    def __str__(self):
        return 'points:\n%s\ndofs %s' % (self.point_X_arr, self.dofs)


@provides(ICellArraySource, IFEUniformDomain, ICellView)
class FEGrid(FEGridActivationMap):

    '''Structured FEGrid consisting of potentially independent
    dof_grid and geo_grid.

    For isoparametric element formulations, the dof_grid and geo_grid may
    share a single cell_grid to save memory.

    Structure of the grid
    ---------------------
    The structure of the grid is defined at two levels:
    1) within a cell specify the distribution of points
    (for dof_r and for geo_r).

    2) the cells are repeated in respective dimension by n_elem
    number of elements

    Services
    --------
    1) For a given element number return the nodal coordinates respecting
       the specification in the geo_r
    2) For a given element number return the array of dof numbers respecting
       the specification in the geo_r
    3) For a given CellSpec return a CellGrid respecting the geometric
       parameters of the FEGrid (applicable for response trace fields
       with finer distribution of nodes.
    4) For a given subdivision of the cell return CellGrid usable as
       visualization field.(where to specify the topology? - probably in
       the CellSpec?
    '''

    # Grid geometry specification
    #
    coord_min = Array(Float, value=[0., 0., 0.])
    coord_max = Array(Float, value=[1., 1., 1.])

    geo_transform = Callable

    # number of elements in the individual dimensions
    shape = Array(Int, value=[1, 1, 1], changes_ndofs=True)

    changed_structure = Event

    @on_trait_change('+changed_structure')
    def set_changed_structure(self):
        self.changed_structure = True

    _tree_label = 'subgrid'

    fets_eval = Instance(IFETSEval)

    # identifier within the refinement level
    idx = Int()

    # links within the dependency
    prev_grid = WeakRef('ibvpy.mesh.fe_grid.FEGrid', allow_none=True)
    next_grid = WeakRef('ibvpy.mesh.fe_grid.FEGrid', allow_none=True)

    _name = Str('')
    name = Property

    def _get_name(self):
        '''Return the name within the level
        '''
        if self._name == '':
            return 'grid ' + str(self.idx)
        else:
            return self._name

    def _set_name(self, value):
        self._name = value

    def __repr__(self):
        return self.name

    # dof offset within the global enumeration
    dof_offset = Property(
        Int, depends_on='prev_grid.dof_offset,level.dof_offset')
    # cached_property

    def _get_dof_offset(self):
        if self.prev_grid:
            return self.prev_grid.dof_offset + self.prev_grid.n_dofs
        elif self.level:
            return self.level.dof_offset
        else:
            return 0

    _level = WeakRef(
        'ibvpy.mesh.fe_refinement_grid.FERefinementGrid', links_changed=True)
    # parent domain
    level = Property

    def _set_level(self, value):
        'reset the parent of this domain'
        if self._level and self._level != value:
            # remove the grid from the level
            self._level._fe_subgrids.remove(self)
            # unlink it from previous and next
            if self.prev_grid:
                self.prev_grid.next_grid = self.next_grid
            if self.next_grid:
                self.next_grid.prev_grid = self.prev_grid
        # set the new parent
        self._level = value
        # link the subgrid within the level
#        prev_grid = self._level.last_subgrid
#        if prev_grid:
#            self.prev_grid = prev_grid
#            prev_grid.next_grid = self
        # add to the level
        self.idx = len(self._level._fe_subgrids)
        self._level._fe_subgrids.append(self)

    def _get_level(self):
        return self._level

    def __del__(self):
        ''' Release the grid from the dependency structure
        '''
        if self.prev_grid:
            self.prev_grid.next_grid = self.next_grid
        if self.next_grid:
            self.next_grid.prev_grid = self.prev_grid

    dots = Property

    @cached_property
    def _get_dots(self):
        '''Construct and return a new instance of domain
        time stepper.
        '''
        return self.fets_eval.dots_class(sdomain=self)

    dof_r = Property

    def _get_dof_r(self):
        return self.fets_eval.dof_r

    geo_r = Property

    def _get_geo_r(self):
        return self.fets_eval.geo_r

    n_nodal_dofs = Property

    def _get_n_nodal_dofs(self):
        return self.fets_eval.n_nodal_dofs

    #-------------------------------------------------------------------------
    # Derived properties
    #-------------------------------------------------------------------------
    # dof point distribution within the cell converted into the CellSpec format
    # CellSpec can derive the shape of the single grid cell, i.e.
    # the number of points specified in the individual directions.
    #
    dof_grid_spec = Property(Instance(CellSpec), depends_on='fets_eval.dof_r')

    def _get_dof_grid_spec(self):
        return CellSpec(node_coords=self.dof_r)

    # geo point distribution within the cell ... the same as above ...
    #
    geo_grid_spec = Property(Instance(CellSpec), depends_on='fets_eval.geo_r')

    def _get_geo_grid_spec(self):
        return CellSpec(node_coords=self.geo_r)

    # dof_grid is represented by the DofCellGrid class maintaining
    # the internal data structure to retrieve the element mappings
    # within the grid.
    #
    dof_grid = Property(Instance(DofCellGrid),
                        depends_on='fets_eval.dof_r,shape,coord_min,coord_max')

    def _get_dof_grid(self):
        return self._grids[0]

    dof_vertex_X_grid = Property

    def _get_dof_vertex_X_grid(self):
        return self.dof_grid.cell_grid.vertex_X_grid

    intg_elem_grid = Property

    def _get_intg_elem_grid(self):
        return self.dof_grid.cell_grid.cell_idx_grid

    ls_mask = Property

    def _get_ls_mask(self):
        return zeros(self.intg_elem_grid.shape, dtype=bool)

    ls_elem_grid = Property

    def _get_ls_elem_grid(self):
        return self.dof_grid.cell_grid.cell_idx_grid

    intg_elem_grid_dof_map = Property

    def _get_intg_elem_grid_dof_map(self):
        return self.dof_grid.cell_grid_dof_map

    def get_cell_point_X_arr(self, elem):
        return self.geo_grid.get_cell_point_X_arr(elem)

    # geo_grid is represented by the GeoCellGrid class maintaining
    # the internal data structure to retrieve the element mappings
    # within the grid.
    #
    geo_grid = Property(Instance(GeoCellGrid),
                        depends_on='fets_eval.geo_r,shape,coord_min,coord_max')

    def _get_geo_grid(self):
        return self._grids[1]

    I = Property

    def _get_I(self):
        return ISliceProperty(fe_grid=self)

    def __getitem__(self, idx):
        if isinstance(idx, tuple) or isinstance(idx, int):
            return FEGridIdxSlice(fe_grid=self, grid_slice=idx)
        elif isinstance(idx, str):
            return FEGridLevelSetSlice(fe_grid=self, ls_function_eval=idx)
        else:
            raise TypeError(type(idx), 'is unsupported type for slicing')

    #-------------------------------------------------------------
    # Implement the FEDomain interface
    #-------------------------------------------------------------

    n_dofs = Property(Int)

    def _get_n_dofs(self):
        return self.dof_grid.n_dofs

    def get_cell_offset(self, idx_tuple):
        return self.dof_grid.get_cell_offset(idx_tuple)
    #-------------------------------------------------------------------------
    # Implement the parent interface
    #-------------------------------------------------------------------------

    def deactivate(self, idx):
        '''Exclude the specified element from the integration.
        '''
        if isinstance(idx, tuple):
            self.inactive_elems.append(self.dof_grid.get_cell_offset(idx))
        elif isinstance(idx, int):
            self.inactive_elems.append(idx)
        if self.level:
            self.level.set_changed_structure()

    #-------------------------------------------------------------
    # Implement the IFEUniformDomain interface
    #-------------------------------------------------------------

    # Full elem dof map ignoring masked elements.
    ls_elem_dof_map = Property

    def _get_ls_elem_dof_map(self):
        return self.dof_grid.elem_dof_map

    # Full elem dof map ignoring masked elements.
    elem_dof_map_unmasked = Property

    def _get_elem_dof_map_unmasked(self):
        return self.dof_grid.elem_dof_map

    elem_dof_map = Property(
        depends_on='fets_eval.dof_r,shape,dof_offset, changed_structure')

    def _get_elem_dof_map(self):
        elem_dof_map = self.dof_grid.elem_dof_map[self.activation_map, :]
        return elem_dof_map

    elem_X_map = Property(
        depends_on='fets_eval.geo_r,shape,coord_min,coord_max,n_nodal_dofs, changed_structure')

    @cached_property
    def _get_elem_X_map(self):
        elem_X_map = self.geo_grid.elem_X_map[self.activation_map, :].copy()
        return elem_X_map

    ls_elem_X_map = Property

    def _get_ls_elem_X_map(self):
        return self.geo_grid.elem_X_map.copy()

    elem_X_map_unmasked = Property

    def _get_elem_X_map_unmasked(self):
        return self.geo_grid.elem_X_map.copy()

    elem_x_map = Property(
        depends_on='fets_eval.geo_r,shape,coord_min,coord_max,n_nodal_dofs, changed_structure')

    @cached_property
    def _get_elem_x_map(self):
        elem_x_map = self.geo_grid.elem_x_map[self.activation_map, :].copy()
        return copy(elem_x_map)

    elem_x_map_unmasked = Property

    def _get_elem_x_map_unmasked(self):
        return self.geo_grid.elem_x_map.copy()

    ls_elem_x_map = Property

    def _get_ls_elem_x_map(self):
        return self.geo_grid.elem_x_map.copy()

    X_dof_arr = Property

    def _get_X_dof_arr(self):
        return self.dof_grid.cell_grid.point_X_arr

    elem_node_map = Property

    def _get_elem_node_map(self):
        return self.dof_grid.cell_grid.cell_node_map

    n_grid_elems = Property

    def _get_n_grid_elems(self):
        return reduce(lambda x, y: x * y, self.shape)

    n_active_elems = Property

    def _get_n_active_elems(self):
        return self.elem_dof_map.shape[0]

    elements = Property(
        List, depends_on='fets_eval.dof_r,fets_eval.geo_r,shape+,coord_min,coord_max,fets_eval.n_nodal_dofs,dof_offset, changed_structure')

    @cached_property
    def _get_elements(self):
        return [MElem(dofs=dofs,
                      point_X_arr=point_X_arr,
                      point_x_arr=point_x_arr)
                for dofs, point_X_arr, point_x_arr
                in zip(self.elem_dof_map,
                       self.elem_X_map,
                       self.elem_x_map)]

    dof_Eid = Property
    '''Mapping of Element, Node, Dimension -> DOF 
    '''

    def _get_dof_Eid(self):
        return self.dof_grid.dof_Eid

    dofs = Property

    def _get_dofs(self):
        return self.dof_grid.dofs

    I_Ei = Property
    '''For a given element and its node number return the global index
    of the node'''

    def _get_I_Ei(self):
        return self.geo_grid.cell_grid.cell_node_map

    X_Id = Property()
    '''Array of containing the coordinate d \in (0,1,3)
    of a node I
    '''

    def _get_X_Id(self):
        return self.geo_grid.point_X_arr

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
        out_grid = zeros(out_grid_shape)

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
        X_el = self.geo_grid.elem_X_map
        # test call to the function with single output - to get the shape of
        # the result.
        out_single = fn(ip_mask[0], X_el[0])
        out_grid_shape = (X_el.shape[0], ip_mask.shape[0],) + out_single.shape
        out_grid = zeros(out_grid_shape)

        for el in range(X_el.shape[0]):
            for ip in range(ip_mask.shape[0]):
                out_grid[el, ip, ...] = fn(ip_mask[ip], X_el[el])

        return out_grid

    #-------------------------------------------------------------
    # Queries
    #-------------------------------------------------------------
    # The delegation had to be replaced with explicit calls to functions.
    # The reason is lost binding if a method
    #
    # fe_domain.get_left_dofs
    #
    # If the discretization parameter changed, then a new dof_grid was generated
    # but the call fe_domain.get_left_dofs went to the original dof_grid.
    #
    # get_all_dofs   = DelegatesTo('dof_grid')
    def get_all_dofs(self): return self.dof_grid.get_all_dofs()
    # get_left_dofs   = Delegate('dof_grid')

    def get_left_dofs(self): return self.dof_grid.get_left_dofs()
    # get_right_dofs  = Delegate('dof_grid')

    def get_right_dofs(self): return self.dof_grid.get_right_dofs()
    # get_top_dofs    = Delegate('dof_grid')

    def get_top_dofs(self): return self.dof_grid.get_top_dofs()
    # get_bottom_dofs = Delegate('dof_grid')

    def get_bottom_dofs(self): return self.dof_grid.get_bottom_dofs()
    # get_front_dofs  = Delegate('dof_grid')

    def get_front_dofs(self): return self.dof_grid.get_front_dofs()
    # get_back_dofs   = Delegate('dof_grid')

    def get_back_dofs(self): return self.dof_grid.get_back_dofs()
    # get_bottom_left_dofs   = Delegate('dof_grid')

    def get_bottom_left_dofs(self): return self.dof_grid.get_bottom_left_dofs()
    # get_bottom_front_dofs   = Delegate('dof_grid')

    def get_bottom_front_dofs(
        self): return self.dof_grid.get_bottom_front_dofs()
    # get_bottom_back_dofs   = Delegate('dof_grid')

    def get_bottom_back_dofs(self): return self.dof_grid.get_bottom_back_dofs()
    # get_top_left_dofs   = Delegate('dof_grid')

    def get_top_left_dofs(self): return self.dof_grid.get_top_left_dofs()
    # get_bottom_right_dofs  = Delegate('dof_grid')

    def get_bottom_right_dofs(
        self): return self.dof_grid.get_bottom_right_dofs()
    # get_top_right_dofs  = Delegate('dof_grid')

    def get_top_right_dofs(self): return self.dof_grid.get_top_right_dofs()
    # get_bottom_middle_dofs  = Delegate('dof_grid')

    def get_bottom_middle_dofs(
        self): return self.dof_grid.get_bottom_middle_dofs()
    # get_top_middle_dofs  = Delegate('dof_grid')

    def get_top_middle_dofs(self): return self.dof_grid.get_top_middle_dofs()
    # get_left_middle_dofs  = Delegate('dof_grid')

    def get_left_middle_dofs(self): return self.dof_grid.get_left_middle_dofs()
    # get_right_middle_dofs  = Delegate('dof_grid')

    def get_right_middle_dofs(
        self): return self.dof_grid.get_right_middle_dofs()
    # get_left_front_bottom_dof  = Delegate('dof_grid')

    def get_left_front_bottom_dofs(
        self): return self.dof_grid.get_left_front_bottom_dofs()
    # get_left_front_middle_dof  = Delegate('dof_grid')

    def get_left_front_middle_dofs(
        self): return self.dof_grid.get_left_front_middle_dofs()

    #-------------------------------------------------------------
    # LevelSet Methods
    #-------------------------------------------------------------

    def get_lset_subdomain(self, lset_function):
        '''@TODO - implement the subdomain selection method
        '''
        raise NotImplementedError

    def get_boundary(self, side=None):
        '''@todo:-implement the boundary extraction
        '''
        raise NotImplementedError

    def get_interior(self):
        '''@todo:-implement the boundary extraction
        '''
        raise NotImplementedError

    def get_ls_mask(self, ls_mask_function):
        '''
        Return a boolean array indicating masked nodes.
        '''
        return self.geo_grid.get_ls_mask(ls_mask_function)

    def get_intersected_elems(self, ls_function, ls_limits=None):
        '''
        Return elems intersected by specified domain.

        Requirements on the e_domain
        - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        The e_domain must be at most n - 1 dimensionality of the grid.

        Two methods must be supported by the e_domain
        1 ) The e_domain must be discretizable within the current grid. That means
           it should return arrays with points on grid lines that intersect the
           e_domain's boundaries
        2) With the list of intersections, it is possible to identify the intersected
           elements.
        These two methods would avoid a full search through all the grid points.

        Applicable:
        -----------
        Boundary conditions including the shape function coefficient
        Local - element enrirchments (XFEM)

        Method result
        -------------
        The method returns the

        '''
        return self.geo_grid.get_intersected_cells(ls_function, ls_limits)

    def get_negative_elems(self, ls_function):
        return self.geo_grid.get_negative_cells(ls_function)

    def get_affected_elems(self, e_domain):
        '''
        3) It should have a notion of inside / outside to decide whether or not
           a point is to be included or not.
        '''

    def get_enclosed_nodes(self, e_domain):
        '''
        Get elements that are inside of the e_domain.
        The elements are not intersected by the boundaries of the e_domain.

        The e_domain must be at least the same dimension as the grid.
        The e_domain must have an operator inside/outside to decide about
        whether or not a point is included.

        The method returns an array of element numbers within the specified e_domain.
        '''

    def get_subgrid(self, bounding_box):
        '''
        Return subgrid with base_node number
        '''

    def get_ls_value(self, X_pnt):
        # @TODO: make it work for 3d
        return self.ls_function.level_set_fn(X_pnt[0], X_pnt[1])

    _grids = Property(
        depends_on='fets_eval.dof_r,fets_eval.geo_r,shape+,coord_min+,coord_max+,fets_eval.n_nodal_dofs,dof_offset,geo_transform')

    @cached_property
    def _get__grids(self):
        '''Regenerate grids based on the specification
        '''
        # Check if the specifiers for both grids are identical
        #
        cell_grid = CellGrid(
            grid_cell_spec=self.dof_grid_spec,
            geo_transform=self.geo_transform,
            shape=self.shape,
            coord_min=self.coord_min,
            coord_max=self.coord_max
        )
        _xdof_grid = DofCellGrid(
            cell_grid=cell_grid,
            n_nodal_dofs=self.n_nodal_dofs,
            dof_offset=self.dof_offset
        )

        # if self.dof_r.all() != self.geo_r.all():
        #
        # @todo check whether this is universally valid (tolerance of coordinate values)
        #
        if not array_equal(self.dof_r, self.geo_r):
            #        if (self.dof_r.shape[0] != self.geo_r.shape[0])or\
            #           ((self.dof_r.shape == self.geo_r.shape) and\
            #            linalg.norm(self.dof_r - self.geo_r) > 1.e-3) :
            # If the geometry grid differs from the dof_grid specifier
            # construct a separate cell grid, otherwise reuse the dof_grid
            cell_grid = CellGrid(grid_cell_spec=self.geo_grid_spec,
                                 geo_transform=self.geo_transform,
                                 shape=self.shape,
                                 coord_min=self.coord_min,
                                 coord_max=self.coord_max)
        _xgeo_grid = GeoCellGrid(cell_grid=cell_grid)
        return (_xdof_grid, _xgeo_grid)


class FECellView(CellView):
    geo_view = Instance(GeoCellView)

    def _geo_view_default(self):
        return GeoCellView()

    dof_view = Instance(DofCellView)

    def _dof_view_default(self):
        return DofCellView()

    @on_trait_change('cell_grid')
    def _reset_view_links(self):
        self.geo_view.cell_grid = self.cell_grid.geo_grid
        self.dof_view.cell_grid = self.cell_grid.dof_grid

    def set_cell_traits(self):
        self.dof_view.cell_idx = self.cell_idx
        self.dof_view.set_cell_traits()
        self.geo_view.cell_idx = self.cell_idx
        self.geo_view.set_cell_traits()

    def redraw(self):
        self.dof_view.redraw()
        self.geo_view.redraw()

