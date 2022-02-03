
from traits.api \
    import HasTraits, Array, \
    on_trait_change, Instance, WeakRef, Int, \
    Interface, provides

#------------------------------------------------------------------------
# Source of the data for array view
#------------------------------------------------------------------------
class ICellArraySource(Interface):

    '''Object representing a structured 1,2,3-dimensional cell grid.
    '''
    pass

#------------------------------------------------------------------------
# Cell view interface
#------------------------------------------------------------------------


class ICellView(Interface):

    '''Interface of the general cell view.
    '''

    def set_cell_traits(self):
        '''Adapt the view to the newly set cell_idx.
        '''
        raise NotImplementedError

    def redraw(self):
        '''Redraw the graphical representation of the cell 
        in the mayavi pipeline.
        '''
        raise NotImplementedError

#------------------------------------------------------------------------
# Default implementation of the cell view
#------------------------------------------------------------------------


@provides(ICellView)
class CellView(HasTraits):

    '''Get the element numbers.
    '''

    cell_idx = Int(-1)

    cell_grid = WeakRef(ICellArraySource)

    def set_cell(self, cell_idx):
        '''Method to be overloaded by subclasses. The subclass 
        can fetch the data required from the cell_grid 
        '''
        self.cell_idx = cell_idx
        self.set_cell_traits()

    def set_cell_traits(self):
        '''Specialize this function to fetch the cell data from 
        the array source.
        '''
        pass

    def redraw(self):
        '''No plotting defined by default'''
        pass

class CellArray(HasTraits):

    data = Array

    cell_view = Instance(ICellView)

    def _cell_view_default(self):
        return CellView()

    current_row = Int(-1)

    @on_trait_change('current_row')
    def redraw(self):
        self.cell_view.redraw()

    @on_trait_change('current_row')
    def _display_current_row(self):
        if self.current_row != -1:
            self.cell_view.set_cell(self.current_row)

