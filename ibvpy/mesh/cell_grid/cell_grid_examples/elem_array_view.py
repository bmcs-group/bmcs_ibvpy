
#-- Imports --------------------------------------------------------------------

from os.path \
    import join, dirname
    
from numpy \
    import sqrt
    
from numpy.random \
    import random

from traits.api \
    import HasTraits, Property, Array, Any, Event, \
    on_trait_change, Instance, WeakRef, Int, Str, Bool, Trait

from ibvpy.view.plot3d.mayavi_util.pipelines import \
    MVPolyData, MVPointLabels, MVStructuredGrid    

class ElemView(HasTraits):
    '''Get the element numbers.
    '''
    elem_num = Int(-1)
    elem_coords = Array

#-- ElemArrayView Class Definition -------------------------------------------------

class ElemArrayView ( HasTraits ):

    data = Array
    rt_domain = WeakRef
    
    
    mvp_elem_labels = Trait( MVPointLabels )
    def _mvp_elem_labels_default(self):
        return MVPointLabels( name = 'Geo node numbers', 
                                  points = self._get_current_elem_coords,
                                  scalars = self._get_current_elem_numbers,
                                  #color = (0.254902,0.411765,0.882353)
                                  color = (0.15,0.85,0.45))
                                 
        
    mvp_elem_geo = Trait( MVPolyData )
    def _mvp_elem_geo_default(self):
        return MVPolyData( name = 'Geo node numbers', 
                               points = self._get_current_elem_coords,
                               lines = self._get_current_elem_lines,
                               #color = (0.254902,0.411765,0.882353))
                               color= (0,55,0,75,0.0))
        
    
    show_elem = Bool(True)
            
    def _get_current_elem_coords(self):
        return self.rt_domain._get_elem_coords( self.current_row )
    
    def _get_current_elem_numbers(self):
        return self.data[self.current_row]
    
    def _get_current_elem_lines(self):
        line_map = self.rt_domain.grid_cell_spec.cell_lines
        return line_map
    
    current_row = Int(-1)
    @on_trait_change('current_row')
    def redraw(self):
        if self.show_elem:
            self.mvp_elem_labels.redraw('label_scalars')
            self.mvp_elem_geo.redraw()
    
    elem_view = Instance( ElemView )
    def _elem_view_default(self):
        return ElemView()
    
    @on_trait_change('current_row')
    def _display_current_row(self):
        if self.current_row != -1:
            self.elem_view.elem_num = self.current_row
            elem_coords = self.rt_domain._get_elem_coords( self.current_row )
            self.elem_view.elem_coords = elem_coords
