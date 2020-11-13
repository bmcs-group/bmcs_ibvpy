
from traits.api import \
    File, HasStrictTraits, \
    Str, Enum, Property, WeakRef, \
    ToolbarButton, on_trait_change


class RTrace(HasStrictTraits):
    name = Str('unnamed')
    record_on = Enum('update', 'iteration')
    clear_on = Enum('never', 'update')
    save_on = Enum(None)

    rmgr = WeakRef(trantient=True)
    sd = WeakRef(trantient=True)

    # path to directory to store the data
    dir = Property

    def _get_dir(self):
        return self.rmgr.dir

    # path to the file to store the data
    file = File

    def setup(self):
        '''Prepare the tracer for recording.
        '''
        pass

    def close(self):
        '''Close the tracer - save its values to file.
        '''
        pass

    refresh_button = ToolbarButton('Refresh',
                                   style='toolbar',
                                   trantient=True)

    @on_trait_change('refresh_button')
    def refresh(self, event=None):
        self.redraw()

    def add_current_values(self, sctx, U_k, t, *args, **kw):
        pass

    # TODO: to avoid class checking in rmngr - UGLY
    def add_current_displ(self, sctx, t, U_k):
        pass

    def register_mv_pipelines(self, e):
        '''
        Eventually register pipeline components within the mayavi sceen.

        do nothing by default
        '''
