r'''

Simulator implementation
========================

'''
import bmcs_utils.api as bu

from traits.api import \
    Instance, on_trait_change, Str, \
    Property, cached_property, Bool, List, provides
from ibvpy.view import BMCSTreeNode, itags_str, IBMCSModel

import traits.api as tr

from .i_tloop import ITLoop
from .i_tstep import ITStep
from .tline_mixin import TLineMixIn

@provides(IBMCSModel)
class Simulator(BMCSTreeNode, TLineMixIn):
    r'''Base class for simulators included in the BMCS Tool Suite.
    It implements the state dependencies within the simulation tree.
    It handles also the communication between the simulation and
    the user interface in several modes of interaction.
    '''
    name = 'simulator'

    @tr.observe('state_changed')
    def _model_structure_changed(self, event=None):
        self.tloop.restart = True

    #=========================================================================
    # TIME LOOP
    #=========================================================================

    tloop = Property(Instance(ITLoop), depends_on='state_changed')
    r'''Time loop constructed based on the current model.
    '''
    @cached_property
    def _get_tloop(self):
        return self.tstep.tloop_type(tstep=self.tstep,
                                     tline=self.tline)

    def __init__(self, tstep, *args, **kw):
        self.tstep = tstep
        super(Simulator, self).__init__(*args, **kw)

    tstep = tr.WeakRef(ITStep)

    hist = tr.Property

    def _get_hist(self):
        return self.tstep.hist

    def run(self):
        r'''Run a thread if it does not exist - do nothing otherwise
        '''
        self.tloop()
        return

    interrupt = tr.DelegatesTo('tloop')

    def reset(self):
        self.tloop.reset()

    ipw_view = bu.View(
        run_method='tloop',
        reset_method='reset',
        interrupt_var='interrupt',
        time_var = 't',
        time_max='t_max',
    )
