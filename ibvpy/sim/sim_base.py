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
    tree_node_list = List([])

    def _tree_node_list_default(self):
        return [
            self.tline,
        ]

    def _update_node_list(self):
        self.tree_node_list = [
            self.tline,
        ]

    title = Str

    desc = Str

    @on_trait_change(itags_str)
    def _model_structure_changed(self):
        self.tloop.restart = True

    #=========================================================================
    # TIME LOOP
    #=========================================================================

    tloop = Property(Instance(ITLoop), depends_on=itags_str)
    r'''Time loop constructed based on the current model.
    '''
    @cached_property
    def _get_tloop(self):
        return self.tstep.tloop_type(tstep=self.tstep,
                                     tline=self.tline)

    tstep = tr.WeakRef(ITStep)

    hist = tr.Property

    def _get_hist(self):
        return self.tstep.hist

    def run(self):
        r'''Run a thread if it does not exist - do nothing otherwise
        '''
        self.tloop()
        return

    def pause(self):
        self.tloop.paused = True

    def stop(self):
        self.tloop.restart = True

    ipw_view = bu.View(
        simulator='tloop',
        time_variable='t',
        time_max='t_max',
        reset_simulator='stop'
    )
