'''
Created on Mar 18, 2020

@author: rch
'''
import traits.api as tr
import bmcs_utils.api as bu

from .tline import TLine


class TLineMixIn(tr.HasTraits):
    #=========================================================================
    # TIME LINE
    #=========================================================================
    tline = bu.Instance(TLine)
    r'''Time line defining the time range, discretization and state,  
    '''

    def _tline_default(self):
        return TLine(
            time_change_notifier=self.time_changed,
            time_range_change_notifier=self.time_range_changed
        )

    def time_changed(self, time):
        if not(self.ui is None):
            self.ui.time_changed(time)

    def time_range_changed(self, tmax):
        self.tline.max = tmax
        if not(self.ui is None):
            self.ui.time_range_changed(tmax)

    def set_tmax(self, time):
        self.time_range_changed(time)

    t = tr.DelegatesTo('tline', 'val')
    t_max = tr.DelegatesTo('tline', 'max')
