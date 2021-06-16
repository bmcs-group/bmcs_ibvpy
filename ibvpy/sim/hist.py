'''
Created on Dec 17, 2018

@author: rch
'''

from traits.api import \
    provides, List, Dict, WeakRef, Property, cached_property
from ibvpy.view.plot2d import Vis2D
import numpy as np
import copy
from .i_hist import IHist


@provides(IHist)
class Hist(Vis2D):
    '''Object storing and managing the history of calculation.
    '''

    tstep_source = WeakRef

    timesteps = List()

    U_list = List()
    F_list = List()

    vis_record = Dict

    record_dict = Property(
        Dict, depends_on='tstep_source.record, tstep_source.record_items'
    )

    @cached_property
    def _get_record_dict(self):
        ts = self.tstep_source
        for vis in self.vis_record.values():
            #vis.sim = self.tstep_source.sim
            vis.tstep = ts
        return {key: vis for key, vis in self.vis_record.items()}

    def __getitem__(self, key):
        return self.record_dict[key]

    state_vars = List()

    def init_state(self):
        self.timesteps = []
        self.U_list = []
        self.F_list = []
        self.state_vars = []
        # self.Eps_t = {}
        for vis in self.record_dict.values():
            vis.setup()

    def record_timestep(self, t, U, F,
                        state_vars=None):
        '''Add the time step and record the 
        corresponding state of the model.
        '''
        self.timesteps.append(t)
        self.U_list.append(np.copy(U))
        self.F_list.append(np.copy(F))
        self.state_vars.append(copy.deepcopy(state_vars))
        for vis in self.record_dict.values():
            vis.update()

    # def insert_Eps(self, Eps):
    #     for Eps_d in Eps:
    #
    #     self.Eps_t
    #     keys = self.Eps_t[0,0].keys()
    #     Eps_t = self.Eps_t[idx,0]
    #     if reduce_dim:
    #         eps_tEms = eps_tEms[0,...]
    #         Eps_Dt = Eps_t[0]
    #     else:
    #         Eps_Dt = {
    #             key: np.array([Eps[key] for i, Eps in enumerate(Eps_t)], dtype=np.float_)
    #             for key in keys
    #         }

    U_t = Property(depends_on='timesteps_items')

    @cached_property
    def _get_U_t(self):
        return np.array(self.U_list, dtype=np.float_)

    F_t = Property(depends_on='timesteps_items')

    @cached_property
    def _get_F_t(self):
        return np.array(self.F_list, dtype=np.float_)

    # todo: capture the mapping between multidomain state representation
    #       and a time slice during evaluation
    Eps_t = Property(depends_on='timesteps_items')
    '''Array of state variable evolution'''
    @cached_property
    def _get_Eps_t(self):
        return np.array(self.state_vars, dtype=np.object)

    t = Property(depends_on='timesteps_items')

    @cached_property
    def _get_t(self):
        return np.array(self.timesteps, dtype=np.float_)

    n_t = Property(depends_on='timesteps_items')

    @cached_property
    def _get_n_t(self):
        return len(self.timesteps)

    def get_time_idx_arr(self, vot):
        '''Get the index corresponding to visual time
        '''
        t = self.t
        if len(t) == 0:
            return 0
        idx = np.array(np.arange(len(t)), dtype=np.float_) + 0.5
        t_idx = np.interp(vot, t, idx)
        return np.array(t_idx, np.int_)

    def get_time(self, vot):
        return self.t[self.get_time_idx(vot)]

    def get_time_idx(self, vot):
        return int(self.get_time_idx_arr(vot))
