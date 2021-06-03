
from os.path import join
from ibvpy.mathkit.mfn import MFnLineArray
from traits.api import Str, \
    Range, Property, cached_property, observe
from bmcs_utils.api import Float, Int, Enum, FloatRangeEditor
from ibvpy.view.plot2d import Viz2D
from ibvpy.view.ui import BMCSLeafNode
from ibvpy.view.reporter import RInputRecord
import bmcs_utils.api as bu
import numpy as np


class LoadingScenario(MFnLineArray, BMCSLeafNode, RInputRecord):
    """Loading scenario.
    """
    def reset(self):
        return

    node_name = Str('loading scenario')

    t_max = Float(1.)

    xdata = Property
    def _get_xdata(self):
        return self.xy_arrays[0]

    ydata = Property
    def _get_ydata(self):
        return self.xy_arrays[1]

    def update_plot(self, axes):
        axes.plot(self.xdata, self.ydata)
        axes.fill_between(self.xdata, self.ydata, 0, alpha=0.1)


class MonotonicLoadingScenario(LoadingScenario):

    n_incr = Int(10, BC=True)
    maximum_loading = Float(1.0, BC=True,
                            enter_set=True, auto_set=False,
                            symbol='\phi_{\max}',
                            desc='load factor at maximum load level',
                            unit='-')

    ipw_view = bu.View(
        bu.Item("n_incr"),
        bu.Item('maximum_loading'),
    )

    xy_arrays = Property(depends_on="state_changed")
    @cached_property
    def _get_xy_arrays(self):
        t_arr = np.linspace(0, self.t_max, self.n_incr)
        d_arr = np.linspace(0, self.maximum_loading, self.n_incr)
        return t_arr, d_arr

    def write_figure(self, f, rdir, rel_study_path):
        print('FNAME', self.node_name)
        fname = 'fig_' + self.node_name.replace(' ', '_') + '.pdf'
        print('FNAME', fname)
        self._update_xy_arrays()
        f.write(r'''
\multicolumn{3}{r}{\includegraphics[width=5cm]{%s}}\\
''' % join(rel_study_path, fname))
        self.savefig(join(rdir, fname))


class CyclicLoadingScenario(LoadingScenario):

    number_of_cycles = Int(1, BC=True,
                           enter_set=True, auto_set=False,
                           symbol='n_\mathrm{cycles}',
                           unit='-',
                           desc='for cyclic loading',
                           )
    maximum_loading = Float(1.0, BC=True,
                            enter_set=True, auto_set=False,
                            symbol='\phi_{\max}',
                            desc='load factor at maximum load level',
                            unit='-')
    number_of_increments = Int(20, BC=True,
                               enter_set=True, auto_set=False,
                               symbol='n_{\mathrm{incr}}',
                               unit='-',
                               desc='number of values within a monotonic load branch')
    unloading_ratio = Float(0.5, BC=True,
                            enter_set=True, auto_set=False,
                            symbol='\phi_{\mathrm{unload}}',
                            desc='fraction of maximum load at lowest load level',
                            unit='-')
    amplitude_type = Enum(options=["increasing", "constant"],
                          enter_set=True, auto_set=False,
                          symbol='option',
                          unit='-',
                          desc='possible values: [increasing, constant]',
                          BC=True)
    loading_range = Enum(options=["non-symmetric", "symmetric"],
                         enter_set=True, auto_set=False,
                         symbol='option',
                         unit='-',
                         desc='possible values: [non-symmetric, symmetric]',
                         BC=True)

    ipw_view = bu.View(
        bu.Item('number_of_cycles'),
        bu.Item('maximum_loading'),
        bu.Item('number_of_increments'),
        bu.Item('unloading_ratio'), # , editor=FloatRangeEditor(low=0, high=1)),
        bu.Item('amplitude_type'),
        bu.Item('loading_range'),
    )
    xy_arrays = Property(depends_on="state_changed")
    @cached_property
    def _get_xy_arrays(self):
        if(self.amplitude_type == "increasing" and
                self.loading_range == "symmetric"):
            d_levels = np.linspace(
                0, self.maximum_loading, self.number_of_cycles * 2)
            d_levels.reshape(-1, 2)[:, 0] *= -1
            d_history = d_levels.flatten()
            d_arr = np.hstack([np.linspace(d_history[i], d_history[i + 1],
                                           self.number_of_increments)
                               for i in range(len(d_levels) - 1)])

        if(self.amplitude_type == "increasing" and
                self.loading_range == "non-symmetric"):
            d_levels = np.linspace(
                0, self.maximum_loading, self.number_of_cycles * 2)
            d_levels.reshape(-1, 2)[:, 0] *= 0
            d_history = d_levels.flatten()
            d_arr = np.hstack([np.linspace(d_history[i], d_history[i + 1],
                                           self.number_of_increments)
                               for i in range(len(d_levels) - 1)])

        if(self.amplitude_type == "constant" and
                self.loading_range == "symmetric"):
            d_levels = np.linspace(
                0, self.maximum_loading, self.number_of_cycles * 2)
            d_levels.reshape(-1, 2)[:, 0] = -self.maximum_loading
            d_levels[0] = 0
            d_levels.reshape(-1, 2)[:, 1] = self.maximum_loading
            d_history = d_levels.flatten()
            d_arr = np.hstack([np.linspace(d_history[i], d_history[i + 1], self.number_of_increments)
                               for i in range(len(d_levels) - 1)])

        if(self.amplitude_type == "constant" and
                self.loading_range == "non-symmetric"):
            d_levels = np.linspace(
                0, self.maximum_loading, self.number_of_cycles * 2)
            d_levels.reshape(-1, 2)[:,
                                    0] = self.maximum_loading * self.unloading_ratio
            d_levels[0] = 0
            d_levels.reshape(-1, 2)[:, 1] = self.maximum_loading
            d_history = d_levels.flatten()
            d_arr = np.hstack([np.linspace(d_history[i], d_history[i + 1], self.number_of_increments)
                               for i in range(len(d_levels) - 1)])

        t_arr = np.linspace(0, self.t_max, len(d_arr))
        return t_arr, d_arr

class Viz2DLoadControlFunction(Viz2D):
    '''Plot adaptor for the pull-out simulator.
    '''
    label = 'Load control'

    def plot(self, ax, vot, *args, **kw):
        bc = self.vis2d.control_bc
        val = bc.value
        tloop = self.vis2d.tloop
        t_arr = np.array(tloop.t_record, np.float_)
        if len(t_arr) == 0:
            return
        f_arr = val * bc.time_function(t_arr)
        ax.plot(t_arr, f_arr, color='black')
        ax.set_ylabel('load factor')
        ax.set_xlabel('time')
        self.plot_marker(ax, vot)

    def plot_marker(self, ax, vot):
        bc = self.vis2d.control_bc
        val = bc.value
        tloop = self.vis2d.tloop
        t_arr = np.array(tloop.t_record, np.float_)
        if len(t_arr) == 0:
            return
        f_arr = val * bc.time_function(t_arr)
        vot_idx = tloop.get_time_idx(vot)
        ax.plot([t_arr[vot_idx]], [f_arr[vot_idx]], 'o',
                color='black', markersize=10)

    def plot_tex(self, ax, vot):
        self.plot(ax, vot)
