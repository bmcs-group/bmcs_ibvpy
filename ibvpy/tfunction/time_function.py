'''
'''

from bmcs_utils.api import \
    Model, Int, Float, FloatRangeEditor, View, Item, EitherType
from scipy.interpolate import interp1d
import traits.api as tr
import numpy as np


class TimeFunction(Model):
    name = 'time function'
    t_max = Float(1.0, TIME=True)
    t_shift = Float(0.0, TIME=True)

    def __call__(self, arg):
        return self.time_function(arg)

    time_function = tr.Property(depends_on='state_changed')
    @tr.cached_property
    def _get_time_function(self):
        return self._generate_time_function()

    def update_plot(self, axes):
        t_range = np.linspace(0,self.t_max,500)
        f_range = self.time_function(t_range)
        axes.plot(t_range, f_range)
        axes.set_xlabel(r'$t$ [-]')
        axes.set_ylabel(r'$\theta$ [-]')

class TFMonotonic(TimeFunction):

    ipw_view = View(
        Item('t_max'),
        Item('range_factor')
    )

    range_factor = Float(10000)

    def _generate_time_function(self):
        # Define the monotonic range beyond the unit range
        # allow for interactive extensions. The fact
        d_history = np.array([0, 1*self.range_factor])
        t_arr = np.array([0, self.t_max*self.range_factor])
        return interp1d(t_arr, d_history, bounds_error=False, fill_value='extrapolate')

class TFBilinear(TimeFunction):
    loading_ratio = Float(0.5, TIME=True)
    time_ratio = Float(0.5, TIME=True)

    ipw_view = View(
        Item('t_max'),
        Item('loading_ratio', editor=FloatRangeEditor(low=0, high=1)),
        Item('time_ratio', editor=FloatRangeEditor(low=0, high=1)),
        Item('range_factor')
    )

    range_factor = Float(1)

    def _generate_time_function(self):
        # Define the monotonic range beyond the unit range
        # allow for interactive extensions. The fact
        d_history = np.array([0, 1 *  self.loading_ratio *self.range_factor , 1 *self.range_factor ])
        t_arr = np.array([0, self.t_max  * self.time_ratio * self.range_factor , self.t_max *self.range_factor])
        return interp1d(t_arr, d_history, bounds_error=False, fill_value='extrapolate')


class TFCyclicSymmetricIncreasing(TimeFunction):
    number_of_cycles = Int(10, TIME=True)

    ipw_view = View(
        Item('number_of_cycles'),
        Item('t_max')
    )

    def _generate_time_function(self):
        n_levels = int(self.number_of_cycles) * 2
        d_levels = np.linspace(0, 1, n_levels)
        d_levels.reshape(-1, 2)[:, 0] *= -1
        d_history = d_levels.flatten()
        t_arr = np.linspace(0, self.t_max, len(d_history))
        return interp1d(t_arr, d_history, bounds_error=False, fill_value=self.t_max)


class TFCyclicNonsymmetricIncreasing(TimeFunction):
    number_of_cycles = Int(10, TIME=True)

    ipw_view = View(
        Item('number_of_cycles'),
        Item('t_max')
    )

    def _generate_time_function(self):
        d_levels = np.linspace(0, 1, self.number_of_cycles * 2)
        d_levels.reshape(-1, 2)[:, 0] *= 0
        d_history = d_levels.flatten()
        t_arr = np.linspace(0, self.t_max, len(d_history))
        return interp1d(t_arr, d_history, bounds_error=False, fill_value=self.t_max)


class TFCyclicSymmetricConstant(TimeFunction):
    number_of_cycles = Int(10, TIME=True)

    ipw_view = View(
        Item('number_of_cycles'),
        Item('t_max')
    )

    def _generate_time_function(self):
        d_levels = np.zeros((self.number_of_cycles * 2, ))
        d_levels.reshape(-1, 2)[:, 0] = -1
        d_levels.reshape(-1, 2)[:, 1] = 1
        d_history = d_levels.flatten()
        t_arr = np.linspace(0, self.t_max, len(d_history))
        return interp1d(t_arr, d_history, bounds_error=False, fill_value=self.t_max)

import sympy as sp

class TFCyclicNonsymmetricConstant(TimeFunction):
    number_of_cycles = Int(10, TIME=True)
    unloading_ratio = Float(0.5, TIME=True)
    shift_cycles = Int(0, TIME=True)

    ipw_view = View(
        Item('number_of_cycles'),
        Item('shift_cycles'),
        Item('unloading_ratio', editor=FloatRangeEditor(low=0, high=1)),
        Item('t_max')
    )

    def _generate_time_function(self):
        d_1 = np.zeros(1)
        d_2 = np.zeros(((self.number_of_cycles + self.shift_cycles) * 2, ))
        d_2.reshape(-1, 2)[self.shift_cycles:, 0] = 1
        d_2.reshape(-1, 2)[self.shift_cycles:, 1] = self.unloading_ratio
        d_history = d_2.flatten()
        d_arr = np.hstack((d_1, d_history))
        t_arr = np.linspace(0, self.t_max, len(d_arr))
        return interp1d(t_arr, d_arr, bounds_error=False, fill_value=self.t_max)


class TFCyclicSin(TimeFunction):
    number_of_cycles = Int(10, TIME=True)
    phase_shift = Float(0, TIME=True)

    ipw_view = View(
        Item('number_of_cycles'),
        Item('phase_shift'),
        Item('t_max')
    )

    def _generate_time_function(self):
        t = sp.symbols(r't')
        p = self.phase_shift
        tf = sp.sin(2 * self.number_of_cycles * sp.pi * t / self.t_max)
        if p > 0:
            T = self.t_max / self.number_of_cycles
            pT = p * T
            tf = sp.Piecewise(
                (0, t < pT),
                (tf.subs(t, t-pT), True)
            )
        return sp.lambdify(t, tf, 'numpy')

class TFSelector(TimeFunction):
    name='time function'
    profile = EitherType(
        options=[
            ('monotonic', TFMonotonic),
            ('bilinear', TFBilinear),
            ('cyclic-sym-incr', TFCyclicSymmetricConstant),
            ('cyclic-sym-const', TFCyclicSymmetricIncreasing),
            ('cyclic-nonsym-incr', TFCyclicNonsymmetricConstant),
            ('cyclic-nonsym-const', TFCyclicNonsymmetricIncreasing)
        ],
        TIME=True
    )

    t_max = tr.Property(Float)
    def _get_t_max(self):
        return self.profile_.t_max

    ipw_view=View(
        Item('profile')
    )

    def __call__(self, arg):
        return self.profile_(arg)

    def update_plot(self, axes):
        print('xxxxx')
        self.profile_.update_plot(axes)