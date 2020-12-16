
from traits.api import \
    Callable, Float, List, Property, cached_property,\
    Instance
from traitsui.api import \
    Item, View, VGroup

from ibvpy.tfunction import \
    LoadingScenario
from ibvpy.bcond import BCDof
from ibvpy.sim.sim_base import Simulator
from ibvpy.xmodel.xdomain_point import XDomainSinglePoint
from ibvpy.view.window import BMCSWindow

from .mats_viz2d import Viz2DSigEps


class MATSExplore(Simulator):
    '''
    Simulate the loading histories of a material point in 2D space.
    '''

    node_name = 'Composite tensile test'

    def _bc_default(self):
        return [BCDof(
            var='u', dof=0, value=-0.001,
            time_function=LoadingScenario()
        )]

    def _model_default(self):
        return MATS3DDesmorat()

    def _xdomain_default(self):
        return XDomainSinglePoint()

    traits_view = View(
        resizable=True,
        width=1.0,
        height=1.0,
        scrollable=True,
    )

    tree_view = traits_view


