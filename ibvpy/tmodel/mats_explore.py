
from ibvpy.tfunction import \
    LoadingScenario
from ibvpy.bcond import BCDof
from ibvpy.sim.sim_base import Simulator
from ibvpy.xmodel.xdomain_point import XDomainSinglePoint
from .mats3D import MATS3DDesmorat

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


