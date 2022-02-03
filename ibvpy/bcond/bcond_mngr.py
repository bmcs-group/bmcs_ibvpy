
from .i_bcond import IBCond
from traits.api import \
    Instance,  \
    List
from ibvpy.view.ui import BMCSTreeNode


class BCondMngr(BMCSTreeNode):

    node_name = 'boundary conditions'

    bcond_list = List(IBCond)

    def _tree_node_list_default(self):
        return self.bcond_list

    selected_bcond = Instance(IBCond)

    def setup(self, sctx):
        '''
        '''
        for bc in self.bcond_list:
            bc.setup(sctx)

    def register(self, K):
        '''Register the boundary condition in the equation system.
        '''
        for bcond in self.bcond_list:
            bcond.register(K)

    def apply(self, step_flag, sctx, K, R, t_n, t_n1):

        for bcond in self.bcond_list:
            bcond.apply(step_flag, sctx, K, R, t_n, t_n1)

