'''
Created on Mar 4, 2017

@author: rch
'''

import numpy as np
from ibvpy.view.plot2d import Vis2D, Viz2D
from ibvpy.view.ui.bmcs_tree_node import BMCSLeafNode


class TP(Viz2D):

    def plot(self, ax, vot):
        t, x, y = self.vis2d.get_sim_results(vot)
        ax.plot(x, y)


class ResponseTracer(BMCSLeafNode, Vis2D):

    node_name = 'response tracer'

    def get_sim_results(self, vot):
        t = np.linspace(0, 1, 100)
        x = t
        y = x**2 - vot
        return t, x, y

    viz2d_classes = {
        'time_profile': TP
    }
