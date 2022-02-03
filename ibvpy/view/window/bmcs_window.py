'''

@author: rch
'''


from traits.api import \
    HasStrictTraits, Instance, Button, Event, \
    DelegatesTo
from traits.etsconfig.api import ETSConfig

from .bmcs_study import BMCSStudy

if ETSConfig.toolkit == 'wx':
    from traitsui.wx.tree_editor import \
        DeleteAction
if ETSConfig.toolkit == 'qt4':
    from traitsui.qt4.tree_editor import \
        DeleteAction
else:
    raise ImportError("tree actions for %s toolkit not available" %
                      ETSConfig.toolkit)


# tree_node = TreeNode(node_for=[BMCSRootNode, BMCSTreeNode],
#                      auto_open=False,
#                      children='tree_node_list',
#                      label='node_name',
#                      view='tree_view',
#                      menu=Menu(plot_self, DeleteAction),
#                      )
#
# leaf_node = TreeNode(node_for=[BMCSLeafNode],
#                      auto_open=True,
#                      children='',
#                      label='node_name',
#                      view='tree_view',
#                      menu=Menu(plot_self)
#                      )
#
# tree_editor = TreeEditor(
#     nodes=[tree_node, leaf_node],
#     selected='selected_node',
#     orientation='vertical'
# )


class BMCSWindow(BMCSStudy):

    selected_node = Instance(HasStrictTraits)

    def _selected_node_changed(self):
        self.selected_node.ui = self

    def get_vot_range(self):
        return self.viz_sheet.get_vot_range()

    vot = DelegatesTo('viz_sheet')

    data_changed = Event

    replot = Button

    def _replot_fired(self):
        self.figure.clear()
        self.selected_node.plot(self.figure)
        self.data_changed = True

    clear = Button()

    def _clear_fired(self):
        self.figure.clear()
        self.data_changed = True
