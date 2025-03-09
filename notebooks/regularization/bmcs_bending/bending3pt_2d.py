'''
Created on 12.01.2016
@author: RChudoba, ABaktheer, Yingxiong

@todo: derive the size of the state array.
'''

from ibvpy.tfunction import MonotonicLoadingScenario, CyclicLoadingScenario
from ibvpy.api import \
    BCSlice, Hist
from ibvpy.bcond.bcond_mngr import BCondMngr
from ibvpy.fets import \
    FETS2D4Q
from ibvpy.tmodel.mats2D import \
    MATS2DElastic, MATS2DMplDamageEEQ, MATS2DScalarDamage, \
    MATS2DMplCSDEEQ
from scipy import interpolate as ip
from ibvpy.api import TStepBC, XDomainFEGrid
from traits.api import \
    Property, cached_property
from bmcs_utils.api import \
    Float, Int, Instance, View, Item, EitherType, Model, ProgressEditor, \
    HistoryEditor
from matplotlib import cm
from ibvpy.view.plot2d import Viz2D, Vis2D
from ibvpy.view.ui import BMCSLeafNode

import numpy as np
import traits.api as tr

from matplotlib.patches import PathPatch
from matplotlib.path import Path
import matplotlib.patches as mpatches
from .viz3d_energy import ForceDisplacement, Vis2DEnergy, Vis2DCrackBand

class Viz2DForceDeflection(Viz2D):
    '''Plot adaptor for the pull-out simulator.
    '''
    label = 'F-w'

    show_legend = tr.Bool(True, auto_set=False, enter_set=True)

    def plot(self, ax, vot, *args, **kw):
        sim = self.vis2d.sim
        P, W = sim.hist['Pw'].Pw
        ymin, ymax = np.min(P), np.max(P)
        L_y = ymax - ymin
        ymax += 0.05 * L_y
        ymin -= 0.05 * L_y
        xmin, xmax = np.min(W), np.max(W)
        L_x = xmax - xmin
        xmax += 0.03 * L_x
        xmin -= 0.03 * L_x
        color = kw.get('color', 'black')
        linewidth = kw.get('linewidth', 2)
        label = kw.get('label', 'P(w)')
        ax.plot(W, P, linewidth=linewidth, color=color, alpha=0.4,
                label=label)
        ax.set_ylim(ymin=ymin, ymax=ymax)
        ax.set_xlim(xmin=xmin, xmax=xmax)
        ax.set_ylabel('Force P [N]')
        ax.set_xlabel('Deflection w [mm]')
        if self.show_legend:
            ax.legend(loc=4)
        self.plot_marker(ax, vot)

    def plot_marker(self, ax, vot):
        sim = self.vis2d.sim
        P, W = sim.hist['Pw'].Pw
        idx = sim.hist.get_time_idx(vot)
        P, w = P[idx], W[idx]
        ax.plot([w], [P], 'o', color='black', markersize=10)

    def plot_tex(self, ax, vot, *args, **kw):
        self.plot(ax, vot, *args, **kw)

    traits_view = View(
        Item('name', style='readonly'),
        Item('show_legend'),
    )

def align_xaxis(ax1, ax2):
    """Align zeros of the two axes, zooming them out by same ratio"""
    axes = np.array([ax1, ax2])
    (ax1_min, ax1_max), (ax2_min, ax2_max) = np.array([ax.get_xlim() for ax in axes])
    ax1_delta = ax1_max - ax1_min
    ax2_delta = ax2_max - ax2_min
    ax1_rmin, ax1_rmax = ax1_min / ax1_delta, ax1_max / ax1_delta
    ax2_rmin, ax2_rmax = ax2_min / ax2_delta, ax2_max / ax2_delta
    rmin = np.min([ax1_rmin, ax2_rmin])
    rmax = np.max([ax1_rmax, ax2_rmax])
    ax1_min, ax1_max = ax1_delta * rmin, ax1_delta * rmax
    ax2_min, ax2_max = ax2_delta * rmin, ax2_delta * rmax
    ax1.set_xlim(ax1_min, ax1_max)
    ax2.set_xlim(ax2_min, ax2_max)


class Viz2DStrainInCrack(Viz2D):

    '''Plot adaptor for the pull-out simulator.
    '''
    label = 'strain in crack'

    show_legend = tr.Bool(True, auto_set=False, enter_set=True)

    ax_sig = tr.Any

    def plot(self, ax, vot, *args, **kw):
        eps = self.vis2d.get_eps_t(vot)
        sig = self.vis2d.get_sig_t(vot)
        a_x = self.vis2d.get_a_x()
        ax.plot(eps, a_x, linewidth=3, color='red', alpha=0.4,
                label='P(w;x=L)')
#        ax.fill_betweenx(eps, 0, a_x, facecolor='orange', alpha=0.2)
        ax.set_xlabel('Strain [-]')
        ax.set_ylabel('Position [mm]')
        if self.ax_sig:
            self.ax_sig.clear()
        else:
            self.ax_sig = ax.twiny()
        self.ax_sig.set_xlabel('Stress [MPa]')
        self.ax_sig.plot(sig, a_x, linewidth=2, color='blue')
        self.ax_sig.fill_betweenx(a_x, sig, facecolor='blue', alpha=0.2)
        align_xaxis(ax, self.ax_sig)
        if self.show_legend:
            ax.legend(loc=4)

    traits_view = View(
        Item('name', style='readonly'),
        Item('show_legend'),
    )


class Viz2DStressInCrack(Viz2D):

    '''Plot adaptor for the pull-out simulator.
    '''
    label = 'stress in crack'

    show_legend = tr.Bool(True, auto_set=False, enter_set=True)

    def plot(self, ax, vot, *args, **kw):
        sig = self.vis2d.get_sig_t(vot)
        a_x = self.vis2d.get_a_x()
        ax.plot(a_x, sig, linewidth=3, color='red', alpha=0.4,
                label='P(w;x=L)')
        ax.fill_between(a_x, 0, sig, facecolor='orange', alpha=0.2)
        ax.set_ylabel('Stress [-]')
        ax.set_xlabel('Position [mm]')
        if self.show_legend:
            ax.legend(loc=4)

    traits_view = View(
        Item('name', style='readonly'),
        Item('show_legend'),
    )


class Viz2DTA(Viz2D):

    '''Plot adaptor for bending test simulator.
    '''
    label = 'crack length'

    show_legend = tr.Bool(True, auto_set=False, enter_set=True)

    def plot(self, ax, vot, *args, **kw):
        t = self.vis2d.get_t()
        a = self.vis2d.get_a_t()
        ax.plot(t, a, linewidth=3, color='blue', alpha=0.4,
                label='Crack length')
        ax.fill_between(t, 0, a, facecolor='blue', alpha=0.2)
        ax.set_xlabel('time')
        ax.set_ylabel('a')
        if self.show_legend:
            ax.legend(loc=4)

    traits_view = View(
        Item('name', style='readonly'),
        Item('show_legend'),
    )


class Viz2DdGdA(Viz2D):
    label = 'Energy release per unit crack length'

    show_legend = tr.Bool(True, auto_set=False, enter_set=True)

    vis2d_cb = tr.WeakRef

    def plot(self, ax, vot, *args, **kw):
        t = self.vis2d.sim.hist.t
        G_t = self.vis2d.get_G_t()
        a_t = self.vis2d_cb.get_a_t()
        b = self.vis2d_cb.sim.cross_section.B

        tck = ip.splrep(a_t * b, G_t, s=0, k=1)
        dG_da = ip.splev(a_t, tck, der=1)

#         ax.plot(a_t, dG_da, linewidth=3, color='blue', alpha=0.4,
#                 label='dG/da')
#         ax.fill_between(a_t, 0, dG_da, facecolor='blue', alpha=0.2)
#         ax.set_xlabel('time')
#         ax.set_ylabel('dG_da')

        tck = ip.splrep(t, G_t, s=0, k=1)
        dG_dt = ip.splev(t[:-1], tck, der=1)
        tck = ip.splrep(t, a_t, s=0, k=1)
        da_dt = ip.splev(t[:-1], tck, der=1)
        nz = da_dt != 0.0
        dG_da = np.zeros_like(da_dt)
        dG_da[nz] = dG_dt[nz] / da_dt[nz] / b
        ax.plot(a_t[1:], dG_da, linewidth=3, color='green', alpha=0.4,
                label='dG/dt / da_dt')

#         ax.plot(a_t, G_t, linewidth=3, color='black')
#         ax.fill_between(a_t, 0, G_t, facecolor='blue', alpha=0.2)

        if self.show_legend:
            ax.legend(loc=4)

    traits_view = View(
        Item('name', style='readonly'),
        Item('show_legend'),
    )

class BendingHistory(Hist, Model):

    vis_record = tr.Dict
    def _vis_record_default(self):
        return {'Fw': ForceDisplacement(),
                'energy' : Vis2DEnergy(),
                'crack_band' : Vis2DCrackBand() }

    t_slider = Float(0)
    t_max = tr.Property()
    def _get_t_max(self):
        return self.t[-1]

    def plot_Fw(self, ax):
        F_range, w_range = self['Fw'].Fw
#        F_range, w_range = self.get_Fw_t()
        idx = self.get_time_idx(self.t_slider)
        F = F_range[idx]
        w = w_range[idx]
        ax.plot(-w, -F * 0.001, marker='o', color='magenta')
        ax.plot(-w_range, -F_range * 0.001)
        ax.set_ylabel(r'$F$ [kN]')
        ax.set_xlabel(r'$w$ [mm]')
        # ax.legend()

    U_bar_t = tr.Property
    def _get_U_bar_t(self):
        vis_energy = self['energy']
        return vis_energy.U_bar_t

    W_t = tr.Property
    def _get_W_t(self):
        vis_energy = self['energy']
        return vis_energy.get_W_t()

    def plot_G_t(self, ax,
             label_U='U(t)', label_W='W(t)',
             color_U='green', color_W='black'):
        vis_energy = self['energy']
        t = vis_energy.get_t()
        U_bar_t = np.array(vis_energy.U_bar_t, dtype=np.float64)
        W_t = vis_energy.get_W_t()
        ax.plot(t, W_t, color=color_W, label=label_W)
        ax.plot(t, U_bar_t, color=color_U, label=label_U)
        # Energy contribution relevant evaluated using internal variables
        # G_omega_t = np.array(vis_energy.G_omega_t, dtype=np.float64)
        # ax.plot(t, U_bar_t+G_omega_t, color='black', linestyle='dashed', label='G_t')
        ax.fill_between(t, W_t, U_bar_t, facecolor='gray', alpha=0.5,
                        label='G(t)')
        ax.fill_between(t, U_bar_t, 0, facecolor=color_U, alpha=0.5,
                        label='U(t)')
        ax.set_ylabel('energy [Nmm]')
        ax.set_xlabel('control displacement [mm]')
        ax.legend()

    def plot_crack_band(self, ax_eps, ax_sig):
        vis_cb = self['crack_band']
        t_idx = self.get_time_idx(self.t_slider)
        eps = vis_cb.get_eps_t(t_idx)
        sig = vis_cb.get_sig_t(t_idx)
        a_x = vis_cb.get_a_x()
        ax_eps.plot(eps, a_x, linewidth=1, color='red', alpha=0.4,
                label=r'$\varepsilon$ [-]')
        ax_eps.fill_betweenx(a_x, 0, eps, facecolor='orange', alpha=0.2)
        ax_eps.set_xlabel(r'$\varepsilon$ [-]')
        ax_eps.set_ylabel(r'$y$ [mm]')
        ax_sig.set_xlabel(r'$\sigma$ [MPa]')
        ax_sig.plot(sig, a_x, linewidth=1, color='blue',
                    label=r'$\sigma$ [MPa]')
        ax_sig.fill_betweenx(a_x, 0, sig, facecolor='blue', alpha=0.2)
        align_xaxis(ax_eps, ax_sig)
        # with bmcs_utils.api.print_output:
        #     print('eps', eps)
        ax_eps.legend(loc=5)
        ax_sig.legend(loc=4)

    warp_factor = Float(1.0)

    def plot_mesh(self, ax):
        tstep = self.tstep_source
        xdomain = tstep.xdomain
        t_idx = self.get_time_idx(self.t_slider)
        U = self.U_t[t_idx]
        o_Ia = xdomain.o_Ia  # bt.n_a + 1:]
        U_Ia = U[o_Ia] * self.warp_factor

        X_Ia = xdomain.mesh.X_Id + U_Ia # bt.n_a + 1:]
        I_Ei = xdomain.mesh.I_Ei
        i_lj = np.array([[0,1],[1,2],[2,3],[3,0]])
        X_Eia = X_Ia[I_Ei]
        X_Elja = X_Eia[:,i_lj,:]
        X_ajL = np.einsum('Elja->ajlE', X_Elja).reshape(2,2,-1)
        tstep.plot_geo(ax)
        ax.plot(X_ajL[0,...], X_ajL[1,...], color='black', linewidth=0.2)
        ax.axis('equal');

        omega_Em_t = np.array([state[0]['omega'] for state in self.state_vars])
        omega_Em = omega_Em_t[t_idx]
        n_Ii = np.bincount(I_Ei.flatten())
        omega_I = np.bincount(I_Ei.flatten(), weights=omega_Em.flatten()) / n_Ii
        X_aI = X_Ia.T
        n_x, n_y = tstep.n_e_x + 1, tstep.n_e_y + 1
        x_I, y_I = X_aI.reshape(2, n_x, n_y)
        omega_II = omega_I.reshape(n_x, n_y)
        ax.pcolormesh(x_I, y_I, omega_II, shading='nearest', cmap=cm.PuBu)
        ax.axis('equal')

    def subplots(self, fig):
        gs = fig.add_gridspec(2,2, width_ratios=[1., 1.])
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[0,1])
        ax3 = fig.add_subplot(gs[1,0])
        ax4 = fig.add_subplot(gs[1,1])
        ax44 = ax4.twiny()
        return ax1, ax2, ax3, ax4, ax44

    def update_plot(self, axes):
        ax_mesh, ax_Fw, ax_x, ax_eps, ax_sig = axes
        self.plot_mesh(ax_mesh)
        self.plot_Fw(ax_Fw)
        self.plot_crack_band(ax_eps, ax_sig)

    ipw_view = View(
        Item('warp_factor'),
        Item('t_slider', readonly=True),
        time_editor=HistoryEditor(
            var='t_slider',
            max_var='t_max',
        )
    )


class CrossSection(BMCSLeafNode):
    '''Parameters of the pull-out cross section
    '''
    name = 'cross-section'

    B = Float(100.0,
              CS=True,
              label='thickness',
              auto_set=False, enter_set=True,
              desc='cross-section width [mm2]')

    ipw_view = View(
        Item('B'),
    )


class Geometry(BMCSLeafNode):

    name = 'geometry'
    H = Float(100.0,
              label='beam depth',
              GEO=True,
              auto_set=False, enter_set=True,
              desc='cross section height [mm2]')
    L = Float(600.0,
              label='beam length',
              GEO=True,
              auto_set=False, enter_set=True,
              desc='Length of the specimen')
    a = Float(20.0,
              GEO=True,
              label='notch depth',
              auto_set=False, enter_set=True,
              desc='Depth of the notch')
    L_cb = Float(4.0,
                GEO=True,
                label='crack band width',
                auto_set=False, enter_set=True,
                desc='Width of the crack band')

    ipw_view = View(
        Item('H'),
        Item('L'),
        Item('a'),
        Item('L_cb', latex=r'L_\mathrm{cb}'),
    )


itags_str = "state_changed"

class BendingTestModel(TStepBC, Model):

    #=========================================================================
    # Tree node attributes
    #=========================================================================
    node_name = 'bending test simulation'
    name = '3PT bending'

    hist_type = BendingHistory

    history = tr.Property()
    @tr.cached_property
    def _get_history(self):
        return self.hist

    time_line = tr.Property()
    @tr.cached_property
    def _get_time_line(self):
        return self.sim.tline

    tree = ['time_line', 'cross_section', 'geometry',
            'material_model', 'loading_scenario', 'history']

    # =========================================================================
    # Test setup parameters
    # =========================================================================
    loading_scenario = EitherType(
        options=[('monotonic', MonotonicLoadingScenario),
                 ('cyclic', CyclicLoadingScenario)],
        report=True, TIME=True,
        desc='object defining the loading scenario'
    )

    cross_section = Instance(CrossSection)

    def _cross_section_default(self):
        return CrossSection()

    geometry = Instance(Geometry)

    def _geometry_default(self):
        return Geometry()

    #=========================================================================
    # Discretization
    #=========================================================================
    n_e_x = Int(20,
                label='# of elems in x-dir',
                MESH=True, auto_set=False, enter_set=True)
    n_e_y = Int(8,
                label='# of elems in y-dir',
                MESH=True, auto_set=False, enter_set=True)

    w_max = Float(-5, BC=True, auto_set=False, enter_set=True)

    k_max = tr.Property(Int, depends_on='state_change')
    @tr.cached_property
    def _get_k_max(self):
        return self.sim.tloop.k_max
    def _set_k_max(self, value):
        self.sim.tloop.k_max = value

    controlled_elem = Property

    def _get_controlled_elem(self):
        return 0

    #=========================================================================
    # Material model
    #=========================================================================
    material_model = EitherType(
        options=[('scalar damage', MATS2DScalarDamage),
                 ('elastic', MATS2DElastic),
                 ('microplane damage (eeq)', MATS2DMplDamageEEQ),
                 ('microplane CSD (eeq)', MATS2DMplCSDEEQ),
                 ],
        MAT=True
    )
    '''Material model'''

    mm = Property

    def _get_mm(self):
        return self.material_model_

    material = Property

    def _get_material(self):
        return self.material_model_


    #=========================================================================
    # Simulator interface
    #=========================================================================
    def run(self):
        self.sim.run()

    def reset(self):
        self.sim.reset()

    t = tr.Property()

    def _get_t(self):
        return self.sim.t

    def _set_t(self, value):
        self.sim.t = value

    t_max = tr.Property()

    def _get_t_max(self):
        return self.sim.t_max

    def _set_t_max(self, value):
        self.sim.t_max = value

    interrupt = tr.Property()

    def _get_interrupt(self):
        return self.sim.interrupt

    def _set_interrupt(self, value):
        self.sim.interrupt = value

    #=========================================================================
    # Finite element type
    #=========================================================================
    fets_eval = Property(Instance(FETS2D4Q),
                         depends_on=itags_str)
    '''Finite element time stepper implementing the corrector
    predictor operators at the element level'''
    @cached_property
    def _get_fets_eval(self):
        return FETS2D4Q()

    bc = Property(Instance(BCondMngr),
                  depends_on=itags_str)
    '''Boundary condition manager
    '''
    @cached_property
    def _get_bc(self):
        return [self.fixed_right_bc,
                self.fixed_x,
                self.control_bc]

    fixed_right_bc = Property(depends_on=itags_str)
    '''Foxed boundary condition'''
    @cached_property
    def _get_fixed_right_bc(self):
        return BCSlice(slice=self.fe_grid[-1, 0, -1, 0],
                       var='u', dims=[1], value=0)

    n_a = Property
    '''Element at the notch
    '''

    def _get_n_a(self):
        a_L = self.geometry.a / self.geometry.H
        return int(a_L * self.n_e_y)

    fixed_x = Property(depends_on=itags_str)
    '''Foxed boundary condition'''
    @cached_property
    def _get_fixed_x(self):
        return BCSlice(slice=self.fe_grid[0, self.n_a:, 0, -1],
                       var='u', dims=[0], value=0)

    control_bc = Property(depends_on=itags_str)
    '''Control boundary condition - make it accessible directly
    for the visualization adapter as property
    '''
    @cached_property
    def _get_control_bc(self):
        return BCSlice(slice=self.fe_grid[0, -1, :, -1],
                       var='u', dims=[1], value=self.w_max)

    xdomain = Property(depends_on=itags_str)
    '''Discretization object.
    '''
    @cached_property
    def _get_xdomain(self):
        dgrid = XDomainFEGrid(coord_max=(self.geometry.L / 2., self.geometry.H),
                              integ_factor=self.cross_section.B,
                              shape=(self.n_e_x, self.n_e_y),
                              fets=self.fets_eval)

        L = self.geometry.L / 2.0
        L_cb = self.geometry.L_cb
        x_x, x_y = dgrid.mesh.geo_grid.point_x_grid
        L_1 = x_x[1, 0]
        d_L = L_cb - L_1
        x_x[1:, :] += d_L * (L - x_x[1:, :]) / (L - L_1)
        return dgrid

    fe_grid = Property

    def _get_fe_grid(self):
        return self.xdomain.mesh

    domains = Property(depends_on=itags_str)

    @cached_property
    def _get_domains(self):
        return [(self.xdomain, self.material_model_)]

    ipw_view = View(
        Item('w_max'),
        Item('k_max'),
        Item('n_e_x'),
        Item('n_e_y'),
        Item('material_model'),
        Item('loading_scenario'),
        time_editor=ProgressEditor(run_method='run',
                                      reset_method='reset',
                                      interrupt_var='interrupt',
                                      time_var='t',
                                      time_max='t_max',
                                      )
    )

    def plot_geo(self, ax):
        L, H = self.geometry.L, self.geometry.H
        L2 = L/2
        ax.plot([-L2, L2, L2, -L2, -L2], [0, 0, H, H, 0], color='black', linewidth=0.5)
        ax.fill([-L2, L2, L2, -L2, -L2], [0, 0, H, H, 0], color='gray', alpha=0.3)
        ax.set_xlabel(r'$x$ [mm]'); ax.set_ylabel(r'$y$ [mm]')
        ax.axis('equal');


        support_width = 0.04 * L
        support_line_distance = 1.3 * support_width

        supports = [(-L2, 0), (L2, 0)]
        codes = []
        vertices = []
        for i, support in enumerate(supports):
            x_loc, def_value = support
            # Draw the triangle of the support
            codes += [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
            vertices += [(x_loc, 0),
                         (x_loc + support_width / 2, -support_width),
                         (x_loc - support_width / 2, -support_width),
                         (x_loc, 0)]

        vertices = np.array(vertices, np.float64)
        path = Path(vertices, codes)
        path_patch = PathPatch(path, facecolor='gray', edgecolor='gray')
        ax.add_patch(path_patch)

        x_tail = 0
        y_tail = H * 1.8
        x_head = 0
        y_head = H
        xy_annotate = (-L2*0.3, H*2)
        arrow = mpatches.FancyArrowPatch((x_tail, y_tail), (x_head, y_head),
                                         color='blue', mutation_scale=10)
        ax.annotate('$w_\max$ = {} mm'.format(self.w_max), xy=xy_annotate, color='black')
        ax.add_patch(arrow)

    def subplots(self, fig):
        (ax_geo, ax_Fw), (ax_energy, ax_eps) = fig.subplots(2, 2)
        self.plot_geo(ax_geo)
        ax_sig = ax_eps.twiny()
        return ax_geo, ax_Fw, ax_energy, ax_eps, ax_sig

    def update_plot(self, axes):
        if len(self.history.U_t) == 0:
            return
        ax_geo, ax_Fw, ax_energy, ax_eps, ax_sig = axes
        self.history.plot_mesh(ax_geo)
        self.history.t_slider = self.t
        self.history.plot_Fw(ax_Fw)
        self.history.plot_G_t(ax_energy)
        self.history.plot_crack_band(ax_eps, ax_sig)

class TensileTestModel(BendingTestModel):
    bc = Property(Instance(BCondMngr),
                  depends_on=itags_str)
    '''Boundary condition manager
    '''
    @cached_property
    def _get_bc(self):
        return [self.control_bc,
                self.fixed_x,
                self.fixed_y]

    control_bc = Property(depends_on=itags_str)
    '''Foxed boundary condition'''
    @cached_property
    def _get_control_bc(self):
        return BCSlice(slice=self.fe_grid[-1, :, -1, :],
                       var='u', dims=[0], value=self.w_max)

    n_a = Property
    '''Element at the notch
    '''

    def _get_n_a(self):
        a_L = self.geometry.a / self.geometry.H
        return int(a_L * self.n_e_y)

    fixed_x = Property(depends_on=itags_str)
    '''Foxed boundary condition'''
    @cached_property
    def _get_fixed_x(self):
        return BCSlice(slice=self.fe_grid[0, self.n_a:-self.n_a, 0, :],
                       var='u', dims=[0], value=0)

    fixed_y = Property(depends_on=itags_str)
    '''Control boundary condition - make it accessible directly
    for the visualization adapter as property
    '''
    @cached_property
    def _get_fixed_y(self):
        n_e_y = int(self.n_e_y / 2) + 1
        return BCSlice(slice=self.fe_grid[0, n_e_y, 0, 0],
                       var='u', dims=[1], value=0)

