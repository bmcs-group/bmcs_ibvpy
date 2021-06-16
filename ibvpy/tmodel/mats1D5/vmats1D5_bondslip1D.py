'''

'''

from os.path import join

from ibvpy.tmodel import MATSEval
from ibvpy.tmodel.mats_damage_fn import \
    IDamageFn, GfDamageFn, ExpSlopeDamageFn, AbaqusDamageFn, \
    LinearDamageFn, FRPDamageFn, WeibullDamageFn
from ibvpy.mathkit.mfn.mfn_line.mfn_line import MFnLineArray
from traits.api import  \
    observe, List, on_trait_change, \
    Instance, Trait, Bool, Property, cached_property

import bmcs_utils.api as bu
import numpy as np
import traitsui.api as ui

import ipyregulartable as rt
class MATSEval1D5(MATSEval):
    """Base class defining the plot
    """
    s_max = bu.Float(0.5, tooltip='Visualization limit [m]')

    def subplots(self, fig):
        ax_tau = fig.subplots(1,1)
        ax_d_tau = ax_tau.twinx()
        return ax_tau, ax_d_tau

    def update_plot(self, axes):
        ax_tau, ax_d_tau = axes
        s_max = self.s_max
        n_s = 100
        s_range = np.linspace(1e-9,s_max,n_s)
        eps_range = np.zeros((n_s, 3))
        eps_range[:,1] = s_range
        state_vars = { var : np.zeros( (n_s,) + shape )
            for var, shape in self.state_var_shapes.items()
        }
        sig_range, D = self.get_corr_pred(eps_range, 1, **state_vars)
        tau_range = sig_range[:,1]
        ax_tau.plot(s_range, tau_range,color='blue')
        d_tau_range = D[...,1,1]
        ax_d_tau.plot(s_range, d_tau_range,
                      linestyle='dashed', color='gray')
        ax_tau.set_xlabel(r'$s$ [mm]')
        ax_tau.set_ylabel(r'$\tau$ [MPa]')
        ax_d_tau.set_ylabel(r'$\mathrm{d} \tau / \mathrm{d} s$ [MPa/mm]')

class MATS1D5BondSlipMultiLinear(MATSEval1D5):
    """Multilinear bond-slip law
    """
    name = "multilinear bond law"

    E_m = bu.Float(28000.0, tooltip='Stiffness of the matrix [MPa]',
                MAT=True, unit='MPa', symbol='E_\mathrm{m}',
                desc='E-modulus of the matrix')

    E_f = bu.Float(170000.0, tooltip='Stiffness of the fiber [MPa]',
                MAT=True, unit='MPa', symbol='E_\mathrm{f}',
                desc='E-modulus of the reinforcement')

    s_data = bu.Str('0,1', tooltip='Comma-separated list of strain values',
                 MAT=True, unit='mm', symbol='s',
                 desc='slip values')

    tau_data = bu.Str('0,1', tooltip='Comma-separated list of stress values',
                   MAT=True, unit='MPa', symbol=r'\tau',
                   desc='shear stress values')

    s_tau_table = Property(depends_on='state_changed')
    @cached_property
    def _get_s_tau_table(self):
        s_data = np.fromstring(self.s_data, dtype=np.float_, sep=',')
        tau_data = np.fromstring(self.tau_data, dtype=np.float_, sep=',')
        if len(s_data) != len(tau_data):
            raise ValueError('s array and tau array must have the same size')
        return s_data, tau_data

    #=========================================================================
    # Configurational parameters
    #=========================================================================
    U_var_shape = (1,)
    '''Shape of the primary variable required by the TStepState.
    '''

    state_var_shapes = {}
    r'''
    Shapes of the state variables
    to be stored in the global array at the level 
    of the domain.
    '''

    node_name = 'multiply linear bond'

    def get_corr_pred(self, eps_n1, t_n1):
        D_shape = eps_n1.shape[:-1] + (3, 3)
        D = np.zeros(D_shape, dtype=np.float_)

        D[..., 0, 0] = self.E_m
        D[..., 2, 2] = self.E_f

        tau = np.einsum('...st,...t->...s', D, eps_n1)
        s = eps_n1[..., 1]
        shape = s.shape
        signs = np.sign(s.flatten())
        s_pos = np.fabs(s.flatten())
        bs_law = self.bs_law
        tau[..., 1] = (signs * bs_law(s_pos)).reshape(*shape)
        D_tau = self.bs_law.diff(s_pos).reshape(*shape)
        D[..., 1, 1] = D_tau

        return tau, D

    bs_law = Property(depends_on='state_changed')
    @cached_property
    def _get_bs_law(self):
        s_data, tau_data = self.s_tau_table
        return MFnLineArray(
            xdata=s_data,
            ydata=tau_data,
            plot_diff=False
        )

    ipw_view = bu.View(
        bu.Item('E_m'),
        bu.Item('E_f'),
        bu.Item('s_data'),
        bu.Item('tau_data'),
    )

    # def plot(self, ax, **kw):
    #     s_data, tau_data = self.s_tau_table
    #     ax.plot(s_data, tau_data, **kw)
    #     ax.fill_between(s_data, tau_data, alpha=0.1, **kw)
    #
    # def update_plot(self, axes):
    #     self.plot(axes)


class MATS1D5BondSlipD(MATSEval1D5):
    name = 'damage model'
    node_name = 'damage model'

    E_m = bu.Float(30000.0, tooltip='Stiffness of the matrix [MPa]',
                MAT=True)

    E_f = bu.Float(200000.0, tooltip='Stiffness of the fiber [MPa]',
                MAT=True)

    E_b = bu.Float(10000.0, tooltip='Stiffness of the fiber [MPa]',
                MAT=True)

    omega_fn = bu.EitherType(
        options=[('linear', LinearDamageFn),
                 ('abaqus', AbaqusDamageFn),
                 ('exp-slope', ExpSlopeDamageFn),
                 ('fracture-energy', GfDamageFn),
                 ('weibull-CDF', WeibullDamageFn),
                 ],
        MAT=True,
        on_option_change='link_omega_to_mats'
    )

    D_alg = bu.Float(0, MAT=True)

    # upon change of the type attribute set the link to the material model
    def link_omega_to_mats(self):
        self.omega_fn_.trait_set(mats=self,
                                 E_name='E_b',
                                 x_max_name='s_max')

    tree = ['omega_fn']

    def omega(self, k):
        return self.omega_fn_(k)

    def omega_derivative(self, k):
        return self.omega_fn_.diff(k)

    state_var_shapes = {
        'kappa_n' : (),
        'omega_n' : ()
    }

    def get_corr_pred(self, eps_n1, t_n1, kappa_n, omega_n):
        D_shape = eps_n1.shape[:-1] + (3, 3)
        D = np.zeros(D_shape, dtype=np.float_)
        D[..., 0, 0] = self.E_m
        D[..., 2, 2] = self.E_f
        s_n1 = eps_n1[..., 1]
        kappa_n[...] = np.max(np.array([kappa_n, np.fabs(s_n1)]), axis=0)
        omega_n[...] = self.omega(kappa_n)
        tau = np.einsum('...st,...t->...s', D, eps_n1)
        tau[..., 1] = (1 - omega_n) * self.E_b * s_n1
        D[..., 1, 1] = (1 - omega_n) * self.E_b
        if self.D_alg > 0:
            I = self.omega_fn_.get_f_trial(np.fabs(s_n1), kappa_n)
            domega_ds_I = self.omega_derivative(kappa_n[I])
            D_red_I = domega_ds_I * np.fabs(s_n1[I]) * self.E_b * self.D_alg
            D[I+(1, 1)] -= D_red_I
        return tau, D

    ipw_view = bu.View(
        bu.Item('E_m', latex=r'E_\mathrm{m}'),
        bu.Item('E_f', latex=r'E_\mathrm{f}'),
        bu.Item('E_b', latex=r'E_\mathrm{b}'),
        bu.Item('omega_fn', latex=r'\omega(s)'),
        bu.Item('D_alg', latex=r'\theta_\mathrm{alg. stiff.}',
                editor=bu.FloatRangeEditor(low=0,high=1)),
        bu.Item('s_max')
    )

class MATSBondSlipDP(MATSEval1D5):

    node_name = 'bond model: damage-plasticity'

    tree_node_list = List([])

    def _tree_node_list_default(self):
        return [self.omega_fn, ]

    @on_trait_change('omega_fn_type')
    def _update_node_list(self):
        self.tree_node_list = [self.omega_fn]

    E_m = bu.Float(30000.0, tooltip='Stiffness of the matrix [MPa]',
                symbol='E_\mathrm{m}',
                unit='MPa',
                desc='Stiffness of the matrix',
                MAT=True,
                auto_set=True, enter_set=True)

    E_f = bu.Float(200000.0, tooltip='Stiffness of the reinforcement [MPa]',
                symbol='E_\mathrm{f}',
                unit='MPa',
                desc='Stiffness of the reinforcement',
                MAT=True,
                auto_set=False, enter_set=False)

    E_b = bu.Float(12900.0,
                symbol="E_\mathrm{b}",
                unit='MPa',
                desc="Bond stiffness",
                MAT=True,
                enter_set=True,
                auto_set=False)

    gamma = bu.Float(100.0,
                  symbol="\gamma",
                  unit='MPa',
                  desc="Kinematic hardening modulus",
                  MAT=True,
                  enter_set=True,
                  auto_set=False)

    K = bu.Float(1000.0,
              symbol="K",
              unit='MPa',
              desc="Isotropic hardening modulus",
              MAT=True,
              enter_set=True,
              auto_set=False)

    tau_bar = bu.Float(5.0,
                    symbol=r'\bar{\tau}',
                    unite='MPa',
                    desc="Reversibility limit",
                    MAT=True,
                    enter_set=True,
                    auto_set=False)

    uncoupled_dp = Bool(False,
                        MAT=True,
                        label='Uncoupled d-p'
                        )

    s_0 = bu.Float(MAT=True,
                desc='Elastic strain/displacement limit')

    def __init__(self, *args, **kw):
        super(MATSBondSlipDP, self).__init__(*args, **kw)
        self._omega_fn_type_changed()
        self._update_s0()

    @on_trait_change('tau_bar,E_b')
    def _update_s0(self):
        if not self.uncoupled_dp:
            if self.E_b == 0:
                self.s_0 = 0
            else:
                self.s_0 = self.tau_bar / self.E_b
            self.omega_fn.s_0 = self.s_0

    omega_fn_type = Trait('multilinear',
                          dict(selfregularized=GfDamageFn,
                               exp_slope=ExpSlopeDamageFn,
                               abaqus=AbaqusDamageFn,
                               # FRP=FRPDamageFn,
                               # multilinear=MultilinearDamageFn
                               ),
                          MAT=True,
                          )

    def _omega_fn_type_changed(self):
        self.omega_fn = self.omega_fn_type_(mats=self, s_0=self.s_0)

    omega_fn = Instance(IDamageFn,
                        report=True)

    def _omega_fn_default(self):
        return MultilinearDamageFn()

    def omega(self, k):
        return self.omega_fn(k)

    def omega_derivative(self, k):
        return self.omega_fn.diff(k)

    state_var_shapes = dict(s_pl_n=(),
                            alpha_n=(),
                            z_n=(),
                            kappa_n=(),
                            omega_n=())

    def get_corr_pred(self, eps_n1, t_n1,
                      s_pl_n, alpha_n, z_n, kappa_n, omega_n):

        D_shape = eps_n1.shape[:-1] + (3, 3)
        D = np.zeros(D_shape, dtype=np.float_)
        D[..., 0, 0] = self.E_m
        D[..., 2, 2] = self.E_f

        s_n1 = eps_n1[..., 1]

        sig_pi_trial = self.E_b * (s_n1 - s_pl_n)

        Z = self.K * z_n

        # for handeling the negative values of isotropic hardening
        h_1 = self.tau_bar + Z
        pos_iso = h_1 > 1e-6

        X = self.gamma * alpha_n

        # for handeling the negative values of kinematic hardening (not yet)
        # h_2 = h * np.sign(sig_pi_trial - X) * \
        #    np.sign(sig_pi_trial) + X * np.sign(sig_pi_trial)
        #pos_kin = h_2 > 1e-6

        f_trial = np.fabs(sig_pi_trial - X) - h_1 * pos_iso

        I = f_trial > 1e-6

        tau = np.einsum('...st,...t->...s', D, eps_n1)
        # Return mapping
        delta_lamda_I = f_trial[I] / (self.E_b + self.gamma + np.fabs(self.K))

        # update all the state variables
        s_pl_n[I] += delta_lamda_I * np.sign(sig_pi_trial[I] - X[I])
        z_n[I] += delta_lamda_I
        alpha_n[I] += delta_lamda_I * np.sign(sig_pi_trial[I] - X[I])

        kappa_n[I] = np.max(
            np.array([kappa_n[I], np.fabs(s_n1[I])]), axis=0)
        omega_n[I] = self.omega(kappa_n[I])

        tau[..., 1] = (1 - omega_n) * self.E_b * (s_n1 - s_pl_n)

        domega_ds_I = self.omega_derivative(kappa_n[I])

        # Consistent tangent operator
        D_ed_I = -self.E_b / (self.E_b + self.K + self.gamma) \
            * domega_ds_I * self.E_b * (s_n1[I] - s_pl_n[I]) \
            + (1 - omega_n[I]) * self.E_b * (self.K + self.gamma) / \
            (self.E_b + self.K + self.gamma)

        D[..., 1, 1] = (1 - omega_n) * self.E_b
        D[I, 1, 1] = D_ed_I

        return tau, D

    tree_view = ui.View(
        ui.VGroup(
            ui.VGroup(
                ui.Item('E_m', full_size=True, resizable=True),
                ui.Item('E_f'),
                ui.Item('E_b'),
                ui.Item('gamma'),
                ui.Item('K'),
                ui.Item('tau_bar'),
            ),
            ui.VGroup(
                ui.Item('uncoupled_dp'),
                ui.Item('s_0'),  # , enabled_when='uncoupled_dp'),
                ui.Item('omega_fn_type'),
            ),
            ui.UItem('omega_fn@')
        )
    )


class MATS1D5BondSlipEP(MATSEval1D5):
    '''Elastc plastic bond slip model
    '''
    name = 'elastic-plastic model'

    E_m = bu.Float(30000.0, tooltip='Stiffness of the matrix [MPa]',
                symbol='E_\mathrm{m}',
                unit='MPa',
                desc='Stiffness of the matrix',
                MAT=True,
                auto_set=True, enter_set=True)

    E_f = bu.Float(200000.0, tooltip='Stiffness of the reinforcement [MPa]',
                symbol='E_\mathrm{f}',
                unit='MPa',
                desc='Stiffness of the reinforcement',
                MAT=True,
                auto_set=False, enter_set=False)

    E_b = bu.Float(12900.0,
                symbol="E_\mathrm{b}",
                unit='MPa',
                desc="Bond stiffness",
                MAT=True,
                enter_set=True,
                auto_set=False)

    gamma = bu.Float(100.0,
                  symbol="\gamma",
                  unit='MPa',
                  desc="Kinematic hardening modulus",
                  MAT=True,
                  enter_set=True,
                  auto_set=False)

    K = bu.Float(1000.0,
              symbol="K",
              unit='MPa',
              desc="Isotropic hardening modulus",
              MAT=True,
              enter_set=True,
              auto_set=False)

    tau_bar = bu.Float(5.0,
                    symbol=r'\bar{\tau}',
                    unite='MPa',
                    desc="Reversibility limit",
                    MAT=True,
                    enter_set=True,
                    auto_set=False)

    state_var_shapes = dict(s_pl_n=(),
                            alpha_n=(),
                            z_n=())

    def get_corr_pred(self, eps_n1, t_n1, s_pl_n, alpha_n, z_n):

        D_shape = eps_n1.shape[:-1] + (3, 3)
        D = np.zeros(D_shape, dtype=np.float_)
        D[..., 0, 0] = self.E_m
        D[..., 2, 2] = self.E_f

        s_pl_n0 = np.copy(s_pl_n)
        z_n0 = np.copy(z_n)
        alpha_n0 = np.copy(alpha_n)

        s_n1 = eps_n1[..., 1]
        sig_pi_trial = self.E_b * (s_n1 - s_pl_n)
        Z = self.K * z_n
        H_1 = self.tau_bar + Z
        X = self.gamma * alpha_n

        f_trial = np.fabs(sig_pi_trial - X) - H_1

        I = f_trial > 1e-6

        tau = np.einsum('...st,...t->...s', D, eps_n1)
        # Return mapping
        delta_lamda_I = f_trial[I] / (self.E_b + self.gamma + self.K)

        # update all the state variables
        s_pl_n[I] += delta_lamda_I * np.sign(sig_pi_trial[I] - X[I])
        z_n[I] += delta_lamda_I
        alpha_n[I] += delta_lamda_I * np.sign(sig_pi_trial[I] - X[I])

        tau[..., 1] = self.E_b * (s_n1 - s_pl_n)

        # Consistent tangent operator
        D_ed_I = (self.E_b * (self.K + self.gamma) /
                  (self.E_b + self.K + self.gamma)
                  )

        D[..., 1, 1] = self.E_b
        D[I, 1, 1] = D_ed_I

        Z = self.K * z_n
        # check if the size of the elastic domain is still nonzero
        J = self.tau_bar + Z < 0
        tau[J, 1] = 0
        D[J, 1, 1] = 0
        return tau, D

    ipw_view = bu.View(
        bu.Item('E_m'),
        bu.Item('E_f'),
        bu.Item('E_b'),
        bu.Item('gamma'),
        bu.Item('K'),
        bu.Item('tau_bar'),
    )

    @observe('state_changed')
    def _reset_s_max(self, event=None):
        self.s_max = self.tau_bar / self.E_b * 10

class MATSBondSlipFatigue(MATSEval1D5):

    node_name = 'bond model: bond fatigue'

    E_m = bu.Float(30000, tooltip='Stiffness of the matrix [MPa]',
                MAT=True,
                auto_set=True, enter_set=True)

    E_f = bu.Float(200000, tooltip='Stiffness of the fiber [MPa]',
                MAT=True,
                auto_set=False, enter_set=False)

    E_b = bu.Float(12900,
                label="E_b",
                desc="Bond Stiffness",
                MAT=True,
                enter_set=True,
                auto_set=False)

    gamma = bu.Float(55.0,
                  label="Gamma",
                  desc="Kinematic hardening modulus",
                  MAT=True,
                  enter_set=True,
                  auto_set=False)

    K = bu.Float(11.0,
              label="K",
              desc="Isotropic harening",
              MAT=True,
              enter_set=True,
              auto_set=False)

    S = bu.Float(0.00048,
              label="S",
              desc="Damage cumulation parameter",
              enter_set=True,
              MAT=True,
              auto_set=False)

    r = bu.Float(0.5,
              label="r",
              desc="Damage cumulation parameter",
              MAT=True,
              enter_set=True,
              auto_set=False)

    c = bu.Float(2.8,
              label="c",
              desc="Damage cumulation parameter",
              MAT=True,
              enter_set=True,
              auto_set=False)

    tau_pi_bar = bu.Float(4.2,
                       label="Tau_pi_bar",
                       desc="Reversibility limit",
                       MAT=True,
                       enter_set=True,
                       auto_set=False)

    pressure = bu.Float(0,
                     label="Pressure",
                     desc="Lateral pressure",
                     MAT=True,
                     enter_set=True,
                     auto_set=False)

    a = bu.Float(1.7,
              label="a",
              desc="Lateral pressure coefficient",
              MAT=True,
              enter_set=True,
              auto_set=False)

    state_var_shapes = dict(xs_pi=(),
                            alpha=(),
                            z=(),
                            kappa=(),
                            omega=()
                            )

    def get_corr_pred(self, eps_n1, t_n1, xs_pi, alpha, z, kappa, omega):

        D_shape = eps_n1.shape[:-1] + (3, 3)
        D = np.zeros(D_shape, dtype=np.float_)
        D[..., 0, 0] = self.E_m
        D[..., 2, 2] = self.E_f

        s_n1 = eps_n1[..., 1]

        tau_pi_trial = self.E_b * (s_n1 - xs_pi)
        X = self.gamma * alpha
        Z = self.K * z

        f_trial = np.fabs(tau_pi_trial - X) - self.tau_pi_bar - \
            Z + self.a * self.pressure / 3

        I = np.where(f_trial > 1e-6)

        sig = np.einsum('...st,...t->...s', D, eps_n1)
        sig[..., 1] = tau_pi_trial

        omega_I = omega[I]

        # Return mapping
        delta_lamda_I = f_trial[I] / \
            (self.E_b / (1 - omega_I) + self.gamma + self.K)

        # update all the state variables
        xs_pi[I] += (delta_lamda_I *
                     np.sign(tau_pi_trial[I] - X[I]) / (1 - omega_I))
        z[I] += delta_lamda_I
        alpha[I] += delta_lamda_I * np.sign(tau_pi_trial[I] - X[I])

        Y_I = 0.5 * self.E_b * (s_n1[I] - xs_pi[I]) ** 2
        omega[I] += (
            delta_lamda_I *
            (1 - omega_I) ** self.c *
            (Y_I / self.S) ** self.r
        )
        sig[..., 1] = (1 - omega) * self.E_b * (s_n1 - xs_pi)
        omega_I = omega[I]
        O = np.where(np.fabs(1. - omega_I) > 1e-5)
        IO = tuple([I[o][O] for o in range(len(I))])
        omega_IO = omega[IO]
        D_ed_IO = (
            self.E_b * (1 - omega_IO) - ((1 - omega_IO) * self.E_b **
                                         2) / (self.E_b + (self.gamma + self.K) * (1 - omega_IO))
            - ((1 - omega_IO) ** self.c * (self.E_b ** 2) * ((Y_I[O] / self.S) ** self.r)
               * np.sign(tau_pi_trial[I][O] - X[I][O]) * (s_n1[I][O] - xs_pi[I][O])) / ((self.E_b / (1 - omega_IO)) + self.gamma + self.K)
        )
        D[..., 1, 1] = (1 - omega) * self.E_b
        IO11 = IO + (1, 1)
        D[IO11] = D_ed_IO

        return sig, D

    tree_view = ui.View(
        ui.VGroup(ui.Group(ui.Item('E_m'),
                           ui.Item('E_f'),
                           ui.Item('E_b'),
                           ui.Item('tau_pi_bar'),
                           show_border=True, label='Stiffnesses and reversibility limit'),
                  ui.Group(ui.Item('gamma'),
                           ui.Item('K'),
                           show_border=True, label='Hardening parameters'),
                  ui.Group(ui.Item('S'),
                           ui.Item('r'),
                           ui.Item('c'),
                           show_border=True, label='Damage cumulation parameters'),
                  ui.Group(ui.Item('pressure'),
                           ui.Item('a'), show_border=True, label='Lateral Pressure')))

