
from os.path import join

from ibvpy.mathkit.mfn.mfn_line.mfn_line import MFnLineArray
from ibvpy.view.reporter import RInputRecord
from scipy.optimize import newton
from traits.api import \
    Instance, Str, \
    Float, on_trait_change,\
    Interface, provides, Range, Property, Button, \
    Array, WeakRef, observe, cached_property
import bmcs_utils.api as bu
from ibvpy.view.ui import BMCSLeafNode
import sympy as sp
import numpy as np


class IDamageFn(Interface):

    def __call__(self, k):
        '''get the value of the function'''

    def diff(self, k):
        '''get the first derivative of the function'''

    def get_f_trial(self, s, k):
        '''get the map of indexes with inelastic behavior'''


class DamageFn(BMCSLeafNode):

    mats = WeakRef

    kappa_0 = bu.Float(0.0004,
                MAT=True,
                symbol="s_0",
                desc="elastic strain limit",
                unit='mm')

    E_name = Str('E')
    '''Name of the stiffness variable in the material model'''

    def get_f_trial(self, eps_eq_Em, kappa_Em):
        k_Em = np.copy(kappa_Em)
        k_Em[k_Em < self.kappa_0] = self.kappa_0
        return np.where(eps_eq_Em >= k_Em)

    plot_min = bu.Float(1e-9)
    plot_max_ = bu.Float(1e-2)
    # name of the trait controlling the plot range
    x_max_name = bu.Str('s_max')
    # if the parameter database is set get the values from there
    plot_max = Property(bu.Float)
    def _get_plot_max(self):
        if self.mats:
            return getattr(self.mats,self.x_max_name)
        else:
            return self.plot_max_

    def plot(self, ax, **kw):
        ax_omega, ax_d_omega = ax
        n_vals = 200
        kappa_range = np.linspace(self.plot_min, self.plot_max, n_vals)
        omega_range = np.zeros_like(kappa_range)
        I = kappa_range > self.kappa_0
        if len(I) > 0:
            omega_range[I] = self.__call__(kappa_range[I])
        color = kw.pop('color', 'green')
        ax_omega.plot(kappa_range, omega_range, color=color, **kw)
        ax_d_omega.plot(kappa_range, self.diff(kappa_range),
                        color='gray', linestyle='dashed', **kw)
        ax_omega.set_xlabel(r'$\kappa$ [mm]')
        ax_omega.set_ylabel(r'$\omega$ [-]')
        ax_d_omega.set_ylabel(r'$\mathrm{d} \omega / \mathrm{d} \kappa$ [-/mm]')

    def subplots(self, fig):
        ax_omega = fig.subplots(1,1)
        ax_d_omega = ax_omega.twinx()
        return ax_omega, ax_d_omega

    def update_plot(self, axes):
        self.plot(axes)

class DamageFnInjectSymbExpr(bu.InjectSymbExpr, DamageFn):
    '''
    Damage function derived symbolically
    '''
    def __call__(self, kappa):
        return self.symb.get_omega_(kappa)

    def diff(self, kappa):
        return self.symb.get_d_omega_(kappa)

    latex_eq = Str(None)

    def _repr_latex_(self):
        return self.symb.omega_._repr_latex_()

    def update_plot(self, axes):
        self.plot(axes)

class GfDamageFnSymbExpr(bu.SymbExpr):
    '''Self regularized damage function'''
    c_1, c_2 = sp.symbols('c_1, c_2', positive=True)
    E, kappa_0 = sp.symbols(r'E_b, kappa_0', positive=True)
    kappa, G_f = sp.symbols(r'kappa, G_f', positive=True)
    f = c_1 * sp.exp(-c_2*kappa)
    f_kappa_0 = sp.Piecewise(
        (1, kappa < kappa_0),
        (f.subs(kappa, kappa - kappa_0), True)
    )
    sig_s = E * kappa * f_kappa_0
    subs_c_1 = sp.solve({sp.Eq(sig_s.subs(kappa, kappa_0), E * kappa_0)}, {c_1})
    G_f_ = sp.simplify( sp.integrate(sig_s.subs(subs_c_1), (kappa,0,sp.oo)) ).factor()
    _, subs_c_2 = sp.solve(sp.Eq(G_f, G_f_), {c_2})
    g_kappa_ = sp.simplify( f_kappa_0.subs(c_2, subs_c_2).subs(c_1,1) )
    omega_ = 1 - g_kappa_
    d_omega_ = omega_.diff(kappa)

    symb_model_params = ['kappa_0', 'E', 'G_f']

    # List of expressions for which the methods `get_`
    symb_expressions = [
        ('omega_', ('kappa',)),
        ('d_omega_', ('kappa',)),
    ]

@provides(IDamageFn)
class GfDamageFn(DamageFnInjectSymbExpr):
    name = 'self-regularized'
    symb_class = GfDamageFnSymbExpr

    G_f = bu.Float(0.1, MAT=True, symbol="G_\mathrm{f}", unit='N/mm',
                desc="derivative of the damage function at the onset of damage" )

    min_G_f = Property(bu.Float, depends_on='state_changed')
    def _get_min_G_f(self):
        return self.E * self.kappa_0**2 / 2

    E_ = bu.Float(10000.0, MAT=True, label="E", desc="Young's modulus")

    E = Property(bu.Float)
    def _get_E(self):
        if self.mats:
            return getattr(self.mats,self.E_name)
        else:
            return self.E_

    def __call__(self, kappa):
        return self.symb.get_omega_(kappa)

    def diff(self, kappa):
        return self.symb.get_d_omega_(kappa)

    ipw_view = bu.View(
        bu.Item('kappa_0', latex=r'\kappa_0 [\mathrm{mm}]'),
        bu.Item('G_f', latex=r'G_\mathrm{f} [\mathrm{N/mm}]'),
        bu.Item('E', latex=r'E [\mathrm{MPa}]', readonly=True),
        bu.Item('min_G_f', latex=r'\min(G_\mathrm{f})', readonly=True),
        bu.Item('plot_max', readonly=True),
    )


class ExpSlopeDamageFnSymbExpr(bu.SymbExpr):

    kappa, kappa_0 = sp.symbols(r'\kappa, \kappa_0')
    kappa_f = sp.symbols(r'kappa_\mathrm{f}')
    omega_kappa_ =  1 - (kappa_0 / kappa * sp.exp(-(kappa-kappa_0)/(kappa_f-kappa_0) ) )
    omega_ = sp.Piecewise( (0, kappa <= kappa_0), (omega_kappa_, True) )

    d_omega_ = omega_.diff(kappa)

    symb_model_params = ['kappa_0', 'kappa_f']

    # List of expressions for which the methods `get_`
    symb_expressions = [
        ('omega_', ('kappa',)),
        ('d_omega_', ('kappa',)),
    ]

@provides(IDamageFn)
class ExpSlopeDamageFn(DamageFnInjectSymbExpr):

    name = 'Exponential with slope'
    symb_class = ExpSlopeDamageFnSymbExpr

    kappa_f = bu.Float(0.001, MAT=True, symbol=r'\kappa_\mathrm{f}', unit='mm/mm',
                desc="derivative of the damage function at the onset of damage" )

    ipw_view = bu.View(
        bu.Item('kappa_0', latex=r'\kappa_0'),
        bu.Item('kappa_f', latex=r'\kappa_\mathrm{f}'),
    )


class AbaqusDamageFnSymbExpr(bu.SymbExpr):
    kappa, kappa_0 = sp.symbols(r'\kappa, \kappa_0')
    alpha, kappa_u = sp.symbols('alpha, kappa_u')
    g2_kappa_ = (kappa_0 / kappa * (1 - (1 - sp.exp(-alpha * ((kappa - kappa_0) / (kappa_u - kappa_0)))) /
                            (1 - sp.exp(-alpha))))
    g2_ = sp.Piecewise((1, kappa < kappa_0), (g2_kappa_, kappa < kappa_u), (0, True))
    omega_kappa_ =  1 - g2_
    omega_ = sp.Piecewise( (0, kappa <= kappa_0), (omega_kappa_, True) )
    d_omega_ = omega_.diff(kappa)

    symb_model_params = ['kappa_0', 'kappa_u', 'alpha']

    # List of expressions for which the methods `get_`
    symb_expressions = [
        ('omega_', ('kappa',)),
        ('d_omega_', ('kappa',)),
    ]

@provides(IDamageFn)
class AbaqusDamageFn(DamageFnInjectSymbExpr):

    name = 'Abaqus damage function'
    symb_class = AbaqusDamageFnSymbExpr

    latex_eq = Str(r'''Damage function (Abaqus)
        \begin{align}
        \omega = g(\kappa) = 1 -\left(\frac{s_0}{\kappa}\right)\left[ 1 - \frac{1 - \exp(- \alpha(\frac{\kappa - s_0}{s_u - s_0})}{1 - \exp(-\alpha)}  \right]
        \end{align}
        where $\kappa$ is the state variable representing 
        the maximum slip that occurred so far in
        in the history of loading.
        ''')

    kappa_u = bu.Float(0.003, MAT=True, symbol="kappa_u", unit='mm',
                desc="parameter of the damage function",
    )

    alpha = bu.Float(0.1, MAT=True, symbol=r"\alpha",
                  desc="parameter controlling the slope of damage",
                  unit='-')

    ipw_view = bu.View(
        bu.Item('kappa_0', latex=r'\kappa_0'),
        bu.Item('kappa_u', latex=r'\kappa_\mathrm{u}'),
        bu.Item('alpha'),
    )

class LinearDamageFnSymbExpr(bu.SymbExpr):
    kappa, kappa_0 = sp.symbols(r'\kappa, \kappa_0')
    kappa_u = sp.symbols(r'\kappa_u')
    omega_ = sp.Piecewise(
        (0, kappa < kappa_0),
        (1/(kappa_u-kappa_0)*(kappa-kappa_0), kappa < kappa_u),
        (1, True)
    )
    d_omega_ = omega_.diff(kappa)

    symb_model_params = ['kappa_0', 'kappa_u']

    # List of expressions for which the methods `get_`
    symb_expressions = [
        ('omega_', ('kappa',)),
        ('d_omega_', ('kappa',)),
    ]

@provides(IDamageFn)
class LinearDamageFn(DamageFnInjectSymbExpr):

    name = 'Linear damage function'
    symb_class = LinearDamageFnSymbExpr

    kappa_0 = 0.01

    kappa_u = bu.Float(0.03, MAT=True, symbol="kappa_u", unit='mm',
                desc="parameter of the damage function",
    )

    ipw_view = bu.View(
        bu.Item('kappa_0', latex=r'\kappa_0'),
        bu.Item('kappa_u', latex=r'\kappa_\mathrm{u}'),
    )

class WeibullDamageFnSymbExpr(bu.SymbExpr):
    kappa = sp.symbols(r'\kappa')
    m, lambda_ = sp.symbols(r'm, \lambda')
    beta_ = 1 / lambda_
    weibull_cdf = (1 - sp.exp(-(beta_ * kappa)**m))
    omega_ = sp.Piecewise(
        (0, kappa < 0),
        (weibull_cdf, True)
    )
    d_omega_ = omega_.diff(kappa)

    symb_model_params = ['m', 'lambda_']

    # List of expressions for which the methods `get_`
    symb_expressions = [
        ('omega_', ('kappa',)),
        ('d_omega_', ('kappa',)),
    ]

@provides(IDamageFn)
class WeibullDamageFn(DamageFnInjectSymbExpr):

    name = 'Linear damage function'
    symb_class = WeibullDamageFnSymbExpr

    kappa_0 = bu.Float(1e-5)
    # the inelastic regime starts right from the beginning

    lambda_ = bu.Float(0.03, MAT=True, symbol="lambda_", unit='mm',
                desc="Weibull scale parameter",
    )

    m = bu.Float(5, MAT=True, symbol="m",
                desc="Weibull shape parameter",
    )

    ipw_view = bu.View(
        bu.Item('lambda_', latex=r'\lambda'),
        bu.Item('m', latex=r'm'),
    )

@provides(IDamageFn)
class LiDamageFn(DamageFn):

    name = 'Two parameter damage'

    latex_eq = Str(r'''Damage function (Li)
        \begin{align}
        \omega = g(\kappa) = \frac{\alpha_1}{1 + \exp(-\alpha_2 \kappa + 6 )}
        \end{align}
        where $\kappa$ is the state variable representing 
        the maximum slip that occurred so far in
        in the history of loading.
        ''')

    alpha_1 = bu.Float(value=1, MAT=True,
                    symbol=r'\alpha_1',
                    unit='-',
                    desc="parameter controlling the shape of the damage function")

    alpha_2 = bu.Float(2000., MAT=True,
                    symbol=r'\alpha_2',
                    unit='-',
                    desc="parameter controlling the shape of the damage function")

    def __call__(self, kappa):
        alpha_1 = self.alpha_1
        alpha_2 = self.alpha_2
        s_0 = self.s_0
        omega = np.zeros_like(kappa, dtype=np.float_)
        d_idx = np.where(kappa >= s_0)[0]
        k = kappa[d_idx]
        omega[d_idx] = 1. / \
            (1. + np.exp(-1. * alpha_2 * (k - s_0) + 6.)) * alpha_1
        return omega

    def diff(self, kappa):
        alpha_1 = self.alpha_1
        alpha_2 = self.alpha_2
        s_0 = self.s_0
        return ((alpha_1 * alpha_2 *
                 np.exp(-1. * alpha_2 * (kappa - s_0) + 6.)) /
                (1 + np.exp(-1. * alpha_2 * (kappa - s_0) + 6.)) ** 2)

    ipw_view = bu.View(
        bu.Item('s_0'),
        bu.Item('alpha_1', editor=bu.FloatRangeEditor(low=0,high=1)),
        bu.Item('alpha_2'),
    )


@provides(IDamageFn)
class MultilinearDamageFn(DamageFn):

    name = 'Multilinear damage function'

    s_data = bu.Str('1', tooltip='Comma-separated list of strain values',
                 MAT=True, unit='mm', symbol='s',
                 desc='slip values',
                 )

    omega_data = bu.Str('1', tooltip='Comma-separated list of damage values',
                     MAT=True, unit='-', symbol=r'\omega',
                     desc='shear stress values',
                     )

    s_omega_table = Property

    def _set_s_omega_table(self, data):
        s_data, omega_data = data
        if len(s_data) != len(omega_data):
            raise ValueError('s array and tau array must have the same size')
        self.damage_law.set(xdata=s_data,
                            ydata=omega_data)

    @observe('state_changed')
    def _update_damage_law(self, event=None):
        s_data = np.fromstring(self.s_data, dtype=np.float_, sep=',')
        omega_data = np.fromstring(self.omega_data, dtype=np.float_, sep=',')
        s_data = np.hstack([[0, self.s_0], s_data])
        omega_data = np.hstack([[0, 0], omega_data])
        n = np.min([len(s_data), len(omega_data)])
        self.damage_law.set(xdata=s_data[:n], ydata=omega_data[:n])

    damage_law = Instance(MFnLineArray)

    def _damage_law_default(self):
        return MFnLineArray(
            xdata=[0.0, 1.0],
            ydata=[0.0, 0.0],
            plot_diff=False)

    def __call__(self, kappa):
        shape = kappa.shape
        return self.damage_law(kappa.flatten()).reshape(*shape)

    def diff(self, kappa):
        shape = kappa.shape
        return self.damage_law.diff(kappa.flatten()).reshape(*shape)

    ipw_view = bu.View(
        bu.Item('s_0'),
        bu.Item('s_data'),
        bu.Item('omega_data'),
    )

class GfDamageFn2(DamageFn):
    '''Class defining the damage function coupled with the fracture
    energy of a cohesive crack model.
    '''
    name = 'damage function Gf'

    L_s = bu.Float(1.0, MAT=True, label="L_s",
                   desc="Length of the softening zone")

    E_ = bu.Float(34000.0, MAT=True, label="E", desc="Young's modulus")

    E = Property(bu.Float)
    def _get_E(self):
        if self.mats:
            return getattr(self.mats,self.E_name)
        else:
            return self.E_

    f_t = bu.Float(4.5, MAT=True, label="f_t", desc="Tensile strength")

    f_t_Em = Array(np.float_, value=None)

    G_f = bu.Float(0.004, MAT=True, label="G_f", desc="Fracture energy",)

    s_0 = Property(bu.Float)
    def _get_s_0(self):
        return self.f_t / self.E

    eps_ch = Property(bu.Float)

    def _get_eps_ch(self):
        return self.G_f / self.f_t

    ipw_view = bu.View(
                bu.Item('L_s', latex=r'L_\mathrm{s}'),
                bu.Item('f_t', latex=r'f_\mathrm{t}'),
                bu.Item('G_f', latex=r'G_\mathrm{f}'),
                bu.Item('E', readonly=True),
                bu.Item('s_0', latex=r's_0', readonly=True),
            )

    def __call__(self, kappa):
        L_s = self.L_s
        f_t = self.f_t
        G_f = self.G_f
        E = self.E
        s_0 = self.s_0
        return (
            1 - f_t * np.exp(-f_t * (kappa - s_0) * L_s / G_f)
            / (E * kappa)
        )

    def diff(self, kappa):
        L_s = self.L_s
        f_t = self.f_t
        G_f = self.G_f
        E = self.E
        s_0 = self.s_0
        return (
            f_t * np.exp(L_s * (s_0 - kappa) * f_t / G_f)
            / (E * G_f * kappa**2) * (G_f + L_s * kappa * f_t)
        )

@provides(IDamageFn)
class FRPDamageFn(DamageFn):

    name = 'FRP damage function'

    B = bu.Float(10.4,
              MAT=True,
              symbol="B",
              unit='mm$^{-1}$',
              desc="parameter controlling the damage maximum stress level")

    Gf = bu.Float(1.19,
               MAT=True,
               symbol="G_\mathrm{f}",
               unit='N/mm',
               desc="fracture energy")

    E_bond = bu.Float(0.0)

    E_b = Property(Float)

    def _get_E_b(self):
        return self.mats.E_b

    def _set_E_b(self, value):
        self.E_bond = value
        self.mats.E_b = value

    @observe('B, Gf')
    def _update_dependent_params(self, event=None):
        self.E_b = 1.734 * self.Gf * self.B ** 2.0
        # calculation of s_0, implicit function solved using Newton method

        def f_s(s_0): return s_0 / \
            (np.exp(- self.B * s_0) - np.exp(-2.0 * self.B * s_0)) - \
            2.0 * self.B * self.Gf / self.E_b
        self.s_0 = newton(f_s, 0.00000001, tol=1e-5, maxiter=20)

    def __call__(self, kappa):

        b = self.B
        Gf = self.Gf
        Eb = self.E_b  # 1.734 * Gf * b**2
        s_0 = self.s_0
        # calculation of s_0, implicit function solved using Newton method

#         def f_s(s_0): return s_0 / \
#             (np.exp(-b * s_0) - np.exp(-2.0 * b * s_0)) - 2.0 * b * Gf / Eb
#         s_0 = newton(f_s, 0.00000001, tol=1e-5, maxiter=20)

        omega = np.zeros_like(kappa, dtype=np.float_)
        I = np.where(kappa >= s_0)[0]
        kappa_I = kappa[I]

        omega[I] = 1 - \
            (2.0 * b * Gf * (np.exp(-b * kappa_I)
                             - np.exp(-2.0 * b * kappa_I))) / (kappa_I * Eb)

        return omega

    def diff(self, kappa):

        nz_ix = np.where(kappa != 0.0)[0]

        b = self.B
        Gf = self.Gf
        Eb = 1.734 * Gf * b**2

        domega_dkappa = np.zeros_like(kappa)
        kappa_nz = kappa[nz_ix]
        domega_dkappa[nz_ix] = (
            (2.0 * b * Gf *
             (np.exp(-b * kappa_nz) -
              np.exp(-2.0 * b * kappa_nz))
             ) / (Eb * kappa_nz**2.0) -
            (2.0 * b * Gf *
             (-b * np.exp(-b * kappa_nz) +
              2.0 * b * np.exp(-2.0 * b * kappa_nz))) /
            (Eb * kappa_nz)
        )
        return domega_dkappa

    latex_eq = r'''Damage function (FRP)
        \begin{align}
        \omega = g(\kappa) = 
        1 - {\frac {{\exp(-2\,Bs)}-{\exp(-Bs)}}{Bs}}
        \end{align}
        where $\kappa$ is the state variable representing 
        the maximum slip that occurred so far in
        in the history of loading.
        '''

    ipw_view = bu.View(
        bu.Item('s_0', readonly=True),
        bu.Item('E_bond', readonly=True),
        bu.Item('B'),
        bu.Item('Gf'),
    )

