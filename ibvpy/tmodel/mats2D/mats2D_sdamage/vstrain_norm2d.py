
from traits.api import HasTraits, WeakRef
from ibvpy.view.ui.bmcs_tree_node import BMCSTreeNode
import bmcs_utils.api as bu
import numpy as np
import sympy as sp

class StrainNorm2D(BMCSTreeNode):

    mats = WeakRef

    def get_eps_eq(self, eps_Emef, kappa_Em):
        raise NotImplementedError

    def get_deps_eq(self, eps_Emef):
        raise NotImplementedError

    def update_plot(self, ax):
        n_i = 100
        eps_i = np.linspace(-3, 3, n_i)
        eps_11, eps_22 = np.meshgrid(eps_i, eps_i)
        eps_11_ = eps_11.flatten()
        eps_22_ = eps_22.flatten()
        eps_ief = np.zeros((len(eps_11_), 2, 2))
        eps_ief[:, 0, 0] = eps_11_
        eps_ief[:, 1, 1] = eps_22_
        kappa_i = np.zeros_like(eps_11_)
        eps_eq_i = self.get_eps_eq(eps_ief, kappa_i)
        eps_eq_ij = eps_eq_i.reshape(n_i,n_i)
        ax.contour(eps_11, eps_22, eps_eq_ij) # , [1]) # , label=r'$\kappa = 1$')
        ax.plot([0,0],[-3,3],color='black', lw=0.3)
        ax.plot([-3,3],[0,0],color='black', lw=0.3)
        ax.set_xlabel(r'$\varepsilon_{1}$')
        ax.set_ylabel(r'$\varepsilon_{2}$')
        ax.axis('equal')

class SN2DRankine(StrainNorm2D):
    '''
    Computes principal strains and makes a norm of their positive part
    '''
    name = 'Rankine strain norm'

    def get_eps_eq(self, eps_Emef, kappa_Em):

        eps_11 = eps_Emef[..., 0, 0]
        eps_22 = eps_Emef[..., 1, 1]
        eps_12 = eps_Emef[..., 0, 1]
        eps_eq_Em = (
            0.5 * (eps_11 + eps_22) +
            np.sqrt(((eps_11 - eps_22) / 2.0)**2.0 + eps_12**2.0)
        )
        e_Em = np.concatenate(
            (eps_eq_Em[..., None], kappa_Em[..., None]), axis=-1
        )
        eps_eq = np.max(e_Em, axis=-1)
        return eps_eq

    def get_deps_eq(self, eps_Emef):
        eps11 = eps_Emef[..., 0, 0]
        eps22 = eps_Emef[..., 1, 1]
        eps12 = eps_Emef[..., 0, 1]
        eps_11_22 = eps11 - eps22

        denom = 2. * np.sqrt(eps_11_22 * eps_11_22 + 4.0 * eps12 * eps12)
        factor = np.zeros_like(denom)
        nz_idx = np.where(denom != 0.0)
        factor[nz_idx] = 1. / denom[nz_idx]
        df_trial1 = factor * np.array([[eps11 - eps22, 4.0 * eps12],
                                       [4.0 * eps12, eps22 - eps11]])
        return (np.einsum('ab...->...ab', df_trial1) +
                0.5 * np.identity(2)[None, :, :])

class SN2MasarsExpr(bu.SymbExpr):

    eps_11, eps_12, eps_22 = sp.symbols(
        r'\varepsilon_{11}, \varepsilon_{22}, \varepsilon_{12}')
    eps_tns = sp.Matrix([[eps_11, eps_12],[eps_12, eps_22]])
    eps_1, eps_2 = eps_tns.eigenvals()
    def pos(x):
        return sp.Piecewise((0, x<0),
                            (x, True))
    eps_1_pos, eps_2_pos = pos(eps_1), pos(eps_2)
    eps_eq = sp.simplify(sp.sqrt( eps_1_pos**2 + eps_2_pos**2))

    d_eps_eq_11 = eps_eq.diff(eps_11)
    d_eps_eq_12 = eps_eq.diff(eps_12)
    d_eps_eq_22 = eps_eq.diff(eps_22)

    symb_model_params = []

    # List of expressions for which the methods `get_`
    symb_expressions = [
        ('eps_eq', ('eps_11', 'eps_22', 'eps_12')),
        ('d_eps_eq_11', ('eps_11', 'eps_22', 'eps_12')),
        ('d_eps_eq_12', ('eps_11', 'eps_22', 'eps_12')),
        ('d_eps_eq_22', ('eps_11', 'eps_22', 'eps_12')),
    ]

class SN2DMasars(StrainNorm2D,bu.InjectSymbExpr):
    '''
    Computes principal strains and makes a norm of their positive part
    '''
    symb_class = SN2MasarsExpr
    name = 'Masars strain norm'

    def get_eps_eq(self, eps_Emef, kappa_Em):
        eps_11 = eps_Emef[..., 0, 0]
        eps_22 = eps_Emef[..., 1, 1]
        eps_12 = eps_Emef[..., 0, 1]
        eps_eq_Em = self.symb.get_eps_eq(eps_11, eps_22, eps_12)
        e_Em = np.concatenate(
            (eps_eq_Em[..., None], kappa_Em[..., None]), axis=-1
        )
        eps_eq = np.max(e_Em, axis=-1)
        return eps_eq

    def get_deps_eq(self, eps_Emef):
        eps_11 = eps_Emef[..., 0, 0]
        eps_22 = eps_Emef[..., 1, 1]
        eps_12 = eps_Emef[..., 0, 1]
        d_eps_eq_11 = self.symb.get_d_eps_eq_11(eps_11, eps_22, eps_12)
        d_eps_eq_22 = self.symb.get_d_eps_eq_22(eps_11, eps_22, eps_12)
        d_eps_eq_12 = self.symb.get_d_eps_eq_12(eps_11, eps_22, eps_12)
        d_eps_eq_efEm = np.array([[d_eps_eq_11, d_eps_eq_12],
                                  [d_eps_eq_12, d_eps_eq_22]])
        return np.einsum('ef...->...ef', d_eps_eq_efEm)

class SN2EnergyExpr(bu.SymbExpr):

    eps_11, eps_12, eps_22 = sp.symbols(
        r'\varepsilon_{11}, \varepsilon_{22}, \varepsilon_{12}')
    eps_tns = sp.Matrix([[eps_11, eps_12],[eps_12, eps_22]])
    eps_1, eps_2 = eps_tns.eigenvals()
    def pos(x):
        return sp.Piecewise((0, x<0),
                            (x, True))
    eps_1_pos, eps_2_pos = pos(eps_1), pos(eps_2)
    eps_eq = sp.simplify(sp.sqrt( eps_1_pos**2 + eps_2_pos**2))

    d_eps_eq_11 = eps_eq.diff(eps_11)
    d_eps_eq_12 = eps_eq.diff(eps_12)
    d_eps_eq_22 = eps_eq.diff(eps_22)

    symb_model_params = []

    # List of expressions for which the methods `get_`
    symb_expressions = [
        ('eps_eq', ('eps_11', 'eps_22', 'eps_12')),
        ('d_eps_eq_11', ('eps_11', 'eps_22', 'eps_12')),
        ('d_eps_eq_12', ('eps_11', 'eps_22', 'eps_12')),
        ('d_eps_eq_22', ('eps_11', 'eps_22', 'eps_12')),
    ]

class SN2DEnergy(StrainNorm2D,bu.InjectSymbExpr):
    '''
    Computes the energy norm

    \kappa = \sqrt{ \varepsilon_{ab} D_{abcd} \varepsilon_{cd}}

    the derivative ...
    \dfrac{\partial \kappa}{\partial \varepsilon_{cd}}
    =
    \dfrac{1}{\kappa} D_{abcd} \varepsilon_{cd}
    '''
    symb_class = SN2EnergyExpr
    name = 'Energy norm'

    def get_eps_eq(self, eps_Emef, kappa_Em):
        D_abcd = self.mats.D_abcd
        eps_eq_Em = np.sqrt( np.einsum('...ab,abcd,...cd->...', eps_Emef, D_abcd, eps_Emef))
        e_Em = np.concatenate(
            (eps_eq_Em[..., None], kappa_Em[..., None]), axis=-1
        )
        eps_eq = np.max(e_Em, axis=-1)
        return eps_eq

    def get_deps_eq(self, eps_Emef):
        D_abcd = self.mats.D_abcd
        eps_eq_Em = np.sqrt( np.einsum('...ab,abcd,...cd->...', eps_Emef, D_abcd, eps_Emef))
        deps_eq = np.einsum('...,...abcd,...cd->...ab', 1 / eps_eq_Em, D_abcd, eps_Emef)
        return deps_eq
