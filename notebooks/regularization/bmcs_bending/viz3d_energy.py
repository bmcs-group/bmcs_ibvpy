'''
Created on May 30, 2018

@author: rch
'''

from scipy import interpolate as ip
from ibvpy.view.plot2d import Viz2D, Vis2D
from scipy.integrate import cumtrapz
import numpy as np
import traits.api as tr

class ForceDisplacement(Vis2D):

    tstep = tr.WeakRef
    Fw = tr.Tuple()

    def _Fw_default(self):
        return ([0], [0])

    def update(self):
        model = self.tstep
        record_dofs = model.control_bc.dofs
        U_ti = model.hist.U_t
        F_ti = model.hist.F_t
        F = np.sum(F_ti[:, record_dofs], axis=1)
        w = U_ti[:, record_dofs[0]]
        self.Fw = F, w


class Vis2DEnergy(Vis2D):

    U_bar_t = tr.List()
    Y_Em_t = tr.List()

    tstep = tr.WeakRef

    def setup(self):
        self.U_bar_t = []
        self.Y_Em_t = []

    def update(self):
        tstep = self.tstep
        mats = tstep.mm
        xdomain = tstep.xdomain
        U = tstep.U_k
        t = tstep.t_n1
        fets = xdomain.fets
        n_c = fets.n_nodal_dofs
        U_Ia = U.reshape(-1, n_c)
        U_Eia = U_Ia[xdomain.I_Ei]
        d = xdomain.integ_factor  # thickness
        eps_Emab = np.einsum(
            'Eimabc,Eic->Emab',
            xdomain.B_Eimabc, U_Eia
        )
        sig_Emab, _ = mats.get_corr_pred(
            eps_Emab, t, **tstep.fe_domain[0].state_n)
        w_m = fets.w_m
        det_J_Em = xdomain.det_J_Em
        U_bar = d / 2.0 * np.einsum(
            'm,Em,Emab,Emab',
            w_m, det_J_Em, sig_Emab, eps_Emab
        )
        self.U_bar_t.append(U_bar)

        Y_Em_t = d / 2.0 * np.einsum(
            'm,Em,Emab,abcd,Emcd->Em',
            w_m, det_J_Em, eps_Emab, mats.D_abcd, eps_Emab
        )
        self.Y_Em_t.append(Y_Em_t)

    def get_t(self):
        return self.tstep.hist.t

    def get_w(self):
        _, w = self.tstep.hist['Fw'].Fw
        return w

    def get_W_t(self):
        P, w = self.tstep.hist['Fw'].Fw
        return cumtrapz(P, w, initial=0)

    G_omega_t = tr.Property
    def _get_G_omega_t(self):
        tstep = self.tstep
        Y_Em_t = np.array(self.Y_Em_t, dtype=np.float_)
        hist = tstep.hist
        t = hist.t
        omega_Em_t = np.array([state[0]['omega'] for state in hist.state_vars])
        # G_omega_Em_t = cumtrapz(Y_Em_t, omega_Em_t, axis=0, initial = 0)
        # return np.sum(G_omega_Em_t, axis=(1,2))

        d_omega_Em_t = (
            (omega_Em_t[1:] - omega_Em_t[:-1]) / (t[1:] - t[:-1])[:,None,None]
        )
        dG_omega_t = np.hstack([
            np.einsum('tEm,tEm->t', Y_Em_t[:-1], d_omega_Em_t), [0]])
        return cumtrapz(dG_omega_t, t, initial=0)

    G_t = tr.Property

    def _get_G_t(self):
        U_bar_t = np.array(self.U_bar_t, dtype=np.float_)
        W_t = self.get_W_t()
        G = W_t - U_bar_t
        return G

    def get_dG_t(self):
        t = self.get_t()
        G = self.get_G_t()
        tck = ip.splrep(t, G, s=0, k=1)
        return ip.splev(t, tck, der=1)


class Vis2DCrackBand(Vis2D):

    tstep = tr.WeakRef

    X_E = tr.Array(np.float_)
    eps_t = tr.List
    sig_t = tr.List
    a_t = tr.List

    def setup(self):
        self.X_E = []
        self.eps_t = []
        self.sig_t = []
        self.a_t = []
        tstep = self.tstep
        xdomain = tstep.xdomain
        crack_band = xdomain.mesh[0, :]  # bt.n_a + 1:]
        E = crack_band.elems
        X = crack_band.dof_X
        X_E = np.average(X[:, :, 1], axis=1)
        self.X_E = X_E

    def update(self):
        tstep = self.tstep
        U = tstep.U_k
        t = tstep.t_n1
        xdomain = tstep.xdomain
        mats = tstep.mm
        fets = xdomain.fets
        n_c = fets.n_nodal_dofs
        U_Ia = U.reshape(-1, n_c)
        U_Eia = U_Ia[xdomain.I_Ei]
        eps_Enab = np.einsum(
            'Einabc,Eic->Enab', xdomain.B_Eimabc, U_Eia
        )
        sig_Enab, _ = mats.get_corr_pred(
            eps_Enab, t, ** tstep.fe_domain[0].state_n
        )
        crack_band = xdomain.mesh[0, :]  # bt.n_a + 1:]
        E = crack_band.elems
        eps_Ey = eps_Enab[E, :, 0, 0]
        eps_E1 = np.average(eps_Ey, axis=1)
        sig_Ey = sig_Enab[E, :, 0, 0]
        sig_E1 = np.average(sig_Ey, axis=1)
        kappa_0 = mats.omega_fn_.kappa_0
        eps_thr = eps_E1 - kappa_0
        a_idx = np.argwhere(eps_thr < 0)
        if len(a_idx) > 0:
            x_1 = self.X_E[0]
            x_a = self.X_E[a_idx[0][0]]
            self.a_t.append(x_a - x_1)
        else:
            self.a_t.append(0)

        self.eps_t.append(eps_E1)
        self.sig_t.append(sig_E1)

    def get_t(self):
        return np.array(self.tstep.hist.t, dtype=np.float_)

    def get_a_x(self):
        return self.X_E

    def get_eps_t(self, t_idx):
        return self.eps_t[t_idx]

    def get_sig_t(self, t_idx):
        return self.sig_t[t_idx]

    def get_a_t(self):
        return np.array(self.a_t, dtype=np.float_)

    def get_da_dt(self):
        a = self.get_a_t()
        t = self.get_t()
        tck = ip.splrep(t, a, s=0, k=1)
        return ip.splev(t, tck, der=1)

