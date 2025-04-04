'''
Created on Feb 14, 2019

@author: rch
'''

from ibvpy.tmodel import MATSEval
from ibvpy.sim.i_tmodel import ITModel

import numpy as np
import traits.api as tr
import bmcs_utils.api as bu


@tr.provides(ITModel)
class MATS1D5DPCumPress(MATSEval):

    node_name = 'Pressure sensitive cumulative damage plasticity'

    E_N = bu.Float(30000, label='E_N',
                   desc='Normal stiffness of the interface',
                   MAT=True,
                   enter_set=True, auto_set=False)

    E_T = bu.Float(12900, label='E_T',
                   desc='Shear modulus of the interface',
                   MAT=True,
                   enter_set=True, auto_set=False)

    gamma = bu.Float(55.0, label='gamma',
                     desc='Kinematic Hardening Modulus',
                     MAT=True,
                     enter_set=True, auto_set=False)

    K = bu.Float(11, label='K',
                 desc='Isotropic hardening modulus',
                 MAT=True,
                 enter_set=True, auto_set=False)

    S = bu.Float(0.005, label='S',
                 desc='Damage accumulation parameter',
                 MAT=True,
                 enter_set=True, auto_set=False)

    r = bu.Float(1, label='r',
                 desc='Damage accumulation parameter',
                 MAT=True,
                 enter_set=True, auto_set=False)

    c = bu.Float(1, Label='c',
                 desc='Damage accumulation parameter',
                 MAT=True,
                 enter_set=True, auto_set=False)
    m = bu.Float(0.3, label='m',
                 desc='Lateral Pressure Coefficient',
                 MAT=True,
                 enter_set=True, auto_set=False)

    tau_bar = bu.Float(4.2, label='tau_bar',
                       desc='Reversibility limit',
                       MAT=True,
                       enter_set=True, auto_set=False)

    state_var_shapes = dict(s_pi=(),
                            alpha=(),
                            z=(),
                            omega=())

    D_rs = tr.Property(depends_on='E_N,E_T')

    @tr.cached_property
    def _get_D_rs(self):
        print('recalculating D_rs')
        return np.array([[self.E_T, 0],
                         [0, self.E_N]], dtype=np.float64)

    def init(self, s_pi, alpha, z, omega):
        r'''
        Initialize the state variables.
        '''
        s_pi[...] = 0
        alpha[...] = 0
        z[...] = 0
        omega[...] = 0

    algorithmic = tr.Bool(True)

    def get_corr_pred(self, u_r, t_n, s_pi, alpha, z, omega):

        s = u_r[..., 0]
        w = u_r[..., 1]
        # For normal direction
        H_w_N = np.array(w <= 0.0, dtype=np.float64)
        E_alg_N = H_w_N * self.E_N
        sig_N = E_alg_N * w
        # For tangential
        # Y = 0.5 * self.E_T * (u_T - s_pi)**2
        tau_pi_trial = self.E_T * (s - s_pi)
        Z = self.K * z
        X = self.gamma * alpha
        f = np.fabs(tau_pi_trial - X) - Z - self.tau_bar + self.m * sig_N
        # Identify inelastic material points
        # @todo: consider the usage of np.where()
        I = f > 1e-6
        sig_T = (1 - omega) * self.E_T * (s - s_pi)
        # Return mapping
        delta_lambda_I = (
            f[I] / (self.E_T / (1 - omega[I]) + self.gamma + self.K)
        )
        # Update all state variables
        s_pi[I] += (delta_lambda_I * 
                    np.sign(tau_pi_trial[I] - X[I]) / (1 - omega[I]))
        Y = 0.5 * self.E_T * (s - s_pi) ** 2
        omega[I] += (
            delta_lambda_I * 
            (1 - omega[I]) ** self.c * (Y[I] / self.S) ** self.r * (self.tau_bar / (self.tau_bar - self.m * sig_N[I]))
        )
        sig_T[I] = (1 - omega[I]) * self.E_T * (s[I] - s_pi[I])
        alpha[I] += delta_lambda_I * np.sign(tau_pi_trial[I] - X[I])
        z[I] += delta_lambda_I
        # Unloading stiffness
     
        E_alg_T = (1 - omega) * self.E_T
       
        # Consistent tangent operator
        if False:
            E_alg_T = (
                (1 - omega) * self.E_T - 
                (1 - omega) * self.E_T ** 2 / 
                (self.E_T + (self.gamma + self.K) * (1 - omega)) - 
                ((1 - omega) ** self.c * (self.E_T ** 2) * ((Y / self.S) ** self.r)
                 * np.sign(tau_pi_trial - X) * (s - s_pi)) / 
                ((self.E_T / (1 - omega)) + self.gamma + self.K)
            )

        
        if False:
            #print('DONT COME HERE')
            E_alg_T = (
                (1 - omega) * self.E_T - 
                ((self.E_T ** 2 * (1 - omega)) / 
                 (self.E_T + (self.gamma + self.K) * (1 - omega)))
                -
                ((1 - omega) ** self.c * 
                 (Y / self.S) ** self.r * 
                    self.E_T ** 2 * (s - s_pi) * self.tau_bar / 
                    (self.tau_bar - self.m * sig_N) * np.sign(tau_pi_trial - X)) / 
                (self.E_T / (1 - omega) + self.gamma + self.K)
            )

        sig = np.zeros_like(u_r)
        sig[..., 0] = sig_T
        sig[..., 1] = sig_N
        E_TN = np.einsum('abEm->Emab',
                         np.array(
                             [
                                 [E_alg_T, np.zeros_like(E_alg_T)],
                                 [np.zeros_like(E_alg_N), E_alg_N]
                             ])
                         )
#         print(' sig_N =',  sig_N)
        abc = open('sigNfortau10000lp5000.txt', 'a+', newline='\n')
        for e in range(len(sig_N)):
            abc.write('%f ' % sig_N[e][0])
        abc.write('\n')
        abc.close()
        # print('s_pi=', s_pi)
        return sig, E_TN

    def _get_var_dict(self):
        var_dict = super(MATS1D5DPCumPress, self)._get_var_dict()
        var_dict.update(
            slip=self.get_slip,
            s_el=self.get_s_el,
            shear=self.get_shear,
            omega=self.get_omega,
            s_pi=self.get_s_pi,
            alpha=self.get_alpha,
            z=self.get_z
        )
        return var_dict

    def get_slip(self, u_r, tn1, **state):
        return self.get_eps(u_r, tn1)[..., 0]

    def get_shear(self, u_r, tn1, **state):
        return self.get_sig(u_r, tn1, **state)[..., 0]

    def get_omega(self, u_r, tn1, s_pi, alpha, z, omega):
        return omega

    def get_s_pi(self, u_r, tn1, s_pi, alpha, z, omega):
        return s_pi

    def get_alpha(self, u_r, tn1, s_pi, alpha, z, omega):
        return alpha

    def get_z(self, u_r, tn1, s_pi, alpha, z, omega):
        return z

    def get_s_el(self, u_r, tn1, **state):
        s = self.get_slip(u_r, tn1, **state)
        s_p = self.get_s_pi(u_r, tn1, **state)
        s_e = s - s_p
        return s_e

#     def get_sig_N(self, u_r, tn1, **state):
#         return self.get_sig(u_r, tn1, **state)[..., 1]

    tree_view = bu.View(
        bu.Item('E_N'),
        bu.Item('E_T'),
        bu.Item('gamma'),
        bu.Item('K'),
        bu.Item('S'),
        bu.Item('r'),
        bu.Item('c'),
        bu.Item('m'),
        bu.Item('tau_bar'),
        bu.Item('D_rs', style='readonly')
    )

    traits_view = tree_view

