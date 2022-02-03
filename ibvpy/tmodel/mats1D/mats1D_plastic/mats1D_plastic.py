#-------------------------------------------------------------------------
#
# Copyright (c) 2009, IMB, RWTH Aachen.
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in simvisage/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.simvisage.com/licenses/BSD.txt
#
# Thanks for using Simvisage open source!
#
# Created on Sep 4, 2009 by: rch

from math import copysign, sin

from traits.api import \
    Trait,    \
    Dict

from ibvpy.tmodel.mats1D.mats1D_eval import MATS1DEval
import numpy as np
import bmcs_utils.api as bu

# from dacwt import DAC
def sign(val):
    return copysign(1, val)


#---------------------------------------------------------------------------
# Material time-step-evaluator for Scalar-Damage-Model
#---------------------------------------------------------------------------


class MATS1DPlastic(MATS1DEval):

    '''
    Scalar Damage Model.
    '''

    E = bu.Float(1.,  # 34e+3,
              label="E",
              desc="Young's Modulus",
              enter_set=True,
              auto_set=False)

    sigma_y = bu.Float(1.,
                    label="sigma_y",
                    desc="Yield stress",
                    enter_set=True,
                    auto_set=False)

    K_bar = bu.Float(0.1,  # 191e-6,
                  label="K",
                  desc="Plasticity modulus",
                  enter_set=True,
                  auto_set=False)

    H_bar = bu.Float(0.1,  # 191e-6,
                  label="H",
                  desc="Hardening modulus",
                  enter_set=True,
                  auto_set=False)

    #--------------------------------------------------------------------------
    # View specification
    #--------------------------------------------------------------------------

    traits_view = bu.View(
        bu.Item('E'),
        bu.Item('sigma_y'),
        bu.Item('K_bar'),
        bu.Item('H_bar'),
        bu.Item('stiffness'),
    )

    #-------------------------------------------------------------------------
    # Setup for computation within a supplied spatial context
    #-------------------------------------------------------------------------

    def get_state_array_size(self):
        '''
        Give back the nuber of floats to be saved
        @param sctx:spatial context

        eps_p_n - platic strain 
        alpha_n - hardening
        q_n - back stress  

        '''
        return 3

    def new_cntl_var(self):
        return np.zeros(1, np.float_)

    def new_resp_var(self):
        return np.zeros(1, np.float_)

    #-------------------------------------------------------------------------
    # Evaluation - get the corrector and predictor
    #-------------------------------------------------------------------------
    def get_corr_pred(self, sctx, eps_app_eng, d_eps, tn, tn1, eps_avg=None):
        '''
        Corrector predictor computation.
        @param eps_app_eng input variable - engineering strain
        '''
        eps_n1 = float(eps_app_eng)
        E = self.E

        if eps_avg == None:
            eps_avg = eps_n1

        if sctx.update_state_on:
            eps_n = eps_avg - float(d_eps)
            sctx.mats_state_array[:] = self._get_state_variables(sctx, eps_n)

        eps_p_n, q_n, alpha_n = sctx.mats_state_array
        sigma_trial = self.E * (eps_n1 - eps_p_n)
        xi_trial = sigma_trial - q_n
        f_trial = abs(xi_trial) - (self.sigma_y + self.K_bar * alpha_n)

        sig_n1 = np.zeros((1,), dtype='float_')
        D_n1 = np.zeros((1, 1), dtype='float_')
        if f_trial <= 1e-8:
            sig_n1[0] = sigma_trial
            D_n1[0, 0] = E
        else:
            d_gamma = f_trial / (self.E + self.K_bar + self.H_bar)
            sig_n1[0] = sigma_trial - d_gamma * self.E * sign(xi_trial)
            D_n1[0, 0] = (self.E * (self.K_bar + self.H_bar)) / \
                (self.E + self.K_bar + self.H_bar)

        return sig_n1, D_n1

    #--------------------------------------------------------------------------
    # Subsidiary methods realizing configurable features
    #--------------------------------------------------------------------------

    def _get_state_variables(self, sctx, eps_n):

        eps_p_n, q_n, alpha_n = sctx.mats_state_array

        # Get the characteristics of the trial step
        #
        sig_trial = self.E * (eps_n - eps_p_n)
        xi_trial = sig_trial - q_n
        f_trial = abs(xi_trial) - (self.sigma_y + self.K_bar * alpha_n)

        if f_trial > 1e-8:

            #
            # Tha last equilibrated step was inelastic. Here the
            # corresponding state variables must be calculated once
            # again. This might be expensive for 2D and 3D models. Then,
            # some kind of caching should be considered for the state
            # variables determined during iteration. In particular, the
            # computation of d_gamma should be outsourced into a separate
            # method that can in general perform an iterative computation.
            #
            d_gamma = f_trial / (self.E + self.K_bar + self.H_bar)
            eps_p_n += d_gamma * sign(xi_trial)
            q_n += d_gamma * self.H_bar * sign(xi_trial)
            alpha_n += d_gamma

        newarr = np.array([eps_p_n, q_n, alpha_n], dtype='float_')

        return newarr

    #-----------------------------------------------------------
    # Response trace evaluators
    #--------------------------------------------------------------------------
    def get_eps_p(self, sctx, eps_app_eng):
        return np.array([sctx.mats_state_array[0]])

    def get_q(self, sctx, eps_app_eng):
        return np.array([sctx.mats_state_array[1]])

    def get_alpha(self, sctx, eps_app_eng):
        return np.array([sctx.mats_state_array[2]])

    # Declare and fill-in the rte_dict - it is used by the clients to
    # assemble all the available time-steppers.
    #
    rte_dict = Trait(Dict)

    def _rte_dict_default(self):
        return {'sig_app': self.get_sig_app,
                'eps_app': self.get_eps_app,
                'eps_p': self.get_eps_p,
                'q': self.get_q,
                'alpha': self.get_alpha}
