'''
Created on 29.03.2017

@author: abaktheer

Microplane damage model 2D - Jirasek [1999]
'''

from numpy import \
    array, einsum, identity, sqrt
from traits.api import \
    Constant, provides, \
    Property, cached_property

from bmcs_utils.api import Float, View, Item

from ibvpy.tmodel.mats2D.mats2D_eval import MATS2DEval
from ibvpy.tmodel.matsXD.vmatsXD_eval import MATSXDEval
import numpy as np
from ibvpy.sim.i_tmodel import ITModel
import traits.api as tr
from ibvpy.tmodel.mats2D.mats2D_microplane.vmats2D_calibration_mixin_Gf import \
    MATS2DCalibrationMixinGf

@provides(ITModel)
class MATS2DMplDamageEEQ(MATS2DEval, MATS2DCalibrationMixinGf):

    epsilon_0 = Float(59e-6,
                      label="a",
                      desc="Lateral pressure coefficient",
                      MAT=True)

    epsilon_f = Float(250e-6,
                      label="a",
                      desc="Lateral pressure coefficient",
                      MAT=True)

    c_T = Float(0.00,
                label="a",
                desc="Lateral pressure coefficient",
                MAT=True)

    ipw_view = View(
        *MATS2DEval.ipw_view.content,
        Item('epsilon_0'),
        Item('epsilon_f'),
        Item('c_T'),
    )

    state_var_shapes = tr.Property(tr.Dict(), depends_on='n_mp')
    '''Dictionary of state variable entries with their array shapes.
    '''

    @cached_property
    def _get_state_var_shapes(self):
        return dict(kappa=(self.n_mp,),
                    omega=(self.n_mp,))

    #-------------------------------------------------------------------------
    # MICROPLANE-Kinematic constraints
    #-------------------------------------------------------------------------

    # get the dyadic product of the microplane normals
    _MPNN = Property(depends_on='n_mp')

    @cached_property
    def _get__MPNN(self):
        # dyadic product of the microplane normals

        MPNN_nij = einsum('ni,nj->nij', self._MPN, self._MPN)
        return MPNN_nij

    # get the third order tangential tensor (operator) for each microplane
    _MPTT = Property(depends_on='n_mp')

    @cached_property
    def _get__MPTT(self):
        # Third order tangential tensor for each microplane
        delta = identity(2)
        MPTT_nijr = 0.5 * (einsum('ni,jr -> nijr', self._MPN, delta) +
                           einsum('nj,ir -> njir', self._MPN, delta) - 2.0 *
                           einsum('ni,nj,nr -> nijr', self._MPN, self._MPN, self._MPN))
        return MPTT_nijr

    def _get_e_na(self, eps_ab):
        r'''
        Projection of apparent strain onto the individual microplanes
        '''
        e_ni = einsum(
            'nb,...ba->...na',
            self._MPN, eps_ab
        )
        return e_ni

    def _get_e_N_n(self, e_na):
        r'''
        Get the normal strain array for each microplane
        '''
        e_N_n = einsum(
            '...na, na->...n',
            e_na, self._MPN
        )
        return e_N_n

    def _get_e_equiv_n(self, e_na):
        r'''
        Returns a list of the microplane equivalent strains
        based on the list of microplane strain vectors
        '''
        # magnitude of the normal strain vector for each microplane
        e_N_n = self._get_e_N_n(e_na)
        # positive part of the normal strain magnitude for each microplane
        e_N_pos_n = (np.abs(e_N_n) + e_N_n) / 2.0
        # normal strain vector for each microplane
        e_N_na = einsum('...n,ni -> ...ni', e_N_n, self._MPN)
        # tangent strain ratio
        c_T = self.c_T
        # tangential strain vector for each microplane
        e_T_na = e_na - e_N_na
        # squared tangential strain vector for each microplane
        e_TT_n = einsum('...ni,...ni -> ...n', e_T_na, e_T_na)
        # equivalent strain for each microplane
        e_equiv_n = sqrt(e_N_pos_n * e_N_pos_n + c_T * e_TT_n)
        return e_equiv_n

    def update_state_variables(self, eps_ab, kappa_n, omega_n):
        e_na = self._get_e_na(eps_ab)
        eps_eq_n = self._get_e_equiv_n(e_na)
        f_trial_n = eps_eq_n - self.epsilon_0
        I = np.where(f_trial_n > 0)
        k_n = np.max(np.array([kappa_n[I], eps_eq_n[I]]), axis=0)
        kappa_n[I] = k_n
        omega_n[I] = self._get_omega(k_n)

    def _get_omega(self, kappa_Emn):
        '''
        Return new value of damage parameter
        @par bbam kappa:
        '''
        omega_Emn = np.zeros_like(kappa_Emn)
        epsilon_0 = self.epsilon_0
        epsilon_f = self.epsilon_f
        kappa_idx = np.where(kappa_Emn > epsilon_0)
        omega_Emn[kappa_idx] = (
            1.0 - (epsilon_0 / kappa_Emn[kappa_idx] *
                   np.exp(-1.0 * (kappa_Emn[kappa_idx] - epsilon_0) /
                          (epsilon_f - epsilon_0))
                   ))
        return omega_Emn

    def _get_phi_Emab(self, kappa_Emn):
        # Returns the 2nd order damage tensor 'phi_mtx'
        # scalar integrity factor for each microplane
        phi_Emn = np.sqrt(1.0 - self._get_omega(kappa_Emn))
        # integration terms for each microplanes
        phi_Emab = einsum('...n,n,nab->...ab', phi_Emn, self._MPW, self._MPNN)
        return phi_Emab

    def _get_beta_Emabcd(self, phi_Emab):
        '''
        Returns the 4th order damage tensor 'beta4' using sum-type symmetrization
        (cf. [Jir99], Eq.(21))
        '''
        delta = identity(2)
        beta_Emijkl = 0.25 * (einsum('...ik,jl->...ijkl', phi_Emab, delta) +
                              einsum('...il,jk->...ijkl', phi_Emab, delta) +
                              einsum('...jk,il->...ijkl', phi_Emab, delta) +
                              einsum('...jl,ik->...ijkl', phi_Emab, delta))

        return beta_Emijkl

    #-------------------------------------------------------------------------
    # Evaluation - get the corrector and predictor
    #-------------------------------------------------------------------------

    def get_corr_pred(self, eps_ab, tn1, kappa, omega):

        self.update_state_variables(eps_ab, kappa, omega)

        #----------------------------------------------------------------------
        # if the regularization using the crack-band concept is on calculate the
        # effective element length in the direction of principle strains
        #----------------------------------------------------------------------
        # if self.regularization:
        #    h = self.get_regularizing_length(sctx, eps_app_eng)
        #    self.phi_fn.h = h

        #------------------------------------------------------------------
        # Damage tensor (2th order):
        #------------------------------------------------------------------
        phi_ab = self._get_phi_Emab(kappa)
        #------------------------------------------------------------------
        # Damage tensor (4th order) using product- or sum-type symmetrization:
        #------------------------------------------------------------------
        beta_abcd = self._get_beta_Emabcd(phi_ab)
        #------------------------------------------------------------------
        # Damaged stiffness tensor calculated based on the damage tensor beta4:
        #------------------------------------------------------------------
        D_ijab = einsum(
            '...ijab, abef, ...cdef -> ...ijcd',
            beta_abcd, self.D_abef, beta_abcd
        )

        sig_ab = einsum('...abef,...ef -> ...ab', D_ijab, eps_ab)

        return sig_ab, D_ijab


    '''Number of microplanes - currently fixed for 3D
    '''
    n_mp = Constant(22)

    _alpha_list = Property(depends_on='n_mp')

    @cached_property
    def _get__alpha_list(self):
        return array([np.pi / self.n_mp * (i - 0.5)
                      for i in range(1, self.n_mp + 1)])

    #-----------------------------------------------
    # get the normal vectors of the microplanes
    #-----------------------------------------------

    _MPN = Property(depends_on='n_mp')

    @cached_property
    def _get__MPN(self):
        # microplane normals:

        alpha_list = np.linspace(0, 2 * np.pi, self.n_mp)

        MPN = np.array([[np.cos(alpha), np.sin(alpha)]
                        for alpha in alpha_list])

        return MPN

    #-------------------------------------
    # get the weights of the microplanes
    #-------------------------------------
    _MPW = Property(depends_on='n_mp')

    @cached_property
    def _get__MPW(self):
        # Note that the values in the array must be multiplied by 6 (cf. [Baz05])!
        # The sum of of the array equals 0.5. (cf. [BazLuz04]))
        # The values are given for an Gaussian integration over the unit
        # hemisphere.
        MPW = np.ones(self.n_mp) / self.n_mp * 2

        return MPW
