'''
'''
from math import \
    pi

from ibvpy.api import TLoop, TLine
from ibvpy.tmodel.mats2D.mats2D_cmdm.mats2D_cmdm import MATS2DMicroplaneDamage
from ibvpy.tmodel.mats2D.mats2D_explorer_bcond import get_value_and_coeff, BCDofProportional
from ibvpy.mathkit.mfn import MFnLineArray

import numpy as np
import pylab as p

if __name__ == '__main__':

    elastic_debug = True
    # Tseval for a material model
    #
    tseval = MATS2DMicroplaneDamage(elastic_debug=elastic_debug)

    value, coeff = get_value_and_coeff(1., 0.0)

#    bcond_alpha = BCDof(var='u', dof=0, value=value,
    bcond_alpha = BCDofProportional(var='u', dof=0, value=value,
                     link_dofs=[1],
                     link_coeffs=[coeff],
                     time_function=lambda t: t)

    ts = TStepper(tse=tseval,
             bcond_list=[ bcond_alpha
                         ],
             rtrace_list=[ RTDofGraph(name='strain 0 - stress 0',
                                  var_x='eps_app', idx_x=0,
                                  var_y='sig_app', idx_y=0,
                                  record_on='update'),
                         RTDofGraph(name='strain 1 - stress 1',
                                  var_x='eps_app', idx_x=1,
                                  var_y='sig_app', idx_y=1,
                                  record_on='update'),
                         RTDofGraph(name='strain 0 - stress 1',
                                  var_x='eps_app', idx_x=0,
                                  var_y='sig_app', idx_y=1,
                                  record_on='update'),
                         RTDofGraph(name='strain 1 - stress 0',
                                  var_x='eps_app', idx_x=1,
                                  var_y='sig_app', idx_y=0,
                                  record_on='update'),
                         RTDofGraph(name='strain 0 - strain 1',
                                  var_x='eps_app', idx_x=0,
                                  var_y='eps_app', idx_y=1,
                                  record_on='update'),
                         ]
                         )

    # Put the time-stepper into the time-loop
    #
    if elastic_debug:
        tmax = 1.
        n_steps = 1
    else:
        tmax = 0.001
        # tmax = 0.0006
        n_steps = 100

    tl = TLoop(tstepper=ts,
             DT=tmax / n_steps, KMAX=100, RESETMAX=0,
             min=0.0, max=tmax)
#             T=TRange(min=0.0, max=tmax))

    from numpy import argmax

    alpha_arr = np.linspace(-pi / 2 * 1.05, 2 * (pi / 2.) + pi / 2.*0.05, 20)

    sig0_m_list = []
    sig1_m_list = []
    eps0_m_list = []
    eps1_m_list = []

    for alpha in alpha_arr:

        value, coeff = get_value_and_coeff(1., alpha)
        bcond_alpha.value = value
        bcond_alpha.link_coeffs[0] = coeff

        tl.eval()

        eps0_sig0 = tl.rtrace_mngr.rtrace_bound_list[0]
        eps1_sig1 = tl.rtrace_mngr.rtrace_bound_list[1]
#        eps0_sig0 = tl.rv_mngr.rv_list[0]
#        eps1_sig1 = tl.rv_mngr.rv_list[1]

        sig0_midx = argmax(np.fabs(eps0_sig0.trace.ydata))
        sig1_midx = argmax(np.fabs(eps1_sig1.trace.ydata))

        sig0_m = eps0_sig0.trace.ydata[ sig0_midx ]
        sig1_m = eps1_sig1.trace.ydata[ sig1_midx ]

        eps0_m = eps0_sig0.trace.xdata[ sig0_midx ]
        eps1_m = eps1_sig1.trace.xdata[ sig1_midx ]

        sig0_m_list.append(sig0_m)
        sig1_m_list.append(sig1_m)
        eps0_m_list.append(eps0_m)
        eps1_m_list.append(eps1_m)

    sig_plot = MFnLineArray(xdata=sig0_m_list,
                              ydata=sig1_m_list)
    eps_plot = MFnLineArray(xdata=eps0_m_list,
                              ydata=eps1_m_list)

#    sig_plot.configure_traits()

    print('sig_plot.xdata', sig_plot.ydata)
    p.plot(sig_plot.xdata, sig_plot.ydata)
    p.show()

    # Put the time-loop into the simulation-framework and map the
    # object to the user interface.
    #
