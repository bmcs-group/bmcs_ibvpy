
import traits.api as tr
import numpy as np

class MATS2DCalibrationMixinGf(tr.HasTraits):
    """Calibrate the values of the material parameters to render the desired stiffness
    strength and fracture energy"""

    def get_sig_eps(self):
        eps_max = self.eps_max
        n_eps = 5000
        eps11_range = np.linspace(1e-9, eps_max, n_eps)
        eps_range = np.zeros((len(eps11_range), 2, 2))
        eps_range[:, 1, 1] = eps11_range
        state_vars = {var: np.zeros((len(eps11_range),) + shape)
                      for var, shape in self.state_var_shapes.items()
                      }
        sig_range, _ = self.get_corr_pred(eps_range, 1, **state_vars)
        sig11_range = sig_range[:, 1, 1]
        eps11_range = eps_range[:, 1, 1]
        argmax_i = np.argmax(sig11_range)
        max_sig, argmax_eps = sig11_range[argmax_i], eps11_range[argmax_i]
        G_f = np.trapz(sig11_range, eps11_range)
        return G_f, max_sig, argmax_eps
