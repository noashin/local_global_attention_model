import sys

import numpy as np
from scipy.stats import multivariate_normal

sys.path.append('./../../')
from src.HMC.hmcparameter import HMCParameter


class VelParam(HMCParameter):
    """
    This class implements a velocity parameter for an HMC parameter, with a Gaussian distribution.
    """
    def __init__(self, init_val, sigma=None):
        """
        :param init_val: initial value for the velocity
        :param sigma: the covariance of the velocity (aka the mass matrix).
                    Only diagonal matrix is supported - sigma is expected to be a vector with the same dim as init_val.
        """
        super().__init__(np.array(init_val))
        dim = np.array(init_val).shape
        self.mu = np.zeros(dim)
        if sigma is None:
            self.sigma = np.array(init_val)
        else:
            self.sigma = np.array(sigma)

    def gen_init_value(self):
        self.value = multivariate_normal.rvs(self.mu, self.sigma)

    def get_energy_grad(self):
        return self.value / self.sigma

    def energy(self, value):
        return np.dot(value / np.sqrt(self.sigma), value / np.sqrt(self.sigma)) / 2

    def get_energy(self):
        return self.energy(self.value)

    def get_energy_for_value(self, value):
        return self.energy(value)
