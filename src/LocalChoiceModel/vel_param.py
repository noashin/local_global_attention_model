import sys

import numpy as np
from scipy.stats import multivariate_normal

sys.path.append('./../../')
from src.HMC.hmcparameter import HMCParameter

class VelParam(HMCParameter):
    def __init__(self, init_val):
        super().__init__(np.array(init_val))
        dim = np.array(init_val).shape
        self.mu = np.zeros(dim)
        self.sigma = 1

    def gen_init_value(self):
        self.value = multivariate_normal.rvs(self.mu, self.sigma)

    def get_energy_grad(self):
        return self.value

    def get_energy(self):
        return np.dot(self.value, self.value) / 2

    def get_energy_for_value(self, value):
        return np.dot(value, value) / 2
