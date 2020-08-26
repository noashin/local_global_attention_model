import sys
import pickle

import numpy as np
from scipy.stats import bernoulli

sys.path.append('./../')
sys.path.append('./../../')
from src.FullModel.model import Model as parent_model
from src.LocalGlobalAttentionModel.model import Model as super_model
from .vel_param import VelParam as vel_param
from src.HMC.hmc import HMC

delta = 10 ** -200

class Model(parent_model):
    """
    This class implements the Fixed Choice model as described in the paper.
    It has the same local and global policis like the full model and the difference is in the calculation of rho.
    Here rho has a fixed value.
    """

    def __init__(self, saliencies, rho, epsilon, xi, cov_ratio=1):
        # epsilon and xi should be the objects from parent_model, with fix_dist_ind = 0
        # rho should be from this model

        super_model.__init__(self, saliencies)

        self.rho = rho
        self.epsilon = epsilon
        self.xi = xi
        self.cov_ratio = cov_ratio
        self.fix_dist_ind = 0

    def calc_ros(self, *args):
        return self.rho.value

    # Methods for generating data
    def generate_gamma(self, s_t):
        """
        This method generates gamma according to a Bernouli distribution with p = rho
        :param s_t: here just to be compatible with the parent class.
        :return: gamma \sim Ber(rho)
        """
        return bernoulli.rvs(self.rho.value)

    # Methods for parameters inference via Gibbs sampling
    def sample_gamma(self):
        """
        This methods samples form the conditional posterior distribution of gamma.
        For details see the paper.
        :return: a sample \gamma_i for each data point
        """
        BF = self.calc_BF()

        gammas = []

        for i, sal_ts in enumerate(self.saliencies_ts):
            gammas.append([])

            for s, subject in enumerate(sal_ts):
                ros = self.rho.value / (self.rho.value + BF[i][s] * (1 - self.rho.value))
                gammas[-1].append(bernoulli.rvs(ros))
        return gammas

    def sample(self, num_samples, save_steps, file_path, sample_gammas=True):
        """
        This method perform Gibbs sampling for the model parameters.
        :param num_samples: number of samples in the chain
        :param save_steps: whether to save the chains.
        :param file_path: path to a file to save the chains
        :param sample_gammas: whether to sample gamma or not,
        :return: array with samples for each of the model parameters - b, s_0, epsilon, xi
        """

        # initialize the arrays that will hold the samples.
        samples_rho = np.zeros(num_samples)
        samples_epsilon = np.zeros((num_samples, 2))
        samples_xi = np.zeros((num_samples, 2))

        # set variables needed for the HMC inference of epsilon and xi
        vel_eps = vel_param([1, 1])
        vel_xi = vel_param([1, 1])
        delta_xi = 0.5
        delta_eps = 0.03
        n = 10
        m = 1
        hmc_eps = HMC(self.epsilon, vel_eps, delta_eps, n, m)
        hmc_xi = HMC(self.xi, vel_xi, delta_xi, n, m)

        if not sample_gammas:
            self.remove_first_gamma()

        for i in range(num_samples):

            if sample_gammas:
                self.gammas = self.sample_gamma()

            if i == 0:
                if not self.rho.is_fixed:
                    self.rho.set_num_time_steps(self.gammas)

            rho_samp = self.rho.conditional_posterior(self.gammas)
            self.rho.set_value(rho_samp)

            if not self.epsilon.is_fixed:
                hmc_eps.HMC(self.xi.value, self.cov_ratio, self.saliencies, self.gammas, self.fix_dists_2,
                            self.dist_mat_per_fix,
                            self.xi.alpha, self.xi.betta)
            epsilon_samp = hmc_eps.state_param.value

            if not self.xi.is_fixed:
                hmc_xi.HMC(self.epsilon.value, self.cov_ratio, self.saliencies, self.gammas, self.fix_dists_2,
                           self.dist_mat_per_fix,
                           self.epsilon.alpha, self.epsilon.betta)
            xi_samp = hmc_xi.state_param.value

            samples_rho[i] = rho_samp
            samples_epsilon[i] = epsilon_samp
            samples_xi[i] = xi_samp

            if save_steps and not i % 50:
                with open(file_path, 'wb') as f:
                    pickle.dump([samples_rho[:i], samples_epsilon[:i], samples_xi[:i]], f)

        if save_steps:
            with open(file_path, 'wb') as f:
                pickle.dump([samples_rho, samples_epsilon, samples_xi], f)

        return samples_rho, samples_epsilon, samples_xi
