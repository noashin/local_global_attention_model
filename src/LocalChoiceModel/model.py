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
import pypolyagamma as pypolyagamma

delta = 10 ** -200


class Model(parent_model):
    """
    This class implements the local choice model as described in the paper.
    It has the same local and global attention policies as the fullmodel, but the calculation of rho is different.
    """

    def __init__(self, saliencies, s_0, b, epsilon, xi, cov_ratio=1):
        # epsilon and xi should be the objects from parent_model
        # b and s_0 should be the objects from super_model

        super_model.__init__(self, saliencies)

        self.b = b
        self.s_0 = s_0
        self.epsilon = epsilon
        self.xi = xi
        self.cov_ratio = cov_ratio

    # data generation methods
    def generate_gamma(self, s_t):
        """
        This methods samples gammas from a Bernouli distribution with rho = sigmoid(b * (s_t - s_0)
        :param s_t: time series of saliency values of consecutive fixations
        :return: samples from Bern(rho). Same shape as s_t
        """
        prob = self.sigmoid(s_t)
        return bernoulli.rvs(prob)

    def get_next_fix(self, im_ind, _, __, cur_fix, s_t):
        """
        This method samples the next fixation location given the current fixation location and the previous fixation location.
        It implements p(z_t|z_{t-1}) of the local choice model. For details see the paper.
        :param im_ind: index of the current image
        :param cur_fix: location (in image coordinates) of the current fixation
        :param s_t: saliency value of the current fixation
        :return:
        """
        gamma_t = self.generate_gamma(s_t)
        next_fix = self.get_step_from_gamma(gamma_t, cur_fix, im_ind)

        return next_fix, gamma_t

    # Methods for parameter inference

    def sample_w(self):
        """
        This method samples the augmenting w parameters from its conditional posterior distribution.
        For details about the augmentation see the paper.
        :return: samples for w_i from a polyagamma distribution.
                list of lists of arrays num_images x num_subjects x T(image, subject).
        """
        nthreads = pypolyagamma.get_omp_num_threads()
        seeds = np.random.randint(2 ** 16, size=nthreads)
        ppgs = [pypolyagamma.PyPolyaGamma(seed) for seed in seeds]

        w = []

        for i in range(len(self.saliencies_ts)):
            w.append([])

            for saliency_ts in self.saliencies_ts[i]:
                T = saliency_ts.shape[0]
                A = np.ones(T)
                w_is = np.zeros(T)
                pypolyagamma.pgdrawvpar(ppgs, A, np.abs(self.b.value * (saliency_ts - self.s_0.value)), w_is)
                w[-1].append(w_is)
        return w

    def sample_gamma(self):
        """
        Sample the augmenting variable gamma from its conditional posterior distribution.
        See the paper for details.
        :return: list of lists of arrays num_images x num_subjects x (T(image, subject) - 2)
                of gammas \in {0,1}
        """
        BF = self.calc_BF()

        gammas = []

        for i, saliency_ts in enumerate(self.saliencies_ts):
            gammas.append([])

            for s, subject in enumerate(saliency_ts):
                sig = self.sigmoid(subject[:-1])
                ros = sig / (sig + BF[i][s] * (1 - sig))
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
        samples_s0 = np.zeros(num_samples)
        samples_b = np.zeros(num_samples)
        samples_epsilon = np.zeros((num_samples, 2))
        samples_xi = np.zeros((num_samples, 2))

        # set variables needed for the HMC inference of epsilon and xi
        vel_eps = vel_param([1, 1])
        vel_xi = vel_param([1, 1])
        delta_xi = 0.5
        delta_eps = 0.03
        n = 8
        m = 1
        hmc_eps = HMC(self.epsilon, vel_eps, delta_eps, n, m)
        hmc_xi = HMC(self.xi, vel_xi, delta_xi, n, m)

        if not sample_gammas:
            self.remove_first_gamma()

        i = 0
        while i < num_samples:

            w = self.sample_w()

            if sample_gammas:
                self.gammas = self.sample_gamma()

            s0_samp = self.s_0.conditional_posterior(self.gammas, self.b.value, w, self.saliencies_ts)
            self.s_0.set_value(s0_samp)

            b_samp = self.b.conditional_posterior(self.gammas, self.s_0.value, w, self.saliencies_ts)
            # if b gets weird value - start the sampling from the beginning
            if b_samp is None or b_samp == 0:
                print('b had an error - restarting')
                self.b.set_value(self.b.prior())
                self.s_0.set_value(self.s_0.prior())
                self.epsilon.set_value(self.epsilon.prior())
                self.xi.set_value(self.xi.prior())

                samples_s0 = np.zeros(num_samples)
                samples_b = np.zeros(num_samples)
                samples_epsilon = np.zeros((num_samples, 2))
                samples_xi = np.zeros((num_samples, 2))
                self.b.set_value(self.b.prior())
                self.s_0.set_value(self.s_0.prior())
                self.epsilon.set_value(self.epsilon.init_value)
                self.xi.set_value(self.xi.init_value)
                i = 0

                continue

            self.b.set_value(b_samp)

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

            samples_s0[i] = s0_samp
            samples_b[i] = b_samp
            samples_epsilon[i] = epsilon_samp
            samples_xi[i] = xi_samp

            if save_steps and not i % 50:
                with open(file_path, 'wb') as f:
                    pickle.dump([samples_s0[:i], samples_b[:i], samples_epsilon[:i], samples_xi[:i]], f)

            i += 1

        if save_steps:
            with open(file_path, 'wb') as f:
                pickle.dump([samples_s0, samples_b, samples_epsilon, samples_xi], f)

        return samples_s0, samples_b, samples_epsilon, samples_xi

    # Methods for calculating the likelihood for a given data-set
    def calc_ros(self, im_ind, sal_ts, for_nss=False, saliencies=None):
        """
        This metods calculates rho according to the local choice from the paper
        for a specific scanpath.
        :param im_ind: index of the image
        :param sal_ts: time series of saliencies values
        :param for_nss: are the rhos calculated for NSS
        :param saliencies: optional list of saliency matrices
        :return: time series of the corresponding rho values
        """
        if for_nss:
            ros = self.sigmoid(saliencies[im_ind])[:, :, np.newaxis]
        else:
            ros = self.sigmoid(sal_ts[im_ind][0][:-1])

        return ros
