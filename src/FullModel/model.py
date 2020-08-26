import sys
import pickle

import numpy as np
from scipy.stats import bernoulli, multivariate_normal

sys.path.append('./../')
sys.path.append('./../../')
from src.LocalGlobalAttentionModel.model import Model as parent_model
from .vel_param import VelParam as vel_param
from src.HMC.hmc import HMC
import pypolyagamma as pypolyagamma

delta = 10 ** -200


def calc_sals_ratio_ts(sals_ts):
    """
    This function calculates the ratio between the saliency values of consecutive fixations.
    Useful for calculating rho of the full model (see paper for details).
    :param sals_ts: list of lists of time serieses of saliency values
    :return: list of lists of arrays num_images x num_subjects x (T(image, subject) - 1)
            of ratios between the saliencies.
    """
    sals_ratio_ts = []
    for sal_ts in sals_ts:
        sals_ratio_ts.append([])
        for subject_ts in sal_ts:
            sals_ratio_ts[-1].append(subject_ts[1:] / subject_ts[:-1])

    return sals_ratio_ts


class Model(parent_model):
    """
    This model is the full model presented in the paper.
    """

    def __init__(self, saliencies, s_0, b, epsilon, xi, cov_ratio=1):
        # epsilon and xi should be the objects from the current folder
        # b and s_0 should be the objects from super_model

        parent_model.__init__(self, saliencies)

        self.b = b
        self.s_0 = s_0
        self.epsilon = epsilon
        self.xi = xi
        self.cov_ratio = cov_ratio
        self.fix_dist_ind = 1

        self.ros_ts = None

    # Methods for setting stuff

    def set_sal_ts_ratio(self):
        """
        This methods sets the saliency ratio data from the saliencie time series.
        Will only work after calling set_saliencies_ts
        """
        self.sals_ratio_ts = calc_sals_ratio_ts(self.saliencies_ts)

    # Methods for data generation
    def local_policy_likelihood(self, cur_fix):
        """
        This methods calculates a probability distribution over the image following the
        local attention policy.
        :param cur_fix: coordinates of the current fixation [x, y]
        :return: matrix of distribution over the image num_pixels x num_pixels
        """

        eps_val = self.epsilon.value
        mean = cur_fix
        rad_rows = (self.rows_grid - mean[0]) ** 2
        rad_cols = (self.cols_grid - mean[1]) ** 2

        # a Gaussian over the image with covariance epsilon
        prob = np.exp(- rad_rows / (2 * eps_val[0]) - rad_cols / (2 * eps_val[1])) / \
               (2 * np.pi * np.sqrt(eps_val[0] * eps_val[1]))

        return prob

    def global_policy_likelihood(self, cur_fix, im_ind):
        """
        This methods calculates a probability distribution over the image following the
        global attention policy.
        :param cur_fix: coordinates of the current fixation [x, y]
        :param im_ind: index of the current image
        :return: matrix of distribution over the image num_pixels x num_pixels
        """

        eps_val = self.epsilon.value
        mean = cur_fix
        rad_rows = (self.rows_grid - mean[0]) ** 2
        rad_cols = (self.cols_grid - mean[1]) ** 2

        xi_val = self.xi.value
        eps_val = eps_val / self.cov_ratio

        gauss_xi = np.exp(- rad_rows / (2 * xi_val[0]) - rad_cols / (2 * xi_val[1])) / \
                   (2 * np.pi * np.sqrt(xi_val[0] * xi_val[1]))

        gauss_eps = np.exp(- rad_rows / (2 * eps_val[0]) - rad_cols / (2 * eps_val[1])) / \
                    (2 * np.pi * np.sqrt(eps_val[0] * eps_val[1]))

        prob = (gauss_xi - gauss_eps)
        prob = np.maximum(prob, delta) * self.saliencies[im_ind]

        return prob

    def generate_gamma(self, s_t_ratio):
        """
        This methods samples gammas from a Bernouli distribution with rho = sigmoid(b * (sal_ratio - s_0)
        :param s_t_ratio: time series of ratio of saliency values of consecutive fixations
        :return: samples from Bern(rho). Same shape as s_t_ratio
        """
        prob = self.sigmoid(s_t_ratio)
        return bernoulli.rvs(prob)

    def get_next_fix(self, im_ind, _, prev_fix, cur_fix, s_t):
        """
        This method samples the next fixation location given the current fixation location and the previous fixation location.
        It implements p(z_t|z_{t-1}, z_{t-2}) of the full model. For details see the paper.
        :param im_ind: index of the current image
        :param prev_fix: location (in image coordinates) of the previous fixation
        :param cur_fix: location (in image coordinates) of the current fixation
        :param s_t: saliency value of the current fixation
        :return:
        """

        # get the saliency value of the previous fixation
        # If the previous fixation was out side of the image return delta
        if prev_fix[0] > self.sal_shape[0] - 1 or \
                prev_fix[1] > self.sal_shape[1] - 1 or \
                prev_fix[0] < 0 or \
                prev_fix[1] < 0:

            s_prev = delta
        else:
            prev_fix_ind = np.rint(prev_fix).astype(int)
            s_prev = self.saliencies[im_ind][prev_fix_ind[0], prev_fix_ind[1]]

        # generate gamma, to decide which strategy to use
        s_ratio = s_t / s_prev
        gamma_t = self.generate_gamma(s_ratio)
        next_fix = self.get_step_from_gamma(gamma_t, cur_fix, im_ind)

        return next_fix, gamma_t

    def get_step_from_gamma(self, gamma_t, cur_fix, im_ind):
        """
        This method generate the next fixation location from either the local or global policy
        according to gamma
        :param gamma_t: \in {0,1} - which policy to chose
        :param cur_fix: fixation location of the current fixation
        :param im_ind: index of the current image
        :return: fixation location [x, y] of the next fixaiton
        """

        # get a probability map over the image following the chosen policy
        if gamma_t:
            prob = self.local_policy_likelihood(cur_fix)
        else:
            prob = self.global_policy_likelihood(cur_fix, im_ind)

        # normalize the distribution to sum to 1
        prob /= prob.sum()

        # sample the next fixation from the probability distribution
        inds = np.random.choice(range(self.pixels_num), 1,
                                p=prob.flatten())  # choice uses the inverse transform method in 1d
        next_fix = np.unravel_index(inds, self.saliencies[im_ind].shape)
        next_fix = np.array([next_fix[0][0], next_fix[1][0]])

        return next_fix

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

        for i in range(len(self.sals_ratio_ts)):
            w.append([])

            for sal_ratio in self.sals_ratio_ts[i]:
                T = sal_ratio.shape[0]
                A = np.ones(T)
                w_is = np.zeros(T)
                pypolyagamma.pgdrawvpar(ppgs, A, np.abs(self.b.value * (sal_ratio - self.s_0.value)), w_is)
                w[-1].append(w_is)
        return w

    def calc_BF(self):
        """
        This method calculates the Bayes factors needed for the sampling of gamma.
        For details of the calculation see paper.
        :return: list of lists of array of the Bayes Factor num_images x num_subjects x (T(image, subject) - 2)
        """

        BF = []
        mean = np.zeros(2)
        eps_global = self.epsilon.value / self.cov_ratio

        for i, saliency_ts in enumerate(self.saliencies_ts):
            BF.append([])

            for s, subject in enumerate(saliency_ts):

                # calculate the probability of the local attention policy
                local_nom = multivariate_normal.pdf(np.sqrt(self.fix_dists_2[i][s][:-1, self.fix_dist_ind:]).T,
                                                       mean=mean,
                                                       cov=self.epsilon.value)
                local_denom = (np.exp(
                    (- self.dist_mat_per_fix[i][s][:, :, self.fix_dist_ind:] / (2 * self.epsilon.value)).sum(
                        axis=3)) / (
                                          2 * np.pi * np.sqrt(self.epsilon.value[0] * self.epsilon.value[1]))).sum(
                    axis=(0, 1))

                local_policy = local_nom / local_denom

                # calculate the probability of the global attention policy
                g_global_small = multivariate_normal.pdf(np.sqrt(self.fix_dists_2[i][s][:-1, self.fix_dist_ind:]).T,
                                                          mean=mean, cov=eps_global)
                g_global_big = multivariate_normal.pdf(np.sqrt(self.fix_dists_2[i][s][:-1, self.fix_dist_ind:]).T,
                                                        mean=mean,
                                                        cov=self.xi.value)

                g_diff_nom = g_global_big - g_global_small
                g_max_nom = np.maximum(g_diff_nom, delta)
                global_nom = g_max_nom * subject[1 + self.fix_dist_ind:]

                g_diff_denom = np.exp(
                    (- self.dist_mat_per_fix[i][s][:, :, self.fix_dist_ind:] / (2 * self.xi.value)).sum(axis=3)) / (
                                       2 * np.pi * np.sqrt(self.xi.value[0] * self.xi.value[1])) - np.exp(
                    (- self.dist_mat_per_fix[i][s][:, :, self.fix_dist_ind:] / (2 * eps_global)).sum(axis=3)) / (
                                       2 * np.pi * np.sqrt(eps_global[0] * eps_global[1]))
                g_max_denom = np.maximum(g_diff_denom, delta)
                norm_factor = (self.saliencies[i][:, :, np.newaxis] * g_max_denom).sum(axis=(0, 1))

                global_policy = global_nom / norm_factor

                # the BF is the ratio between the probabilities of the two distributions
                BF[-1].append(global_policy / local_policy)
        return BF

    def sample_gamma(self):
        """
        Sample the augmenting variable gamma from its conditional posterior distribution.
        See the paper for details.
        :return: list of lists of arrays num_images x num_subjects x (T(image, subject) - 2)
                of gammas \in {0,1}
        """
        BF = self.calc_BF()

        gammas = []

        for i, sal_ratio in enumerate(self.sals_ratio_ts):
            gammas.append([])

            for s, subject in enumerate(sal_ratio):
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
        n = 10
        m = 1
        hmc_eps = HMC(self.epsilon, vel_eps, delta_eps, n, m)
        hmc_xi = HMC(self.xi, vel_xi, delta_xi, n, m)

        if not sample_gammas:
            self.remove_first_gamma()
            self.remove_first_gamma()

        i = 0
        while i < num_samples:
            w = self.sample_w()

            if sample_gammas:
                self.gammas = self.sample_gamma()

            s0_samp = self.s_0.conditional_posterior(self.gammas, self.b.value, w, self.sals_ratio_ts)
            self.s_0.set_value(s0_samp)

            b_samp = self.b.conditional_posterior(self.gammas, self.s_0.value, w, self.sals_ratio_ts)
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

            # add the new sampled values to the chains
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

    # Methods necessary for the calculation of the likelihood of specific dataset
    def calc_prob_local(self, im_ind, fixs_dists_2, fixs, for_nss=False):
        """
        This methods calculates the probability of the data given the local attention policy.
        :param im_ind: index of the current image
        :param fixs_dists_2: an array of shape 3 x (T -1). see set_fix_dist_2 for description.
        :param sal_ts: time series of the saliency value for each fixation. Array of length T.
        :param fixs: data set of fixations from one subject
        :param for_nss: if we calculate the NSS instead of the likelihood
        :return: the probability of the data according to the local attention policy
        """

        epsilon = self.epsilon.value
        # fixs[:-1] because we need z_t_1
        radx = (self.rows_grid[:, :, np.newaxis] - fixs[im_ind][0][0, self.fix_dist_ind:-1]) ** 2
        rady = (self.cols_grid[:, :, np.newaxis] - fixs[im_ind][0][1, self.fix_dist_ind:-1]) ** 2

        prob_for_all_pixels = (np.exp(- radx / (2 * epsilon[0]) - rady / (2 * epsilon[1])) / (
                2 * np.pi * np.sqrt(epsilon[0] * epsilon[1])))

        if for_nss:
            prob_local = prob_for_all_pixels / prob_for_all_pixels.sum(axis=(0, 1))
        else:
            X = fixs_dists_2[im_ind][0][:, self.fix_dist_ind:]
            prob_for_fixs = np.exp(- 0.5 * X[0] / epsilon[0] - 0.5 * X[1] / epsilon[1]) / \
                            (2 * np.pi * np.sqrt(epsilon[0] * epsilon[1]))

            prob_local = prob_for_fixs / prob_for_all_pixels.sum(axis=(0, 1))

        return prob_local

    def calc_prob_global(self, im_ind, fixs_dists_2, sal_ts, fixs, for_nss=False):
        """
        This methods calculates the probability of the data given the global attention policy.
        :param im_ind: index of the current image
        :param fixs_dists_2: an array of shape 3 x (T -1). see set_fix_dist_2 for description.
        :param sal_ts: time series of the saliency value for each fixation. Array of length T.
        :param fixs: data set of fixations from one subject
        :param for_nss: if we calculate the NSS instead of the likelihood
        :return: the probability of the data according to the local attention policy
        """

        X = fixs_dists_2[im_ind][0][:, self.fix_dist_ind:]
        xi = self.xi.value
        eps = self.epsilon.value
        eps_global = eps / self.cov_ratio

        # fixs[:-1] because we need z_t_1
        radx = (self.rows_grid[:, :, np.newaxis] - fixs[im_ind][0][0, self.fix_dist_ind:-1]) ** 2
        rady = (self.cols_grid[:, :, np.newaxis] - fixs[im_ind][0][1, self.fix_dist_ind:-1]) ** 2

        gauss_big = np.exp(- radx / (2 * xi[0]) - rady / (2 * xi[1])) / (2 * np.pi * np.sqrt(xi[0] * xi[1]))
        gauss_small = np.exp(- radx / (2 * eps_global[0]) - rady / (2 * eps_global[1])) / (
                2 * np.pi * np.sqrt(eps_global[0] * eps_global[1]))

        tmp_diff = gauss_big - gauss_small
        tmp_max = np.maximum(tmp_diff, delta)
        prob_for_all_pixels = tmp_max * self.saliencies[im_ind][:, :, np.newaxis]

        if for_nss:
            prob_global = prob_for_all_pixels / prob_for_all_pixels.sum(axis=(0, 1))
        else:
            nominator_gauss_big = np.exp(- 0.5 * X[0] / xi[0] - 0.5 * X[1] / xi[1]) / \
                                  (2 * np.pi * np.sqrt(xi[0] * xi[1]))
            nominator_gauss_small = np.exp(- 0.5 * X[0] / eps_global[0] - 0.5 * X[1] / eps_global[1]) / \
                                    (2 * np.pi * np.sqrt(eps_global[0] * eps_global[1]))
            tmp_diff = nominator_gauss_big - nominator_gauss_small
            tmp_max = np.maximum(tmp_diff, delta)
            prob_for_fixs = tmp_max * sal_ts[im_ind][0][1 + self.fix_dist_ind:]

            prob_global = prob_for_fixs / prob_for_all_pixels.sum(axis=(0, 1))
        return prob_global

    def calc_ros(self, im_ind, sal_ts, for_nss=False, saliencies=None):
        """
        This metods calculates rho according to the full model from the paper
        for a specific scanpath.
        :param im_ind: index of the image
        :param sal_ts: time series of saliencies values
        :param for_nss: are the rhos calculated for NSS
        :param saliencies: optional list of saliency matrices
        :return: time series of the corresponding rho values
        """
        s = 0
        if for_nss:
            sal_ratio_ts = saliencies[im_ind][:, :, np.newaxis] / np.array(sal_ts[im_ind][s][:-1])[np.newaxis,
                                                                  np.newaxis, :]
            ros = self.sigmoid(sal_ratio_ts[:, :, :-1])
        else:
            sal_ratio_ts = np.array(sal_ts[im_ind][s][1:]) / np.array(sal_ts[im_ind][s][:-1])
            ros = self.sigmoid(sal_ratio_ts[:-1])
        return ros

