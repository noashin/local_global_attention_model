import sys
import pickle

import numpy as np

sys.path.append('./../')
sys.path.append('./../../')
from src.LocalGlobalAttentionModel.model import Model as parent_model
from .vel_param import VelParam as vel_param
from src.HMC.hmc import HMC


class Model(parent_model):
    """
    This class describes a model where fixations are chosen from the static saliency
    convolved with a Gaussian.
    p(z_t|z_{t-1}) = s(t) * n(z_t|z_{t-1}, xi)
    """

    def __init__(self, saliencies, xi):
        super().__init__(saliencies)

        self.xi = xi
        self.gammas = None

    def get_next_fix(self, im_ind, sub_ind, prev_fix, cur_fix, s_t):
        """
        This method samples the next fixation given the current fixation from
        p(z_t|z_{t-1}) = s(t) * n(z_t|z_{t-1}, xi).
        It includes
        :param im_ind: index of the current image
        :param sub_ind:
        :param prev_fix:
        :param cur_fix: coordinates of the current fixation
        :param s_t:
        :return: [z_x, z_y] coordinates of the next fixation location.
        """

        xi_val = self.xi.value
        mean = cur_fix
        rad_rows = (self.rows_grid - mean[0]) ** 2
        rad_cols = (self.cols_grid - mean[1]) ** 2

        # normal distribution over the entire image
        gauss = np.exp(- rad_rows / (2 * xi_val[0]) - rad_cols / (2 * xi_val[1])) / \
                (2 * np.pi * np.sqrt(xi_val[0] * xi_val[1]))

        prob = gauss * self.saliencies[im_ind]
        prob /= prob.sum()

        # chose a pixel in the image from the distribution defined above
        inds = np.random.choice(range(self.pixels_num), 1,
                                p=prob.flatten())  # choice uses the inverse transform method in 1d
        next_fix = np.unravel_index(inds, self.saliencies[im_ind].shape)
        next_fix = np.array([next_fix[0][0], next_fix[1][0]])

        return next_fix, 0

    def generate_gammas(self):
        """
        In this model gamma = 1 for each data point.
        """
        self.gammas = []
        for i in range(len(self.fix_dists_2)):
            self.gammas.append([])
            for s in range(len(self.fix_dists_2[i])):
                self.gammas[-1].append(np.zeros(self.fix_dists_2[i][s].shape[1]))

    def sample(self, num_samples, save_steps, file_path):
        """
        This methods generates samples from the posterior distribution of xi.
        Since there is no explicit form for the posterior distribution of xi an HMC sampler is used.
        See paper for further information.
        :param num_samples: number of sampled to be generated.
        :param save_steps: whether to save the chain
        :param file_path: path where to save the chain
        :return: list of length num_samples with samples of xi
        """

        if not self.gammas:
            self.generate_gammas()
        vel = vel_param([0.1, 0.1])
        delta = 1.5
        n = 10
        m = num_samples
        # initiate an HMC instance
        hmc = HMC(self.xi, vel, delta, n, m)

        gammas_xi = [[self.gammas[i][s].copy() - 1] for i in range(len(self.gammas)) for s in
                     range(len(self.gammas[i]))]

        # perform the sampling
        hmc.HMC(gammas_xi, self.saliencies, self.fix_dists_2, self.dist_mat_per_fix)
        samples_xi = hmc.get_samples()

        if save_steps:
            with open(file_path, 'wb') as f:
                pickle.dump([samples_xi], f)

        return samples_xi

    def calc_prob_local(self, *args):
        """
        This method calculates the probability of a local step which is always 0 in the case of this model.
        :return: 0
        """
        return 0

    def calc_prob_global(self, im_ind, fixs_dists_2, sal_ts, fixs, for_nss=False):
        """
        This method calculates the probability of a global step according to the local saliency model,
        for an entire scanpath.
        p(z_t|z_{t-1}) = s(z_t) * n(z_t|z_{t-1}, xi)
        :param im_ind: index of the image
        :param fixs_dists_2: an array of shape 3 x (T -1). see set_fix_dist_2 for description.
        :param sal_ts: time series of the saliency value for each fixation. Array of length T.
        :param fixs: fixation locations. Array of shape 2 x T
        :param for_nss: whether to standerize the density for NSS or not.
        :return: array of length T with the probability of each fixation
        """

        xi = self.xi.value

        radx = (self.rows_grid[:, :, np.newaxis] - fixs[im_ind][0][0, :-1]) ** 2
        rady = (self.cols_grid[:, :, np.newaxis] - fixs[im_ind][0][1, :-1]) ** 2
        gauss = np.exp(- radx / (2 * xi[0]) - rady / (2 * xi[1])) / (2 * np.pi * np.sqrt(xi[0] * xi[1]))
        prob_all_pixels = gauss * self.saliencies[im_ind][:, :, np.newaxis]

        if for_nss:
            prob_global = prob_all_pixels / prob_all_pixels.sum(axis=(0, 1))
        else:
            # we assume here just one subject
            sub = 0
            X = fixs_dists_2[im_ind][sub]

            nominator_gauss = np.exp(- 0.5 * X[0] / xi[0] - 0.5 * X[1] / xi[1]) / \
                              (2 * np.pi * np.sqrt(xi[0] * xi[1]))

            nominator = nominator_gauss * sal_ts[im_ind][0][1:]

            prob_global = nominator / prob_all_pixels.sum(axis=(0, 1))

        return prob_global

    def calc_ros(self, *args):
        """
        This methods calculates the probability of a local step. In this model it is always 0.
        :return: 0
        """
        return 0
