import sys
import autograd.numpy as np
from autograd import grad
from scipy.stats import invgamma

sys.path.append('./../../')
from src.HMC.hmcparameter import HMCParameter


class XiParam(HMCParameter):
    """
    This class implements the xi parameter and the methods needed for its inference by the HMC sampler.
    For the gradient of the energy it uses the automatic derivation with the autograd library.
    """
    def __init__(self, is_fixed=False, alpha=4, betta=200):
        """

        :param is_fixed: If set True the parameter is not sampled, and only its value is returned.
        :param alpha: shape of the prior distribution.
        :param betta: scale of the prior distribution.
        """
        self.alpha = np.array(alpha)
        self.betta = np.array(betta)
        self.eff_xi = 1
        self.is_fixed = is_fixed
        self.energy_grad_func = grad(self.energy, 0) # set the gradient function using autograd

    def check_value(self, value):
        """
        Makes sure that the value is length 2 array [z_x, z_y]
        :param value:
        :return:
        """
        if np.array(value).shape and len(np.array(value)) == 2:
            return True
        else:
            return False

    def set_init_value(self, init_value):
        self.init_value = init_value.copy()
        self.value = init_value.copy()

    def get_energy_grad(self, gammas, saliencies, fix_dist_2, dist_mat_per_fix):
        """
        This method returns the gradient of the energy of xi.
        :param gammas: list of lists of arrays of length T(image, sub) with 0s
        :param saliencies: list of all saliencies
        :param fix_dist_2: a list of arrays of shape 3 x (T -1).
        see set_fix_dist_2 in LocalGlobalAttentionModel.model.Model for description
        :param dist_mat_per_fix: a list of of lists of mtrices num_pixels x num_pixels x T.
        Each num_pixels x num_pixels matrix contains the distance f all pixels in the image from fixation location
        fixs[image][subject][t]
        :return: the gradient of the energy. scalar.
        """
        res = self.energy_grad_func(self.value, gammas, saliencies, fix_dist_2, dist_mat_per_fix)

        return res

    def energy(self, xi, gammas, saliencies, fix_dist_2, dist_mat_per_fix):
        """
        This method implements the energy (- log (likelihood * prior)) of the model parameter xi.
        For a description of the calculation look at the likelihood of the model at the paper.
        :param xi: current value of the model parameter.
        :param gammas: list of lists of arrays of length T(image, sub) with 0s
        :param saliencies: list of all saliencies
        :param fix_dist_2: a list of arrays of shape 3 x (T -1).
        see set_fix_dist_2 in LocalGlobalAttentionModel.model.Model for description
        :param dist_mat_per_fix: a list of of lists of mtrices num_pixels x num_pixels x T.
        Each num_pixels x num_pixels matrix contains the distance f all pixels in the image from fixation location
        fixs[image][subject][t]
        :return: the energy of xi. scalar.
        """
        # xi must be larger than 0.
        # If it is smaller than 0 we would like to reject the sample so we set its energy to inf.
        if (xi < 0).any():
            return np.inf

        # this is the negative log likelihood
        summing = sum([sum((gammas[i][s]) * (- fix_dist_2[i][s][0, :] / (2 * xi[0]) \
                                                 - fix_dist_2[i][s][1, :] / (2 * xi[1])
                                                 - np.log(
                    (saliencies[i][:, :, np.newaxis] * np.exp(
                        - dist_mat_per_fix[i][s][:, :, :, 0] / (2 * xi[0]) \
                        - dist_mat_per_fix[i][s][:, :, :, 1] / (2 * xi[1]))).sum(axis=(0, 1)))))
                       for i in range(len(dist_mat_per_fix))
                       for s in range(len(dist_mat_per_fix[i]))])

        # we add the negative log prior to the negative log likelihood to get the energy.
        tmp = summing + (self.alpha[0] + 1) * np.log(xi[0]) + self.betta[0] / xi[0] + \
              (self.alpha[1] + 1) * np.log(xi[1]) + self.betta[1] / xi[1]
        return tmp

    def get_energy(self, gammas, saliencies, fix_dist_2, dist_mat_per_fix):
        return self.energy(self.value, gammas, saliencies, fix_dist_2, dist_mat_per_fix)

    def get_energy_for_value(self, value, gammas, saliencies, fix_dist_2, dist_mat_per_fix):
        return self.energy(value, gammas, saliencies, fix_dist_2, dist_mat_per_fix)

    def prior(self):
        """
        The prior of xi is set to be inverse gamma.
        :return:
        """
        return invgamma.rvs(self.alpha, scale=self.betta)
