import sys
import autograd.numpy as np
from scipy.stats import invgamma
from .energy_function import get_tot_energy, get_tot_energy_grad

sys.path.append('./../../')
from src.HMC.hmcparameter import HMCParameter


class EpsParam(HMCParameter):
    """
    This class implements the necessary methods for the epsilon parameter of the model.
    for the details of the energy see the module energy_function
    """
    def __init__(self, is_fixed=False, alpha=4, betta=200, fix_dist_start_ind=1):
        """

        :param is_fixed:
        :param alpha: shape of the prior distribution
        :param betta: scale of the prior distribution
        :param fix_dist_start_ind: 0 if the model is 1 markov, 2 if the model is 1 markov
        """
        self.alpha = np.array(alpha)
        self.betta = np.array(betta)
        self.is_fixed = is_fixed
        self.fix_dist_start_ind = fix_dist_start_ind

    def check_value(self, value):
        """
        Make sure the value is length 2
        :param value:
        :return: True if value is of length 2. Otherwise False
        """
        if np.array(value).shape and len(np.array(value)) == 2:
            return True
        else:
            return False

    def set_init_value(self, init_value):
        self.init_value = init_value.copy()
        self.value = init_value.copy()

    def get_energy_grad(self, xi, cov_ratio, saliencies, gammas, fix_dists_2, dist_mat_per_fix, alpha_xi,
                        betta_xi):
        """
        This method returns the gradient of the energy of the current value of epsilon
        :param xi: the value of xi parameter
        :param cov_ratio: fixed scaling of epsilon
        :param saliencies: list of all the saliency maps
        :param gammas: list of lists of array of length T(image, subject) of the policy indicators (\in {0,1})
        :param fix_dists_2: a list of arrays of shape 3 x (T -1).
        see set_fix_dist_2 in LocalGlobalAttentionModel.model.Model for description
        :param dist_mat_per_fix: a list of of lists of mtrices num_pixels x num_pixels x T.
        Each num_pixels x num_pixels matrix contains the distance f all pixels in the image from fixation location
        fixs[image][subject][t]
        :param alpha_xi: the shape of the prior distribution of xi
        :param betta_xi: the scale of the prior distribution of xi
        :return: The gradient of the energy of epsilon for its current value
        """
        res = get_tot_energy_grad(self.value, xi, cov_ratio, saliencies, gammas, fix_dists_2, dist_mat_per_fix,
                                  self.alpha, self.betta, alpha_xi,
                                  betta_xi, 0, self.fix_dist_start_ind)

        return res

    def get_energy(self, xi, cov_ratio, saliencies, gammas, fix_dists_2, dist_mat_per_fix, alpha_xi, betta_xi):
        """
        This method calculates the energy of epsilon for its current value.
        :param xi: the value of xi parameter
        :param cov_ratio: fixed scaling of epsilon
        :param saliencies: list of all the saliency maps
        :param gammas: list of lists of array of length T(image, subject) of the policy indicators (\in {0,1})
        :param fix_dists_2: a list of arrays of shape 3 x (T -1).
        see set_fix_dist_2 in LocalGlobalAttentionModel.model.Model for description
        :param dist_mat_per_fix: a list of of lists of mtrices num_pixels x num_pixels x T.
        Each num_pixels x num_pixels matrix contains the distance f all pixels in the image from fixation location
        fixs[image][subject][t]
        :param alpha_xi: the shape of the prior distribution of xi
        :param betta_xi: the scale of the prior distribution of xi
        :return: the energy of the current value of epsilon
        """
        return get_tot_energy(self.value, xi, cov_ratio, saliencies, gammas, fix_dists_2, dist_mat_per_fix,
                                       self.alpha, self.betta, alpha_xi, betta_xi, self.fix_dist_start_ind)

    def get_energy_for_value(self, value, xi, cov_ratio, saliencies, gammas, fix_dists_2, dist_mat_per_fix, alpha_xi,
                             betta_xi):
        """
        This method calculates the energy of epsilon for a given value
        :param value: specific value of epsilon for which the energy will be calculated
        :param xi: the value of xi parameter
        :param cov_ratio: fixed scaling of epsilon
        :param saliencies: list of all the saliency maps
        :param gammas: list of lists of array of length T(image, subject) of the policy indicators (\in {0,1})
        :param fix_dists_2: a list of arrays of shape 3 x (T -1).
        see set_fix_dist_2 in LocalGlobalAttentionModel.model.Model for description
        :param dist_mat_per_fix: a list of of lists of mtrices num_pixels x num_pixels x T.
        Each num_pixels x num_pixels matrix contains the distance f all pixels in the image from fixation location
        fixs[image][subject][t]
        :param alpha_xi: the shape of the prior distribution of xi
        :param betta_xi: the scale of the prior distribution of xi
        :return: the energy of epsilon at value
        """
        return get_tot_energy(value, xi, cov_ratio, saliencies, gammas, fix_dists_2, dist_mat_per_fix,
                                       self.alpha, self.betta, alpha_xi, betta_xi, self.fix_dist_start_ind)

    def prior(self):
        """
        The prior for epsilon is an inverse gamma distribution
        :return: a sample from the prior distribution
        """
        return invgamma.rvs(self.alpha, scale=self.betta)
