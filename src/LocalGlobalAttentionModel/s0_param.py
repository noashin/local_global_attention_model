import numpy as np
from scipy.stats import truncnorm

from .model_param import ModelParam


class S0Param(ModelParam):

    def __init__(self, is_fixed=False, m_p=0.00018, s_p=0.0000001, lower=0.00000001, upper=5.):
        """
        :param is_fixed:
        :param m_p: mean of the prior
        :param s_p: variance of the prior
        :param lower: lower bound
        :param upper: upper bound
        """

        super().__init__(is_fixed)

        self.m_p = m_p
        self.s_p = s_p
        self.lower = lower
        self.upper = upper

    def prior(self):
        """
        A truncated normal distribution
        :return: A sample for a truncated normal distribution according to the parameter's values.
        """
        return truncnorm.rvs((self.lower - self.m_p) / np.sqrt(self.s_p),
                             (self.upper - self.m_p) / np.sqrt(self.s_p), loc=self.m_p, scale=np.sqrt(self.s_p))

    def conditional_posterior(self, gammas, b, w, saliencies_ts):

        """
        This method generates a sample from the conditional posterior distribution of
        the b parameter of the model (see details in the paper). The distribution is Gaussian.
        :param gammas: The current values of the gamma parameters. list of lists of arrays.
                        Shape is num_images x num_subjects x (T-1) (T is different for each image and subject).
        :param b: current value of the b parameter.
        :param w: current values of the w augmenting parameter. Same structure as gammas.
                  num_images x num_subjects x T
        :param saliencies_ts: The saliency value of each fixation in the data. Same structure as w.
        :return: a sample from the posterior distribution if the parameter is not fixed.
                If fixed returns the current value of the parameter.
        """

        if not self.is_fixed:
            s = self.s_p / (1 + self.s_p * b ** 2 * sum([sum(w[i][su][:-1]) for i in range(len(saliencies_ts))
                                                         for su in range(len(saliencies_ts[i]))]))

            m = (sum(
                [sum(-b * (gammas[i][su] - 0.5) + saliencies_ts[i][su][:-1] * w[i][su][:-1] * b ** 2) for i in
                 range(len(saliencies_ts))
                 for su in range(len(saliencies_ts[i]))]) + self.m_p / self.s_p) * s

            return np.random.normal(m, np.sqrt(s))

        else:
            return self.value
