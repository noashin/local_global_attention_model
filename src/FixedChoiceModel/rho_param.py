import sys
from scipy.stats import beta

sys.path.append('./../../')
from src.LocalGlobalAttentionModel.model_param import ModelParam


class RhoParam(ModelParam):
    """
    This class implements the rho parameter as it appears in
    the Fixed Choice model. For details see the paper.
    """

    def __init__(self, is_fixed=False, a_p=4, b_p=4):
        super().__init__(is_fixed)

        self.a_p = a_p
        self.b_p = b_p
        self.time_steps = None

    def prior(self):
        """
        This method samples from the prior distribution of rho.
        The prior distribution of rho is a betta distribution.
        :return: sample from the prior distribution of rho
        """
        return beta.rvs(self.a_p, self.b_p)

    def set_num_time_steps(self, gammas):
        self.time_steps = sum([len(gammas[i][su]) for i in range(len(gammas)) for su in range(len(gammas[i]))])

    def conditional_posterior(self, gammas):
        """
        This method samples from the conditional distribution of rho.
        As the model is conjugate with respect to rho the posterior is a betta distribution.
        :param gammas: the policy indicators for all data points.
        :return: A sample from the conditional posterior distribution of rho.
        """

        if not self.is_fixed:
            sum_gammas = sum([sum(gammas[i][su]) for i in range(len(gammas)) for su in range(len(gammas[i]))])

            a = sum_gammas + self.a_p
            b = self.time_steps - sum_gammas + self.b_p
            return beta.rvs(a, b)

        else:
            return self.value
