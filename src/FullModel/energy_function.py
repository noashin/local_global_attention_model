import autograd.numpy as np
from autograd import grad


def likelihood_energy_per_image(epsilon, xi, cov_ratio, saliencies_im, gammas_im, fix_dists_2_im, dist_mat_per_fix_im,
                                fix_dist_ind):
    """
    This function calculates the energy of the data from one image (one saliency map)
    The data is assumed to be only for one subject.
    This is the negative log(likelihood) for xi and epsilon. See paper for the computation details.
    :param epsilon: the value of epsilon
    :param xi: the value of xi
    :param cov_ratio: scaling of epsilon
    :param saliencies_im: the saliency map of the image
    :param gammas_im: an array of the gamma value of each fixation in the scanpath of the image.
    :param fix_dists_2_im: a list of arrays of shape 3 x (T -1).
        see set_fix_dist_2 in LocalGlobalAttentionModel.model.Model for description
    :param dist_mat_per_fix_im: a list of matrices num_pixels x num_pixels x T.
    Each num_pixels x num_pixels matrix contains the distance f all pixels in the image from fixation location
    fixs[image][subject][t]
    :param fix_dist_ind: 0 if the model is 1 markov, 1 if the model is 2 markov
    :return: the negative log likelihood for the terms that involve epsilon and xi
    """
    delta = 10 ** -200

    # Assuming only one subject!!!
    sub = 0

    local_policy_nominator = - fix_dists_2_im[sub][0, fix_dist_ind:] / (2 * epsilon[0]) - fix_dists_2_im[sub][1,
                                                                                          fix_dist_ind:] / (
                                     2 * epsilon[1])
    local_policy_denominator = np.log(np.exp(
        - dist_mat_per_fix_im[sub][:, :, fix_dist_ind:, 0] / (2 * epsilon[0]) - dist_mat_per_fix_im[sub][:, :,
                                                                                fix_dist_ind:, 1] /
        (2 * epsilon[1])).sum(axis=(0, 1)))

    local_policy = - gammas_im[sub] * (local_policy_nominator - local_policy_denominator)

    # the difference between a gaussian with xi covariance and a gaussian with epsilon/cov_ratio covariance.
    # this creates the repulsion of the global attention policy.
    tmp_diff = np.exp(
        - fix_dists_2_im[sub][0, fix_dist_ind:] / (2 * xi[0]) - fix_dists_2_im[sub][1, fix_dist_ind:] / (
                2 * xi[1])) / (
                       2 * np.pi * np.sqrt(xi[0] * xi[1])) - \
               np.exp(- fix_dists_2_im[sub][0, fix_dist_ind:] / (2 * (epsilon / cov_ratio)[0]) - fix_dists_2_im[sub][1,
                                                                                                 fix_dist_ind:] / (
                              2 * (epsilon / cov_ratio)[1])) / (
                       2 * np.pi * np.sqrt((epsilon / cov_ratio)[0] * (epsilon / cov_ratio)[1]))
    tmp_max = np.maximum(tmp_diff, delta)
    global_policy_nominator = np.log(tmp_max)

    global_policy_denominator = np.log((np.maximum(
        np.exp(- dist_mat_per_fix_im[sub][:, :, fix_dist_ind:, 0] / (2 * xi[0]) - dist_mat_per_fix_im[sub][:, :,
                                                                                  fix_dist_ind:, 1] /
               (2 * xi[1])) / (2 * np.pi * np.sqrt(xi[0] * xi[1])) -
        np.exp(
            - dist_mat_per_fix_im[sub][:, :, fix_dist_ind:, 0] / (2 * (epsilon / cov_ratio)[0]) - dist_mat_per_fix_im[
                                                                                                      sub][
                                                                                                  :,
                                                                                                  :, fix_dist_ind:, 1] /
            (2 * (epsilon / cov_ratio)[1])) / (
                2 * np.pi * np.sqrt((epsilon / cov_ratio)[0] * (epsilon / cov_ratio)[1])), delta) * saliencies_im[:,
                                                                                                    :,
                                                                                                    np.newaxis]).sum(
        axis=(0, 1)))

    global_policy = (gammas_im[sub] - 1) * (global_policy_nominator - global_policy_denominator)

    energy = (local_policy + global_policy).sum()

    return energy


def priors_energy(epsilon, xi, alpha_eps, betta_eps, alpha_xi, betta_xi):
    """
    This function calculates the log(prior) for xi and epsilon
    :param epsilon: value of epsilon [eps_x, eps_y]
    :param xi: value of xi [xi_x, xi_y]
    :param alpha_eps: shape of epsilon prior [alpha_eps_x, alpha_eps_y]
    :param betta_eps: scale of epsilon prior [betta_eps_x, betta_eps_y]
    :param alpha_xi: shape of xi prior [alpha_xi_x, alpha_xi_y]
    :param betta_xi: scale of xi prior [alpha_xi_x, alpha_xi_y]
    :return: the negative log prior that involves epsilon and xi
    """
    return (alpha_xi[0] + 1) * np.log(xi[0]) + betta_xi[0] / xi[0] + (alpha_xi[1] + 1) * np.log(xi[1]) + \
           betta_xi[1] / xi[1] + \
           (alpha_eps[0] + 1) * np.log(epsilon[0]) + betta_eps[0] / epsilon[0] + (alpha_eps[1] + 1) * np.log(
        epsilon[1]) + betta_eps[1] / xi[1]


def get_tot_energy(epsilon, xi, cov_ratio, saliencies, gammas, fix_dists_2, dist_mat_per_fix, alpha_eps,
                   betta_eps, alpha_xi, betta_xi, fix_dist_ind=1):
    """
    This function calculates the energy of xi and epsilon which is the negative log likelihood
    with the terms that depend on epsilon and xi.
    :param epsilon: value for epsilon. np.array([eps_x, eps_y])
    :param xi: value for xi. np.array([xi_x, xi_y])
    :param cov_ratio: factor for scaling epsilon in the repulsion
    :param saliencies: list of saliencies. num_images x num_pixels x num_pixels
    :param gammas: list of lists of time series with the policy indication (\in {0,1}).
                    num_images x num_subjects x T(image, subject)
    :param fix_dists_2: a list of lists of arrays of shape 3 x (T(image, subject) -1).
        see set_fix_dist_2 in LocalGlobalAttentionModel.model.Model for description
    :param dist_mat_per_fix: a list of matrices num_pixels x num_pixels x T.
    Each num_pixels x num_pixels matrix contains the distance f all pixels in the image from fixation location
    fixs[image][subject][t]
    :param alpha_eps: shape of epsilon prior [alpha_eps_x, alpha_eps_y]
    :param betta_eps: scale of epsilon prior [betta_eps_x, betta_eps_y]
    :param alpha_xi: shape of xi prior [alpha_xi_x, alpha_xi_y]
    :param betta_xi: scale of xi prior [alpha_xi_x, alpha_xi_y]
    :param fix_dist_ind: 0 if the model is 1 markov, 1 if the model is 2 markov
    :return: the energy of the model (only terms that depend on xi and epsilon)
    """

    tot_energy = np.array([likelihood_energy_per_image(epsilon, xi, cov_ratio, saliencies[i], gammas[i], fix_dists_2[i],
                                                       dist_mat_per_fix[i], fix_dist_ind) for i in
                           range(len(saliencies))]).sum()
    tot_energy += priors_energy(epsilon, xi, alpha_eps, betta_eps, alpha_xi, betta_xi)
    return tot_energy


def get_tot_energy_grad(epsilon, xi, cov_ratio, saliencies, gammas, fix_dists_2, dist_mat_per_fix, alpha_eps, betta_eps,
                        alpha_xi,
                        betta_xi, param_index, fix_dist_ind=1):
    """
    This function calculates the gradient of the energy with respect to either xi or epsilon.
    :param epsilon: value for epsilon. np.array([eps_x, eps_y])
    :param xi: value for xi. np.array([xi_x, xi_y])
    :param cov_ratio: factor for scaling epsilon in the repulsion
    :param saliencies: list of saliencies. num_images x num_pixels x num_pixels
    :param gammas: list of lists of time series with the policy indication (\in {0,1}).
                    num_images x num_subjects x T(image, subject)
    :param fix_dists_2: a list of lists of arrays of shape 3 x (T(image, subject) -1).
        see set_fix_dist_2 in LocalGlobalAttentionModel.model.Model for description
    :param dist_mat_per_fix: a list of matrices num_pixels x num_pixels x T.
    Each num_pixels x num_pixels matrix contains the distance f all pixels in the image from fixation location
    fixs[image][subject][t]
    :param alpha_eps: shape of epsilon prior [alpha_eps_x, alpha_eps_y]
    :param betta_eps: scale of epsilon prior [betta_eps_x, betta_eps_y]
    :param alpha_xi: shape of xi prior [alpha_xi_x, alpha_xi_y]
    :param betta_xi: scale of xi prior [alpha_xi_x, alpha_xi_y]
    :param param_index: 0 if the gradient is with respect to epsilon. 1 if the gradient is with respect to  xi.
    :param fix_dist_ind: 0 if the model is 1 markov, 1 if the model is 2 markov
    :return: the gradient of the energy with respect to either xi or epsilon.
    """
    energy_likelihood_grad = grad(likelihood_energy_per_image, param_index)
    tot_grad = np.array([energy_likelihood_grad(epsilon, xi, cov_ratio, saliencies[i], gammas[i], fix_dists_2[i],
                                                dist_mat_per_fix[i], fix_dist_ind) for i in range(len(saliencies))]).sum()

    energy_prior_grad = grad(priors_energy, param_index)
    tot_grad += energy_prior_grad(epsilon, xi, alpha_eps, betta_eps, alpha_xi, betta_xi)
    return tot_grad
