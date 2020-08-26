import numpy as np


def calc_dist_2_degs(fixations, start_ind):
    """This method uses the data structure num_images X num_subjecs x T containing scanpaths
     and prepares a datasructure num_images X num_subjecs x 3 X (T - 1) containing the squared L2 distance between fixations
      (square of the saccade amplitude), and the squared distance along x axis and the squared distance along y axis.
    """
    fix_dists_2 = []
    for i, image in enumerate(fixations):
        fix_dists_2.append([])

        for s, subject in enumerate(image):
            z_t = subject[:, 1 + start_ind:]
            z_t_1 = subject[:, start_ind:-1]

            a = (z_t - z_t_1) ** 2  # squared distance along each axis
            b = (z_t[0] - z_t_1[0]) ** 2 + (z_t[1] - z_t_1[1]) ** 2  # L2 squared distance
            fix_dists_2[-1].append(np.vstack([a, b]))
    return fix_dists_2


class Model:
    def __init__(self, saliencies):
        """
        :param saliencies: a list of the saliencies maps for the model.
        """

        self.fixations = None
        self.gammas = None
        self.set_start_ind()

        self.saliencies = saliencies
        self.num_images = len(saliencies)

        self.sal_shape = saliencies[0].shape
        self.pixels_num = self.sal_shape[0] * self.sal_shape[1]
        self.rows_grid, self.cols_grid = np.meshgrid(np.arange(self.sal_shape[0]), np.arange(self.sal_shape[1]),
                                                     indexing='ij')

        self.fix_dist_ind = 0

        self.num_subjects = 1
        return

    # Methods for setting stuff

    def set_start_ind(self):
        """
        start_ind is 0 if the model is first order Markov.
        For a second order Markov model start_ind should be set to 1.
        """
        self.start_ind = 0
        return

    def set_fixations(self, fixations):
        """

        :param fixations: list of lists of arrays num_images x num_subjects x T
        T is the length of each scanpath. It is different for each scanpath and that's whay we need
        to use lists rather than an array.
        :return:
        """
        self.fixations = fixations
        return

    def set_gammas(self, gammas):
        self.gammas = gammas
        return

    def remove_first_gamma(self):
        """
        The gammas of the first fixation in a generated scanpath should be ignored.
        Thus, they are removes for all scanpaths.
        """
        gammas_new = []
        for gamma in self.gammas:
            gammas_new.append([])
            for g in gamma:
                gammas_new[-1].append(g[1 + self.start_ind:])

        self.gammas_old = self.gammas
        self.gammas = gammas_new

    def retrieve_first_gamma(self):
        """
        This method allows to retrieve the gammas of the first fixation of each scanpath if neede.
        :return:
        """
        self.gammas = self.gammas_old

    # Methods for processing the fixations

    def set_saliencies_ts(self):
        """
        This method creates an attribute that includes the saliency value of each fixation in all scanpaths.
        This attribute has the same shape as self.fixations.
        :return:
        """
        self.saliencies_ts = []
        for i, saliency in enumerate(self.saliencies):
            self.saliencies_ts.append([])
            rows, cols = saliency.shape
            for s, subject in enumerate(self.fixations[i]):
                time_steps = subject.shape[1]
                tmp = []
                for t in range(self.start_ind, time_steps):
                    # get fixation location as round indices
                    r_t, c_t = int(round(subject[0, t])), int(round(subject[1, t]))
                    # make sure the indices are in the image limit
                    if r_t >= rows:
                        r_t = rows - 1
                    elif r_t < 0:
                        r_t = 0
                    if c_t >= cols:
                        c_t = cols - 1
                    elif c_t < 0:
                        c_t = 0
                    tmp.append(saliency[r_t, c_t])
                self.saliencies_ts[-1].append(np.array(tmp))

    def set_fix_dist_2_degs(self):
        """This method uses the data structure num_images X num_subjecs x T containing scanpaths
         and prepares a datasructure num_images X num_subjecs x 3 X (T - 1) containing the squared L2 distance between fixations
          (square of the saccade amplitude), and the squared distance along x axis and the squared distance along y axis.
        """
        self.fix_dists_2_degs = calc_dist_2_degs(self.fixs_degs, self.start_ind)

    def set_fix_dist_2(self):
        """This method uses the data structure num_images X num_subjecs x T containing scanpaths
         and prepares a datasructure num_images X num_subjecs x 3 X (T - 1) containing the squared L2 distance between fixations
          (square of the saccade amplitude), and the squared distance along x axis and the squared distance along y axis.
        """
        self.fix_dists_2 = calc_dist_2_degs(self.fixations, self.start_ind)

    def set_fix_dist(self):
        """This method uses the data structure num_images X num_subjecs x T containing the scanpaths
        and creates a datasructure num_images X num_subjecs x (T - 1) containing the z_t - z_t_1 for fixations
         (the sccade vector).

        :param fixations:
        :return:
        """
        self.fix_dists = []

        for i, image in enumerate(self.fixations):
            self.fix_dists.append([])

            for s, subject in enumerate(image):
                z_t = subject[:, 1:]
                z_t_1 = subject[:, :-1]

                self.fix_dists[-1].append(z_t - z_t_1)

    def calc_dist_per_fix_2(self, i_ind, s_ind):
        """
        Given a specific scanpath (for one subject and one image) this method returns
        (x_i - x_t) ** 2 and (y_i - y_t) ** 2 where i is the index of each pixel and the image
        and t is the index of each dixation in the scanpath. This will be used for set_dist_mat_per_fix.
        :param i_ind: image index
        :param s_ind: subject index
        :return:
        """

        log_norm = np.stack(
            [(self.rows_grid[:, :, np.newaxis] - self.fixations[i_ind][s_ind][:, self.start_ind:-1][0]) ** 2,
             (self.cols_grid[:, :, np.newaxis] - self.fixations[i_ind][s_ind][:, self.start_ind:-1][1]) ** 2],
            axis=3)

        return log_norm

    def set_dist_mat_per_fix(self):
        """
        This method creates a data structure num_images x num_subjects x image_height x image_width x 2
        with the values from calc_dist_per_fix_2.
        """
        self.dist_mat_per_fix = []
        for i in range(len(self.fixations)):
            self.dist_mat_per_fix.append([])
            for s in range(len(self.fixations[i])):
                res = self.calc_dist_per_fix_2(i, s)
                self.dist_mat_per_fix[-1].append(res)

    def set_angles_ts(self):
        """
        This methods calculates the angle between each saccade and the x axis
        and sets it to a class attribute.
        The shape od the created angles data is num_images x num_subjects x (T - 1)
        """
        angles_ts = []
        for i, image in enumerate(self.fixations):
            angles_ts.append([])

            for s, subject in enumerate(image):
                z_t = subject[:, 1:]
                z_t_1 = subject[:, :-1]

                angs = np.arctan2((z_t[1] - z_t_1[1]), (z_t[0] - z_t_1[0]))

                angles_ts[-1].append(angs)

        self.angles_x_ts = angles_ts

    def set_angles_between_saccades_ts(self):
        """
        This methods calculates the angle between consecutive saccade vectors
        and sets it to a class attribute.
        The shape od the created angles data is num_images x num_subjects x (T - 2)
        """
        angles_between_ts = []

        for i, image in enumerate(self.fixations):
            angles_between_ts.append([])

            for s, subject in enumerate(image):
                z_t = subject[:, 2:]
                z_t_1 = subject[:, 1:-1]
                z_t_2 = subject[:, :-2]

                vec2 = z_t_1 - z_t_2
                vec1 = z_t - z_t_1

                dot = (vec1 * vec2).sum(axis=0)
                det = (vec1[1, :] * vec2[0, :] - vec2[1, :] * vec1[0, :])
                ang = np.arctan2(det, dot)

                angles_between_ts[-1].append(ang)

        self.angles_between_ts = angles_between_ts

    # Methods for generating data from the model

    def sigmoid(self, s_t):
        """
        This method calculates the sigmoid function given a saliency value s_t.
        It is suppose to be safe also for very small or very big values of s_t
        :param s_t: saliency value (either array or scalar)
        :return: 1. / (1. + exp(b*(s_t - s_0))
        """
        arg = self.b.value * (s_t - self.s_0.value)

        try:
            return 1. / (1. + np.exp(-arg))
        except RuntimeWarning:
            if not isinstance(arg, (list, np.ndarray)):
                arg = np.array([arg])
            res = np.zeros(arg.shape)
            if len(res.shape) == 1:
                for r in range(len(res)):
                    try:
                        res[r] = 1. / (1. + np.exp(-arg[r]))
                    except RuntimeWarning:
                        if -arg[r] > 0:
                            res[r] = 0.00001
                        if -arg[r] < 0:
                            res[r] = 0.99999
            elif len(res.shape) == 3:
                for i in range(res.shape[0]):
                    for j in range(res.shape[1]):
                        for k in range(res.shape[2]):
                            try:
                                res[i, j, k] = 1. / (1. + np.exp(-arg[i, j, k]))
                            except RuntimeWarning:
                                if -arg[i, j, k] > 0:
                                    res[i, j, k] = 0.00001
                                if -arg[i, j, k] < 0:
                                    res[i, j, k] = 0.99999
            else:
                print('input should be either 1 or 3 D')
                raise ValueError
            return np.array(res)

    def generate_gamma(self, s_t):
        """
        This method generates the gamma (indecator of which policy is chosen),
        given the saliency value of the current fixation.
        Should be different for each model.
        :param s_t: saliency value
        :return: either 0 or 1
        """
        raise NotImplementedError

    def get_next_fix(self, im_ind, sub_ind, prev_fix, cur_fix, s_t):
        """
        This method generates the next fixation following a specific model.
        :param im_ind: index of the image
        :param sub_ind: index of the subject
        :param prev_fix: previous fixation location
        :param cur_fix: current fixation location
        :param s_t: value of the salirncy of the current fixation.
        :return: fixation location [x,y]
        """
        raise NotImplementedError

    def generate_data(self, time_steps, im_ind, sub_ind, random=False):
        """
        This function generates a scanpath.
        :param time_steps: length of the scanpath
        :param im_ind: index of the image - scalar (int)
        :param sub_ind: index of the subject - scalar (int)
        :param random: whether the length of the scanpath should be pertubated or not.
        :return: scanpath - 2 x T array of fixation locations [x_t, y_t]; gammas -
                array of length T of the gamma value of each fixation.
        """

        # create data sets of different lengths
        addition = np.random.randint(0, 7) if random else 0
        time_steps = time_steps + addition

        fixations = np.empty((2, time_steps))
        gammas = np.empty(time_steps)
        sal = self.saliencies[im_ind]

        # normalize the saliency if it does not sum to 1
        try:
            fix_prev = np.unravel_index(np.random.choice(range(self.pixels_num), 1,
                                                         p=sal.flatten()), sal.shape)
        except ValueError:
            sal /= sal.sum()

        # chose random initial fixation location from the saliency map.
        fix = np.unravel_index(np.random.choice(range(self.pixels_num), 1,
                                                p=sal.flatten()), sal.shape)

        steps = 0
        while steps < time_steps:
            fix_ind = np.rint(fix).astype(int)
            s_fix = self.saliencies[im_ind][fix_ind[0], fix_ind[1]]  # saliency value of the fixation location

            next_fix, gamma_t = self.get_next_fix(im_ind, sub_ind, fix_prev, fix, s_fix)
            # don't allow fixations outside of the image
            if not (next_fix[0] > self.sal_shape[0] - 1 or next_fix[1] > self.sal_shape[1] - 1 or next_fix[0] < 0 or
                    next_fix[1] < 0) and not (next_fix[0] == fix[0] and next_fix[1] == fix[1]):
                fix_prev = fix
                fix = next_fix
                fixations[:, steps] = fix
                gammas[steps] = gamma_t
                steps += 1

        return gammas, fixations

    def generate_dataset(self, time_steps, num_subjects, random=False):
        """
        This method generates scanpaths for each image and each subject.
        :param time_steps: length of the scanpaths. Either a scalar or an array.
        :param num_prticipants: number of subjects
        :param random: whether the length of the scanpath should be pertubated or not.
        :return: dataset of size num_images x num_subjects x 2 x T containing a scanpath
                for each image and each subject.
        """
        data_fix = []
        data_gamma = []

        if not isinstance(time_steps, (np.ndarray, list)):
            time_steps = np.repeat(time_steps, self.num_images)
            random = True

        for im_ind in range(self.num_images):
            data_fix.append([])
            data_gamma.append([])
            for j in range(num_subjects):
                gammas, fixations = self.generate_data(time_steps[im_ind], im_ind, j, random)
                data_gamma[-1].append(gammas)
                data_fix[-1].append(fixations)

        return data_gamma, data_fix

    # Methods for calculating the likelihood for a given data-set

    def calc_prob_local(self):
        """
        This method calculates the probability of a local step according to the specific model.
        :return:
        """
        raise NotImplementedError

    def calc_prob_global(self):
        """
        This method calculates the probability of a global step according to the specific model.
        :return:
        """
        raise NotImplementedError

    def calc_ros(self):
        """
        This methods calculates the probability of a local step.
        :return:
        """
        raise NotImplementedError

    def calc_log_likelihood_per_scanpath(self, im_ind, fixs_dists_2, sal_ts, fixs, per_fixation, for_nss=False,
                                         saliencies=None):
        """
        This method calculates the likelihood of a scanpath according to:
        log(p(scanpath)) = sum_{fix}(log_2(rho * p_{local}(fix) + (1 - rho) * p_{global}(fix))).
        It can return either the sum above or a vector of the log likelihood of each fixation.
        It can also return the equivalent of NSS - at each step normalizing the likelihood
        to have 0 mean and variance 1 over the entire image.
        :param im_ind: index of the image
        :param fixs_dists_2: an array of shape 3 x (T -1). see set_fix_dist_2 for description.
        :param sal_ts: time series of the saliency value for each fixation. Array of length T.
        :param fixs: fixation locations. Array of shape 2 x T
        :param per_fixation: Whether to return the sum over the entire scanpath or a vector.
        :param for_nss: whether to standerize the density for NSS or not.
        :param saliencies: if not specified self.saliencies will be used
        :return: the log-likelihood of the scanpath, either as a vector (log-likelihood per fixation)
                or the value for the entire scanpath.
        """

        if saliencies is None:
            saliencies = self.saliencies

        ros = self.calc_ros(im_ind, sal_ts, for_nss, saliencies)

        prob_local = self.calc_prob_local(im_ind, fixs_dists_2, fixs, for_nss)

        prob_global = self.calc_prob_global(im_ind, fixs_dists_2, sal_ts, fixs, for_nss)

        prob = ros * prob_local + (1 - ros) * prob_global

        if for_nss:
            normed_prob = (prob - prob.mean(axis=(0, 1))) / prob.std(axis=(0, 1))
            scanpath_length = normed_prob.shape[2]
            nss_per_fix = normed_prob[
                tuple(fixs[im_ind][0][0, self.fix_dist_ind + 1:].astype(int)), tuple(
                    fixs[im_ind][0][1, self.fix_dist_ind + 1:].astype(int)), tuple(
                    np.arange(scanpath_length))]
            return nss_per_fix

        log_like = np.log2(prob)

        if per_fixation:
            return log_like
        else:
            return log_like.sum()

    def calculate_likelihood_per_subject(self, fixs_dists_2=None, sal_ts=None, fixs=None, per_fixation=False,
                                         for_nss=False, saliencies=None):

        """
        This method calculates the likelihood of a set of scanpath according to:
        log(p(scanpath)) = sum_{scanpaths}sum_{fix}(log_2(rho * p_{local}(fix) + (1 - rho) * p_{global}(fix))).
        It can return either the sum above or a vector of the log likelihood of each fixation.
        It can also return the equivalent of NSS - at each step normalizing the likelihood
        to have 0 mean and variance 1 over the entire image.
        :param fixs_dists_2: a list of arrays of shape 3 x (T -1). see set_fix_dist_2 for description.
        :param sal_ts: a list of arrays of the saliency value for each fixation in each scanpath.
        :param fixs: fixation locations. Array of shape 2 x T. If not given than self.fixs is used.
        :param per_fixation: Whether to return the sum over the entire scanpath or a vector.
        :param for_nss: whether to standerize the density for NSS or not.
        :param saliencies: if not specified self.saliencies will be used
        :return: the log-likelihood of the dataset, either as a vector for each scanpath (log-likelihood per fixation)
                or the value for the entire dataset.
        """

        if fixs is None:
            fixs = self.fixations
            sal_ts = self.saliencies_ts
            fixs_dists_2 = self.fix_dists_2
        likelihood = 0
        likelihood_per_fixation = []
        nss_per_fixation = []
        for im_ind in range(self.num_images):
            like = self.calc_log_likelihood_per_scanpath(im_ind, fixs_dists_2, sal_ts, fixs,
                                                         per_fixation, for_nss, saliencies)

            if per_fixation:
                likelihood_per_fixation.append(like)
            elif for_nss:
                nss_per_fixation.append(like)
            else:
                likelihood += like

        if per_fixation:
            return likelihood_per_fixation
        elif for_nss:
            return nss_per_fixation
        else:
            return likelihood

    # Methods for sampling the model parameters

    def sample(self, *args):
        """
        A Gibbs sampler for the model parameters. Should be implemented separately for each model.
        :return: Chain of samples for each parameter.
        """
        raise NotImplementedError
