class ModelParam:
    """
    This class describes a parameter of the model.
    """

    def __init__(self, is_fixed):
        """

        :param is_fixed: if fixed the parameter will not be sampled during the inference.
        """
        self.is_fixed = is_fixed

        self.init_value = None
        self.value = None
        self.num_images = None

    def sample_init_value(self):
        """
        This method samples the initial value for the parameter from the prior distribution.
        """
        val = self.prior()

        self.init_value = val
        self.value = val

    def set_value(self, value):
        self.value = value

    def set_init_value(self, init_value):
        self.init_value = init_value
        self.value = init_value

    def set_num_images(self, num_images):
        self.num_images = num_images

    def get_shape(self):
        """

        :return: The shape of the parameter. Either 1 is the parameter is scalar, or 2 if it is a 2d parameter.
        """
        raise NotImplementedError

    def prior(self):
        """
        Implementation of the prior distribution for the model parameter.
        """
        raise NotImplementedError

    def conditional_posterior(self, *args):
        """
        This method implement the conditional posterior of the parameter (if it is available in explicit form).
        """
        raise NotImplementedError

    def get_sample(self, *args):
        """
         If the parameter is fixed its value is returned. Otherwise a value is sampled from the conditional posterior.
        :return: a sample for the chain
        """
        if self.is_fixed:
            return self.value
        else:
            return self.conditional_posterior(*args)
