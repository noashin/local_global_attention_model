class HMCParameter:
    """
    This class is a template for a parameter (either state or velocity) for an HMC sampler.
    """
    def __init__(self, init_val):
        self.value = init_val

    def check_value(self, value):
        return True

    def set_value(self, value):
        if self.check_value(value):
            self.value = value
        else:
            raise ValueError

    def get_value(self):
        return self.value

    def get_energy_grad(self, *args):
        raise NotImplementedError

    def get_energy_for_value(self, value, *args):
        raise NotImplementedError

    def get_energy(self, *args):
        raise NotImplementedError
