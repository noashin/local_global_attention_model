import pickle

import numpy as np


class HMC():
    """
    This class is a minimal implementation of the HMC sampler with a leap frog integrator.
    """

    def __init__(self, state_param, velocity_param, delta=0.1, n=10, m=500, save_steps=False, output_path=''):
        """

        :param state_param: The state - the parameter which we are actually interested to sample.
        :param velocity_param: The velocity - the augmenting parameter.
        :param delta: step size
        :param n: number of samples.
        :param m: number of leap frog steps.
        :param save_steps: whether to save intermediate samples or not.
        :param output_path: path to a file where the results should be saved.
        """
        self.state_param = state_param
        self.velocity_param = velocity_param
        self.delta = delta
        self.n = n
        self.m = m

        self.save_steps = save_steps
        self.output_path = output_path

        self.state_samps = []
        self.vel_samps = []
        self.alphas = []
        self.all_state_samps = []
        self.all_vel_samps = []
        self.leap_frog_steps = []

    def get_state(self):
        return self.state_param

    def get_samples(self):
        return self.state_samps

    def integrate(self, save_steps, *args):
        self.leapfrog(save_steps, self.state_param, self.velocity_param, *args)

    def leapfrog(self, save_steps, state_param, velocity_param, *args):
        """
        This method implements the leap frog integrator.
        This method updates the values of the state and velocity parameters, so it doesnt return anything.
        :param save_steps: whether to save the intermediate steps of the leapfrog integrator or not.
        :param state_param: the state of the system
        :param velocity_param: the velocity of the systems
        :param args: parameters for the energy and energy gradient functions
        """

        # Half a step
        tmp_vel = velocity_param.get_value() - self.delta / 2 * state_param.get_energy_grad(*args)
        velocity_param.set_value(tmp_vel)

        for i in range(1, self.n):

            state_val = state_param.get_value() + self.delta * velocity_param.get_energy_grad()
            state_param.set_value(state_val)

            vel_val = velocity_param.get_value() - self.delta * state_param.get_energy_grad(*args)
            velocity_param.set_value(vel_val)

            if save_steps:
                self.leap_frog_steps.append([state_val, vel_val])

        # another half a step
        state_val = state_param.get_value() + self.delta * velocity_param.get_energy_grad()
        state_param.set_value(state_val)

        vel_val = velocity_param.get_value() - self.delta / 2 * state_param.get_energy_grad(*args)
        # negate momentum to make the proposal symmetric
        velocity_param.set_value(- vel_val)

    def HMC(self, *args):
        """
        The HMC sampler
        :param args: arguments needed for the energy and energy gradient functions.
        :return:
        """
        self.state_samps.append(self.state_param.get_value())
        for i in range(self.m):
            self.velocity_param.gen_init_value()
            vel_val_old = self.velocity_param.get_value()
            state_val_old = self.state_param.get_value()

            self.integrate(False, *args)

            # if the value of the state is none we would like to reject the sample, so we set the energy to be inf.
            if np.isnan(self.state_param.get_value()).any():
                state_energy_new = np.inf
            else:
                state_energy_new = self.state_param.get_energy(*args)

            # probability to accept the samples
            prob = np.exp(- state_energy_new + self.state_param.get_energy_for_value(
                state_val_old, *args) - self.velocity_param.get_energy() + self.velocity_param.get_energy_for_value(
                vel_val_old))

            alpha = np.min((1, prob))

            self.alphas.append(alpha)
            self.all_state_samps.append(self.state_param.get_value())
            self.all_vel_samps.append(self.velocity_param.get_value())

            p = np.random.random()

            if not p < alpha:
                # p < nan always False and we would like to reject if this happens
                # reject
                self.state_param.set_value(state_val_old)
                self.velocity_param.set_value(vel_val_old)

            # accept
            self.state_samps.append(self.state_param.get_value())
            self.vel_samps.append(self.velocity_param.get_value())

            # save intermediate results every 50 iterations
            if self.save_steps and not i % 50:
                with open(self.output_path, 'wb') as f:
                    pickle.dump(self.state_samps, f)
