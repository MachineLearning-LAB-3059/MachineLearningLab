import functools
import numpy as np


class RBFNeuron:
    def __init__(self, center, sigma):
        self.center = center
        self.dimension = len(center)
        self.sigma = sigma

    def get_activation(self, input_vector):
        difference_vector = [0] + [self.center[i] - input_vector[i] for i in range(self.dimension)]
        numerator = - functools.reduce(lambda y, x: y + (x * x), difference_vector)
        denominator = 2 * self.sigma * self.sigma

        exp_term = numerator / denominator
        return np.exp(exp_term)

