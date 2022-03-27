import random
import numpy as np
from LabeledInput import LabeledInput
from ActivationFunctions import step, sigmoid, identity


class Perceptron:
    max_iterations = 100

    def __init__(self, num_inputs, activation_function, learning_rate):
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.num_inputs = num_inputs
        self.weights = [random.random() for i in range(self.num_inputs)]
        self.inputs = [0 for i in range(self.num_inputs)]
        self.bias = random.random()
        self.output = 0
        self.error_term = 0

        print(f'initial weights: {self.weights}')

    def get_weights(self):
        return self.weights

    def compute(self, inputs):
        self.inputs = inputs.copy()
        res = 0
        for i in range(self.num_inputs):
            res += (inputs[i] * self.weights[i])

        self.output = self.activation_function(res + self.bias)
        return self.output