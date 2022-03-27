import random
import numpy as np
from LabeledInput import LabeledInput
from ActivationFunctions import step, sigmoid, identity


class Perceptron:
    max_iterations = 100

    def __init__(self, activation_function, labelled_input: LabeledInput, learning_rate):
        self.activation_function = activation_function
        self.labelled_input = labelled_input
        self.learning_rate = learning_rate
        self.num_inputs = labelled_input.get_input_vector_size()
        self.weights = np.array([[random.random()] for i in range(self.num_inputs)])

        print(f'initial weights: {self.weights}')
        print(f'input vectors in column major')
        print(labelled_input.input_vectors_column_major)
        print(f'output vectors in columns major ')
        print(labelled_input.output_vector)

    def get_output_for_ith_training_example(self, i):
        input_vector = self.labelled_input.get_input_vector(i)
        res = ((np.transpose(self.weights)).dot(input_vector))[0][0]
        return self.activation_function(res)

    def train(self, training_algorithm):
        training_algorithm(self)

    def get_output_for_all_training_examples(self):
        res = []
        for i in range(self.labelled_input.get_number_of_training_examples()):
            res.append(self.get_output_for_ith_training_example(i))

        return res

    def get_weights(self):
        return self.weights

    def plot_2d(self, plotter, lim):
        a = self.weights[1][0]
        b = self.weights[2][0]
        c = self.weights[0][0]

        x = np.linspace(-lim, lim, 1000)
        y = [((-c - (a * v)) / b) for v in x]

        plotter.plot(x, y)

    def plot_1d(self, plotter, lim):
        a = self.weights[1][0]
        b = self.weights[0][0]

        x = np.linspace(-lim, lim, 1000)
        y = [(a * v) + b for v in x]

        plotter.plot(x, y)