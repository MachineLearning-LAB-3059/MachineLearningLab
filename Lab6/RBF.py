import functools
import random
from math import sqrt

import numpy as np
from LabeledInput import LabeledInput
from RBFNeuron import RBFNeuron

class RBF:
    def __init__(self, num_rbf_neurons, labeled_input: LabeledInput):
        self.W = None
        self.num_rbf_neurons = num_rbf_neurons
        self.labeled_input = labeled_input

        # choosing random centers
        indices = [i for i in range(labeled_input.number_of_training_examples)]
        random.shuffle(indices)
        indices = indices[:num_rbf_neurons]

        # finding max distance between any pair of centers
        max_distance = self.find_max_distance(indices)
        self.sigma = max_distance / sqrt(2 * num_rbf_neurons)

        self.rbf_neurons = []
        for i in range(num_rbf_neurons):
            random_input_vector = labeled_input.get_input_vector_row_major(indices[i]).copy()
            self.rbf_neurons.append(RBFNeuron(random_input_vector, self.sigma))

    def find_max_distance(self, indices):
        res = 0
        for i in range(len(indices)):
            center_1 = self.labeled_input.get_input_vector_row_major(i)
            for j in range(i + 1, len(indices)):
                center_2 = self.labeled_input.get_input_vector_row_major(j)
                difference = [0] + [center_1[k] - center_2[k] for k in range(len(center_1))]
                res = max(res, sqrt(functools.reduce(lambda y, x: y + (x * x), difference)))

        return res

    def get_activations(self):
        G = [[0 for j in range(self.num_rbf_neurons)] + [1] for i in range(self.labeled_input.number_of_training_examples)]

        for i in range(self.labeled_input.number_of_training_examples):
            input_vector = self.labeled_input.get_input_vector_row_major(i)
            for j in range(self.num_rbf_neurons):
                neuron = self.rbf_neurons[j]
                G[i][j] = neuron.get_activation(input_vector)

        return G

    def train(self):
        G = self.get_activations()
        t = self.labeled_input.get_all_outputs_column_major()

        G_inverse = np.linalg.pinv(G)
        self.W = np.dot(G_inverse, t)

        print('training_complete')

    def compute_outputs(self):
        G = self.get_activations()
        return np.dot(G, self.W)


