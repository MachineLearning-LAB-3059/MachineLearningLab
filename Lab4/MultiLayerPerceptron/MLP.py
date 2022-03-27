from Perceptron import Perceptron
from MLPLayer import MLPLayer
import numpy as np

class MLP:

    def __init__(self, num_layers, layer_dimension, input_dimension, output_dimension, activation_function, learning_rate):
        self.num_layers = num_layers
        self.input_dimension = input_dimension
        self.output_dimensions = output_dimension
        self.activation_function = activation_function
        self.learning_rate = learning_rate

        self.input_layer = MLPLayer(2, 2, activation_function, learning_rate)

        self.output_layer = MLPLayer(output_dimension, layer_dimension, activation_function, learning_rate)
        # one hidden layer as of now
        self.layers = [self.input_layer, MLPLayer(layer_dimension, 2, activation_function, learning_rate), self.output_layer]

    def feed_forward(self, inputs):
        cur_inputs = inputs
        for layer in self.layers:
            layer.update_outputs(cur_inputs)
            cur_inputs = layer.get_outputs()

        return self.output_layer.get_outputs()
        # print(f'output after feeding forward: {self.output_layer.get_outputs()}')


    def back_propogate(self, correct_outputs):
        # calculating error term for output layer
        for i, output_layer_perceptron in enumerate(self.output_layer.perceptrons):
            o = output_layer_perceptron.output
            error_term = o * (1 - o) * (correct_outputs[i] - o)
            output_layer_perceptron.error_term = error_term
            self.output_layer.error_terms[i] = error_term

        # calculating error terms for every other layer
        for i in reversed(range(self.num_layers - 1)):
            layer = self.layers[i]
            next_layer = self.layers[i + 1]

            for j, perceptron in enumerate(layer.perceptrons):
                error_term = 0

                for k, next_layer_perceptron in enumerate(next_layer.perceptrons):
                    error_term += (next_layer.error_terms[k] * next_layer_perceptron.weights[j])

                o = perceptron.output
                error_term *= ((o) * (1 - o))
                perceptron.error_term = error_term
                layer.error_terms[j] = error_term


    def update_weights(self):
        for layer in self.layers:
            for i, perceptron in enumerate(layer.perceptrons):
                const_factor = perceptron.error_term * self.learning_rate
                for j in range(perceptron.num_inputs):
                    input = perceptron.inputs[j]
                    delta_w = const_factor * input
                    perceptron.weights[j] += delta_w

                perceptron.bias += const_factor

