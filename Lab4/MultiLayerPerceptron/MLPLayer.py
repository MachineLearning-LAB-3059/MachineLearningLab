from Perceptron import Perceptron

class MLPLayer:
    def __init__(self, num_perceptrons, num_inputs, activation_function, learning_rate):
        self.num_perceptrons = num_perceptrons
        self.perceptrons = [Perceptron(num_inputs, activation_function, learning_rate) for i in range(num_perceptrons)]
        self.outputs = [0 for i in range(num_perceptrons)]
        self.error_terms = [0 for i in range(num_perceptrons)]

    def update_outputs(self, inputs):
        for i in range(self.num_perceptrons):
            self.outputs[i] = self.perceptrons[i].compute(inputs)

    def get_outputs(self):
        return self.outputs
