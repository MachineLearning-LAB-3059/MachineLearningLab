import matplotlib.pyplot as plotter
from Perceptron import Perceptron
from LearningAlgorithms import perceptron_training_rule, gradient_descent, stochastic_gradient_descent
from ActivationFunctions import step, sigmoid, identity
from LabeledInput import LabeledInput
from MLP import MLP

if __name__ == '__main__':
    mlp = MLP(num_layers=3,
              layer_dimension=1,
              input_dimension=2,
              output_dimension=1,
              activation_function=sigmoid,
              learning_rate=0.5)

    for i in range(20000):
        mlp.feed_forward([1, 1])
        mlp.back_propogate([0])
        mlp.update_weights()
        mlp.feed_forward([0, 0])
        mlp.back_propogate([0])
        mlp.update_weights()
        mlp.feed_forward([0, 1])
        mlp.back_propogate([1])
        mlp.update_weights()
        mlp.feed_forward([1, 0])
        mlp.back_propogate([1])
        mlp.update_weights()

    print(f'----------------------------------')
    print(mlp.feed_forward([0, 0]))
    print(mlp.feed_forward([0, 1]))
    print(mlp.feed_forward([1, 0]))
    print(mlp.feed_forward([1, 1]))




