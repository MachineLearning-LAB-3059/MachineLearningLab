import matplotlib.pyplot as plotter
from Perceptron import Perceptron
from LearningAlgorithms import perceptron_training_rule, gradient_descent, stochastic_gradient_descent
from ActivationFunctions import step, sigmoid, identity
from LabeledInput import LabeledInput


def train_perceptron_2d(file_name):
    labeled_input = LabeledInput(file_name)
    perceptron = Perceptron(sigmoid, labeled_input, 0.5)
    perceptron.train(stochastic_gradient_descent)
    print(f'\n weights after training: \n{perceptron.get_weights()}')
    print(f'\n calculated outputs: {perceptron.get_output_for_all_training_examples()}')

    lim = 3
    plotter.xlim(-lim, lim)
    plotter.ylim(-lim, lim)
    plotter.grid()
    labeled_input.plot_2d(plotter)

    perceptron.plot_2d(plotter, lim)

    plotter.show()


def train_perceptron_1d(file_name):
    labeled_input = LabeledInput(file_name)
    perceptron = Perceptron(identity, labeled_input, 0.005)
    perceptron.train(gradient_descent)
    print(f'\n weights after training: \n{perceptron.get_weights()}')
    print(f'\n calculated outputs: {perceptron.get_output_for_all_training_examples()}')

    x_lim = 6
    y_lo = 30000
    y_hi = 70000
    plotter.xlim(0, x_lim)
    plotter.ylim(y_lo, y_hi)
    plotter.grid()
    labeled_input.plot_1d(plotter)

    perceptron.plot_1d(plotter, x_lim)

    plotter.show()


if __name__ == '__main__':
    train_perceptron_1d('C:\\Users\\shiva\\PycharmProjects\\MachineLearningLab\\Lab4\\input_years_to_salary.csv')
