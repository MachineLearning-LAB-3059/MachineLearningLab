import numpy as np
from Perceptron import Perceptron


def perceptron_training_rule(perceptron):
    for iteration in range(Perceptron.max_iterations):
        tot_error = 0
        for i in range(perceptron.labelled_input.number_of_training_examples):
            correct_output = perceptron.labelled_input.get_output(i)
            calculated_output = perceptron.get_output_for_ith_training_example(i)
            difference = correct_output - calculated_output
            tot_error += abs(difference)

            const_factor = perceptron.learning_rate * difference
            delta_w = const_factor * perceptron.labelled_input.get_input_vector(i)

            perceptron.weights = perceptron.weights + delta_w
        if tot_error == 0:
            print(f'training completed after {iteration} iterations...')
            return

    print(f'did not converge after {Perceptron.max_iterations} iterations')


def gradient_descent(perceptron):
    for iteration in range(perceptron.max_iterations):
        X = perceptron.labelled_input.get_all_input_vectors()
        Y = perceptron.labelled_input.get_all_outputs()
        O = perceptron.get_output_for_all_training_examples()
        D = np.array(Y) - np.array(O)
        D_transpose = (np.array(D)[np.newaxis]).T
        delta_w = perceptron.learning_rate * (np.dot(X, D_transpose))
        # if(0 <= iteration <= 5):
        #     print(f'X: {X}')
        #     print(f'Y: {Y}')
        #     print(f'O: {O}')
        #     print(f'D: {D}')
        #     print(f'D_T: {D_transpose}')
        #     print(f'delta_w: {delta_w}')
        perceptron.weights = perceptron.weights + delta_w

    print(f'{perceptron.max_iterations} iterations completed')

def stochastic_gradient_descent(perceptron):
    for iteration in range(Perceptron.max_iterations):
        delta_w = np.array([[0] for i in range(perceptron.labelled_input.get_input_vector_size())])
        for i in range(perceptron.labelled_input.number_of_training_examples):
            correct_output = perceptron.labelled_input.get_output(i)
            calculated_output = perceptron.get_output_for_ith_training_example(i)
            difference = correct_output - calculated_output

            const_factor = perceptron.learning_rate * difference
            delta_w = delta_w + (const_factor * perceptron.labelled_input.get_input_vector(i))

        perceptron.weights = perceptron.weights + delta_w

