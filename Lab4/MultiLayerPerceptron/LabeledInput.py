import numpy as np
import pandas as pd


class LabeledInput:
    def __init__(self, file_name):
        # reading the input csv file
        dataframe = pd.read_csv(file_name)
        # all the columns except the last is taken as input
        input_df = dataframe.iloc[:, :-1]
        # the last column is taken as output
        output_df = dataframe.iloc[:, -1:]
        # reading the input vectors in column major
        self.input_vectors_column_major = np.transpose(input_df.values.tolist())
        # reading the dimensions of input'
        self.number_of_training_examples = len(self.input_vectors_column_major[0])
        # 1 is added because we are going to add a row of ones (1 to each input vector)
        self.input_vector_size = len(self.input_vectors_column_major) + 1
        # adding a one to each input vector (each column)
        ones = [1 for i in range(self.number_of_training_examples)]
        self.input_vectors_column_major = np.vstack([ones, self.input_vectors_column_major])
        # getting the row major representation
        self.input_vectors_row_major = self.input_vectors_column_major.T

        # getting the outputs
        self.output_vector = np.array(output_df.values.tolist()).flatten()

    def get_number_of_training_examples(self):
        return self.number_of_training_examples

    def get_input_vector_size(self):
        return self.input_vector_size

    def get_all_input_vectors(self):
        return self.input_vectors_column_major

    def get_input_vector(self, k):
        res = np.array(self.input_vectors_row_major[k])[np.newaxis]
        return res.T

    def get_all_outputs(self):
        return self.output_vector

    def get_output(self, k):
        return self.output_vector[k]

    def plot_2d(self, plotter):
        for i in range(self.number_of_training_examples):
            x = [self.get_input_vector(i)[1]]
            y = [self.get_input_vector(i)[2]]

            output = self.get_output(i)

            point_color = 'green' if output == 1 else 'red'
            plotter.plot(x, y, marker="o", markersize=5, markerfacecolor=point_color, markeredgecolor='black')

    def plot_1d(self, plotter):
        for i in range(self.number_of_training_examples):
            x = [self.get_input_vector(i)[1]]
            y = [self.get_output(i)]

            point_color = 'green'
            plotter.plot(x, y, marker="o", markersize=5, markerfacecolor=point_color, markeredgecolor='black')




