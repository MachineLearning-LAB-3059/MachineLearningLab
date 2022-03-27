"""
    * First create a new Hypothesis which consists of all nulls
    * Read the csv line by line and for each line, create a hypothesis out of it and merge it with the existing hypothesis
"""
import csv


class Value:
    general = '?'
    null = '#'

    def __init__(self, value):
        self.value = value

    def is_null(self):
        return self.value == Value.null

    def is_general(self):
        return self.value == Value.general

    def is_specific(self):
        return not(self.is_null()) and not(self.general)

    def merge(self, other):
        # If any of them are general, result is general
        if self.is_general() or other.is_general():
            return Value(Value.general)
        # If one of them is null, then the result is the other
        if self.is_null():
            return Value(other.value)
        if other.is_null():
            return Value(self.value)
        # both are specific values, check if they are equal
        if self.value == other.value:
            return self

        # both are specific values and unequal, return general
        return Value(Value.general)

    def __repr__(self):
        return self.value

class Hypothesis:
    def __init__(self, row):
        self.list_of_values = []
        self.is_positive_hypothesis = row[-1] == '+'
        for i in range(len(row) - 1):
            self.list_of_values.append(Value(row[i]))

    def merge(self, other):
        new_list = []
        for i, v in enumerate(self.list_of_values):
            new_list.append(v.merge(other.list_of_values[i]))

        self.list_of_values = new_list

    @staticmethod
    def get_most_specific_hypothesis(num_columns):
        row = []
        for i in range(num_columns + 1):
            row.append(Value.null)
        return Hypothesis(row)

    def print(self):
        print(self.list_of_values)


if __name__ == '__main__':
    with open('input.csv') as file_obj:
        heading = next(file_obj)

        initial_hypothesis = Hypothesis.get_most_specific_hypothesis(4)
        print('initial hypothesis:')
        initial_hypothesis.print()
        print()

        reader_obj = csv.reader(file_obj)

        for row in reader_obj:
            new_hypothesis = Hypothesis(row)
            print('new hypothesis read')
            new_hypothesis.print()
            print()
            if not new_hypothesis.is_positive_hypothesis:
                print('Negative Hypothesis, Skipping')
                print("--------------------------------------------------")
                print()
                continue

            initial_hypothesis.merge(new_hypothesis)
            print('after merging with current hypothesis')
            initial_hypothesis.print()
            print("--------------------------------------------------")
            print()

        print('The final Hypothesis is')
        initial_hypothesis.print()


