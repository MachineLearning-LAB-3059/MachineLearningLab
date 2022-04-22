import csv
from collections import defaultdict

# reading the dataset into list of lists
with open('input_2.csv') as dataset_csv_file:
    rows = csv.reader(dataset_csv_file)
    dataset = list(rows)
    test_input = dataset.pop()

nrows = len(dataset)
ncols = len(dataset[0])

# # converting into list of lists
# for i, row in enumerate(dataset):
#     dataset[i] = list(row)
#
# # type casting each item into a string
# for row in dataset:
#     for i, v in enumerate(row):
#         row[i] = str(v)

# finding the possible outputs
possible_outcomes = set()
for row in dataset:
    possible_outcomes.add(row[-1])
possible_outcomes = list(possible_outcomes)

# finding counts of each value in every column
single_counts = [defaultdict(int) for i in range(ncols)]

for col in range(ncols):
    cur_dict = single_counts[col]
    for row in range(nrows):
        cur_value = dataset[row][col]
        cur_dict[cur_value] += 1

# finding pairwise counts of each (value, target) pair
pairwise_counts = [defaultdict(int) for i in range(ncols - 1)]
for col in range(ncols - 1):
    cur_dict = pairwise_counts[col]
    for row in range(nrows):
        target = dataset[row][-1]
        pair = (dataset[row][col], target)
        cur_dict[pair] += 1

# for d in single_counts:
#     for v in d.items():
#         print(v)
#
# for d in pairwise_counts:
#     for v in d.items():
#         print(v)

# calculating the probability of each outcome for the test input
for outcome in possible_outcomes:
    numerator = denominator = 1
    print(test_input)
    for i, value in enumerate(test_input):
        num_value_and_outcome = pairwise_counts[i][(value, outcome)]
        num_value_and_outcome = max(1, num_value_and_outcome)
        num_outcome = single_counts[-1][outcome]
        p_value_given_outcome = num_value_and_outcome / num_outcome
        numerator *= p_value_given_outcome

        num_value = single_counts[i][value]
        tot_count = nrows
        p_value = num_value / tot_count
        denominator *= p_value

    num_outcome = single_counts[-1][outcome]
    tot_count = nrows
    p_outcome = num_outcome / tot_count
    numerator *= p_outcome

    print(numerator)
    print(denominator)

    probability = numerator / denominator
    print(f'probability for outcome: {outcome} = {probability}')


