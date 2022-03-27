import numpy as np
import pandas as pd

# reading the csv data into a dataframe
data = pd.read_csv('input.csv')

# for the concepts, take all the rows and ignore the last column
concepts = np.array(data.iloc[:, 0:-1])
print("\nInstances are:\n", concepts)

# the target values are taken from the last column of every row
target = np.array(data.iloc[:, -1])
print("\nTarget Values are: ", target)


def learn(concepts, target):

    # The specific boundary is initially just the first concept
    specific_h = concepts[0].copy()
    print("\nInitialization of specific_h and genearal_h")
    print("\nSpecific Boundary: ", specific_h)

    # The general boundary is initialized to [? ? ? ...] for how many ever attributes there are
    general_h = [["?" for i in range(len(specific_h))] for i in range(len(specific_h))]
    print("\nGeneric Boundary: ", general_h)

    # Getting each concept
    for i, h in enumerate(concepts):
        print("\nInstance", i + 1, "is ", h)

        # Positive instance
        if target[i] == "yes":
            print("Instance is Positive ")
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'

        # Negative instance
        if target[i] == "no":
            print("Instance is Negative ")
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'

        print("Specific Bundary after ", i + 1, "Instance is ", specific_h)
        print("Generic Boundary after ", i + 1, "Instance is ", general_h)
        print("\n")

    indices = [i for i, val in enumerate(general_h) if val == ['?', '?', '?', '?', '?', '?']]
    for i in indices:
        general_h.remove(['?', '?', '?', '?', '?', '?'])
    return specific_h, general_h


s_final, g_final = learn(concepts, target)

print("Final Specific_h: ", s_final, sep="\n")
print("Final General_h: ", g_final, sep="\n")