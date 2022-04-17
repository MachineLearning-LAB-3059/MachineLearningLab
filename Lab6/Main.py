from Lab6.LabeledInput import LabeledInput
from Lab6.RBF import RBF

labeled_input = LabeledInput('data.csv')
rbf = RBF(3, labeled_input)
rbf.train()
print(rbf.compute_outputs())