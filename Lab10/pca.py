import numpy as np

X = [[2.58, 2.16, 3.27], [4.46, 6.22, 3.52]]

def subtract_means(a):
    mean = np.mean(a)
    print(f'mean: {mean}')
    for i, v in enumerate(a):
        a[i] = v - mean


subtract_means(X[0])
subtract_means(X[1])

print('input after subtracting mean: ')
B = list(zip(X[0], X[1]))
for v in B:
    print(v)

B_B_Transpose = np.dot(X, B)
print()
print(f'BBt: {B_B_Transpose}')
print()

for row in B_B_Transpose:
    for i, v in enumerate(row):
        row[i] = v / (len(B) - 1)

print(f'Covariance matrix')
print(f'BBt: {B_B_Transpose}')
print()

eigen_values, eigen_vectors = np.linalg.eig(B_B_Transpose)

print(f'eigen values: {eigen_values}')
print(f'eigen_vectors: {eigen_vectors}')

