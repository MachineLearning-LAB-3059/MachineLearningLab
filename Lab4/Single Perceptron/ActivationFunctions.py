import numpy as np


def identity(x):
    return x


def step(x):
    return 1 if x > 0 else 0


def sigmoid(x):
    z = np.exp(-x)
    sig = 1 / (1 + z)
    return sig
