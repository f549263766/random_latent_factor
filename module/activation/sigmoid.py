import numpy as np


def sigmoid(x):
    y = x.copy()
    y[x >= 0] = 1. / (1 + np.exp(-x[x >= 0]))
    y[x < 0] = np.exp(x[x < 0]) / (1 + np.exp(x[x < 0]))

    return y