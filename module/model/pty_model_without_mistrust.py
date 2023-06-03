import numpy as np
from module.activation.sigmoid import sigmoid


class PtyModelWithoutMistrust:

    def __init__(self,
                 num_input_dim,
                 num_hidden_dim,
                 num_output_dim,
                 activation='sigmoid',
                 regularization_factor_c=10):
        self.num_input_dim = num_input_dim
        self.num_hidden_dim = num_hidden_dim
        self.num_output_dim = num_output_dim
        self.c = regularization_factor_c
        self.activation = sigmoid
        rnd = np.random.RandomState()

        # Initialize the w → [input_dim, hidden_dim]
        self.w = rnd.uniform(-2, 2, (self.num_input_dim, self.num_hidden_dim))
        # Initialize the bias → [hidden_dim]
        self.bias = np.array([rnd.uniform(-0.2, 0.2) for _ in range(num_hidden_dim)], dtype=float)
        # Initialize the beta → [hidden_dim, output_dim]
        self.beta = np.zeros([self.num_hidden_dim, self.num_output_dim], dtype=float)
        # Initialize the gamma → [hidden_dim, output_dim]
        self.gamma = np.zeros([self.num_hidden_dim, self.num_output_dim], dtype=float)
        # Initialize the mu → [hidden_dim, output_dim]
        self.mu = np.zeros([self.num_hidden_dim, self.num_output_dim], dtype=float)
        # Initialize the hidden output
        self.h = None

    def fix(self, x, y, x_co, y_co, *args):
        # beta-step
        h_matrix = np.asarray(self.activation(np.dot(x, self.w) + self.bias))
        h_matrix_inverse = np.dot(np.linalg.inv(np.dot(h_matrix.T, h_matrix) + np.asarray(1e-5)), h_matrix.T)
        self.beta = np.dot(h_matrix_inverse, y)
        # gamma-step
        self.gamma = np.dot(h_matrix_inverse, y_co)
        # h-step
        self.h = np.dot((np.dot(x, self.beta.T) + np.dot(x_co, self.gamma.T)),
                        np.linalg.inv(np.dot(self.beta, self.beta.T) + np.dot(self.gamma, self.gamma.T)
                                      + np.asarray(1e-5)))

        return self.h, self.beta, self.gamma

    def __call__(self, x):
        output = np.dot(self.h, self.beta)

        return output
