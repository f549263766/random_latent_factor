import numpy as np
from module.activation.activation_select import ActivationSelect


class Elm:
    def __init__(self,
                 num_input_dim,
                 num_hidden_dim,
                 num_output_dim,
                 activation='sigmoid',
                 regularization_factor_c=2e-5,
                 regulating_factors_alpha=0.5):
        self.num_input_dim = num_input_dim
        self.num_hidden_dim = num_hidden_dim
        self.num_output_dim = num_output_dim
        self.c = regularization_factor_c
        activation_select = ActivationSelect(activation)
        self.activation = getattr(activation_select, f"{activation}_function")()
        self.regulating_factors_alpha = regulating_factors_alpha

        rnd = np.random.RandomState()
        # Initialize the w → [input_dim, hidden_dim]
        self.w = rnd.uniform(-1, 1, (self.num_input_dim, self.num_hidden_dim))
        # Initialize the bias → [hidden_dim]
        self.bias = np.array([rnd.uniform(-0.4, 0.4) for _ in range(num_hidden_dim)], dtype=float)
        # Initialize the beta → [hidden_dim, output_dim]
        self.beta = np.zeros([self.num_hidden_dim, self.num_output_dim], dtype=float)

    def fix(self, x, y, *args):
        h_matrix = np.asarray(self.activation(np.dot(x, self.w) + self.bias))
        h_matrix_inverse = np.dot(np.linalg.inv(
                np.dot(h_matrix.T, h_matrix) + self.c * np.eye(int(h_matrix.shape[1]))), h_matrix.T)
        self.beta = np.dot(h_matrix_inverse, y)

        return self.beta

    def __call__(self, x):
        h_matrix = self.activation(np.dot(x, self.w) + self.bias)
        output = np.dot(h_matrix, self.beta)

        return output
