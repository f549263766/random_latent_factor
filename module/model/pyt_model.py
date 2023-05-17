import numpy as np
from module.activation.sigmoid import sigmoid


class PtyModel:

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
        self.w = rnd.uniform(0, 1, (self.num_input_dim, self.num_hidden_dim))
        # Initialize the bias → [hidden_dim]
        self.bias = np.array([rnd.uniform(0, 0.4) for _ in range(num_hidden_dim)], dtype=float)
        # Initialize the beta → [hidden_dim, output_dim]
        self.beta = np.zeros([self.num_hidden_dim, self.num_output_dim], dtype=float)
        # Initialize the gamma → [hidden_dim, output_dim]
        self.gamma = np.zeros([self.num_hidden_dim, self.num_output_dim], dtype=float)
        # Initialize the hidden output
        self.h = None

    def fix(self, x, y, x_co, y_co):
        # beta-step
        h_matrix = np.asarray(self.activation(np.dot(x, self.w) + self.bias))
        h_matrix_inverse = np.linalg.inv(2e-5 + np.dot(h_matrix.T, h_matrix))
        h_matrix_output = np.dot(h_matrix.T, y)
        self.beta = np.dot(h_matrix_inverse, h_matrix_output)
        # gamma-step
        gamma_matrix = np.asarray(self.activation(np.dot(x, self.w) + self.bias))
        gamma_matrix_inverse = np.linalg.inv(2e-5 + np.dot(gamma_matrix.T, gamma_matrix))
        gamma_matrix_output = np.dot(gamma_matrix.T, y_co)
        self.gamma = np.dot(gamma_matrix_inverse, gamma_matrix_output)
        # h-step
        h_part1 = 2e-5 * (np.dot(y, self.beta.T) + np.dot(y_co, self.gamma.T))
        h_part2 = np.asarray(2e-5) + np.dot(self.beta, self.beta.T)
        h_part3 = np.asarray(2e-5) + np.dot(self.gamma, self.gamma.T)
        try:
            h_part4 = np.linalg.inv(h_part2 + h_part3)
        except np.linalg.LinAlgError:
            h_part4 = np.linalg.pinv(h_part2 + h_part3)
        h_part5 = np.dot(h_part1, h_part4)
        self.h = h_part5

        return self.h, self.beta, self.gamma

    def __call__(self, x):
        output = np.dot(self.h, 0.5 * self.beta + 0.5 * self.gamma)

        return output
