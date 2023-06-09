import numpy as np
from module.activation.activation_select import ActivationSelect


class PtyModel:

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
        self.alpha = regulating_factors_alpha

        rnd = np.random.RandomState()
        # Initialize the w → [input_dim, hidden_dim]
        self.w = rnd.uniform(-1, 1, (self.num_input_dim, self.num_hidden_dim))
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

    def fix(self, x, y, x_co, y_co, x_mistrust, y_mistrust, *args):
        # beta-step
        h_matrix = np.asarray(self.activation(np.dot(x, self.w) + self.bias))
        h_matrix_inverse = np.dot(np.linalg.inv(np.dot(h_matrix.T, h_matrix) +
                                                self.c * np.eye(int(h_matrix.shape[1]))), h_matrix.T)
        self.beta = np.dot(h_matrix_inverse, y)
        # gamma-step
        self.gamma = np.dot(h_matrix_inverse, y_co)
        # mu-step
        self.mu = np.dot(h_matrix_inverse, y_mistrust)
        # h-step
        self.h = np.dot((np.dot(x, self.beta.T) + self.alpha * np.dot(x_co, self.gamma.T) -
                         self.alpha * np.dot(x_mistrust, self.mu.T)),
                        np.linalg.inv(np.dot(self.beta, self.beta.T) +
                                      self.alpha * np.dot(self.gamma, self.gamma.T) -
                                      self.alpha * np.dot(self.mu, self.mu.T) + np.asarray(self.c)))

        return self.h, self.beta, self.gamma

    def __call__(self, x):
        output = np.dot(self.h, self.beta)

        return output
