from module.activation.sigmoid import sigmoid
from module.activation.pty_exp import pty_exp


class ActivationSelect:

    def __init__(self,
                 activation_name='sigmoid'):
        assert isinstance(activation_name, str) and activation_name in ['sigmoid', 'exp']

        self.activation_name = activation_name

    @staticmethod
    def sigmoid_function():
        return sigmoid

    @staticmethod
    def exp_function():
        return pty_exp
