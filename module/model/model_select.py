from module.model.elm import Elm
from module.model.rlfn import Rlfn
from module.model.pyt_model import PtyModel
from module.model.pty_model_without_mistrust import PtyModelWithoutMistrust

model_list = ['elm', 'rlfn', 'pty_model', 'pty_model_without_mistrust']


class ModelSelect:

    def __init__(self,
                 num_input_dim,
                 num_hidden_dim,
                 num_output_dim,
                 activation='sigmoid',
                 regularization_factor_c=2e-5,
                 regulating_factors_alpha=0.5,
                 model_name="elm"):
        assert isinstance(num_input_dim, int) and isinstance(num_hidden_dim, int) and isinstance(num_output_dim, int)
        assert isinstance(activation, str) and activation in ["sigmoid", "exp"]
        assert isinstance(regularization_factor_c, float)
        assert isinstance(regulating_factors_alpha, float)
        assert isinstance(model_name, str)
        assert model_name in model_list, f"please input model name {model_list}"

        self.num_input_dim = num_input_dim
        self.num_hidden_dim = num_hidden_dim
        self.num_output_dim = num_output_dim
        self.activation = activation
        self.regularization_factor_c = regularization_factor_c
        self.regulating_factors_alpha = regulating_factors_alpha
        self.model_name = model_name

    def elm_model(self):
        elm = Elm(self.num_input_dim,
                  self.num_hidden_dim,
                  self.num_output_dim,
                  self.activation,
                  self.regularization_factor_c)

        return elm

    def rlfn_model(self):
        rlfn = Rlfn(self.num_input_dim,
                    self.num_hidden_dim,
                    self.num_output_dim,
                    self.activation,
                    self.regularization_factor_c)

        return rlfn

    def pty_model(self):
        pty = PtyModel(self.num_input_dim,
                       self.num_hidden_dim,
                       self.num_output_dim,
                       self.activation,
                       self.regularization_factor_c)

        return pty

    def pty_model_without_mistrust_model(self):
        pty_without_mistrust = PtyModelWithoutMistrust(self.num_input_dim,
                                                       self.num_hidden_dim,
                                                       self.num_output_dim,
                                                       self.activation,
                                                       self.regularization_factor_c)

        return pty_without_mistrust


if __name__ == '__main__':
    model = ModelSelect(1, 2, 4)
    print(getattr(model, 'elm_model')())
