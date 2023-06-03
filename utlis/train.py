import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn import metrics
from module.model.elm import Elm
from module.model.rlfn import Rlfn
from module.model.pyt_model import PtyModel
from module.model.pty_model_without_mistrust import PtyModelWithoutMistrust
from module.datasets.D1_processing_data import Pty1Rating
from module.datasets.D2_processing_data import Pty2Rating
from module.datasets.D3_processing_data import Pty3Rating
from module.datasets.D4_processing_data import Pty4Rating
from module.datasets.D5_processing_data import Pty5Rating
from module.datasets.D6_processing_data import Pty6Rating
from utlis.logger import get_root_logger


def command_line():
    # Setting command line parameters
    parser = argparse.ArgumentParser("random learning")
    parser.add_argument("-m", "--model", type=str, default="elm",
                        help="Select the model for training with random learning method."
                             "Default to elm, and optional are elm.")
    parser.add_argument("-lb", "--hidden_dim_lb", type=int, default=5,
                        help="The minimum number of neurons in the hidden layer of SLFN."
                             "Default to 5.")
    parser.add_argument("-ub", "--hidden_dim_ub", type=int, default=205,
                        help="The maximum number of neurons in the hidden layer of SLFN."
                             "Default to 100.")
    parser.add_argument("-interval", "--incremental_interval", type=int, default=5,
                        help="Incremental interval of the number of neurons in the hidden layer."
                             "Default to 1.")
    parser.add_argument("-d", "--dataset", type=str, default="RegressionDataset",
                        help="Select the data set to be used this time. "
                             "Default to RegressionDataset.")
    parser.add_argument("--loss", type=str, default="mse",
                        help="Select the loss function used for training."
                             "Default to mse, and optional are mse, mae and rmae.")

    return parser.parse_args()


def train(cfg):
    # data_set = Pty1Rating(root="D:/QYZ/Code/random_latent_factor/dataset/",
    #                       user_rating_data_path="D1-movie-ratings.xlsx",
    #                       user_trust_data_path="D1-trust-data.xlsx",
    #                       logger=logger)
    data_set = Pty2Rating(root="D:/QYZ/Code/random_latent_factor/dataset/",
                          user_rating_data_path="D2-movie-ratings.xlsx",
                          user_trust_data_path="D2-trust-data.xlsx",
                          logger=logger)
    # data_set = Pty3Rating(root="D:/QYZ/Code/random_latent_factor/dataset/",
    #                       user_rating_data_path="D3-movie-ratings.xlsx",
    #                       user_trust_data_path="D3-trust-data.xlsx",
    #                       logger=logger)
    # data_set = Pty4Rating(root="D:/QYZ/Code/random_latent_factor/dataset/",
    #                       user_rating_data_path="D4-movie-ratings.xlsx",
    #                       user_trust_data_path="D4-trust-data.xlsx",
    #                       logger=logger)
    # data_set = Pty5Rating(root="D:/QYZ/Code/random_latent_factor/dataset/",
    #                       user_rating_data_path="D5-movie-ratings.xlsx",
    #                       user_trust_data_path="D5-trust-data.xlsx",
    #                       logger=logger)
    # data_set = Pty6Rating(root="D:/QYZ/Code/random_latent_factor/dataset/",
    #                       user_rating_data_path="D6-movie-ratings.xlsx",
    #                       user_trust_data_path="D6-trust-data.xlsx",
    #                       logger=logger)

    x, y, x_trust, y_trust, x_mistrust, y_mistrust = data_set.x, data_set.y, data_set.x_trust, data_set.y_trust, \
                                                     data_set.x_mistrust, data_set.y_mistrust
    print(x.shape, y.shape)
    # plt.plot(x_train, y_train, 'or')

    for num_hidden_dim in range(cfg.hidden_dim_lb, cfg.hidden_dim_ub, cfg.incremental_interval):
        try:
            # slfn = Elm(num_input_dim=x.shape[1],
            #            num_hidden_dim=num_hidden_dim,
            #            num_output_dim=y.shape[1])
            # slfn = Rlfn(num_input_dim=x.shape[1],
            #             num_hidden_dim=num_hidden_dim,
            #             num_output_dim=y.shape[1])
            slfn = PtyModel(num_input_dim=x.shape[1],
                            num_hidden_dim=num_hidden_dim,
                            num_output_dim=y.shape[1])
            # slfn = PtyModelWithoutMistrust(num_input_dim=x.shape[1],
            #                                num_hidden_dim=num_hidden_dim,
            #                                num_output_dim=y.shape[1])

            _ = slfn.fix(x, y, x_trust, y_trust, x_mistrust, y_mistrust)
        except np.linalg.LinAlgError:
            continue
        predict = slfn(x)
        rmse = metrics.mean_squared_error(y, predict) ** 0.5
        mae = metrics.mean_absolute_error(y, predict)
        print(f"number of layer {num_hidden_dim}, rmse: {rmse}, mae: {mae}")

    #     plt.plot(x_train, predict)
    # plt.legend([['original'], ['hidden_5'], ['hidden_10'], ['hidden_15'], ['hidden_20'], ['hidden_25']],
    #            loc='upper right')
    # plt.ylim((-2, 30))
    # plt.show()


if __name__ == '__main__':
    args = command_line()
    logger = get_root_logger()
    train(args)
