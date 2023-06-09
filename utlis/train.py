import argparse
import os.path as osp
import numpy as np
import time
import os
from sklearn import metrics
from module.datasets.data_select import DataSelect
from module.model.model_select import ModelSelect
from utlis.logger import get_root_logger
from utlis.tools import save_result_with_excel, save_best_result


def command_line():
    # Setting command line parameters
    parser = argparse.ArgumentParser("random learning")
    parser.add_argument("-m", "--model", type=str, default="pty_model",
                        help="Select the model for training with random learning method."
                             "Default to elm, and optional are elm, rlfn, pty_model and pty_model_without_mistrust.")
    parser.add_argument("-lb", "--hidden_dim_lb", type=int, default=5,
                        help="The minimum number of neurons in the hidden layer of SLFN."
                             "Default to 5.")
    parser.add_argument("-ub", "--hidden_dim_ub", type=int, default=205,
                        help="The maximum number of neurons in the hidden layer of SLFN."
                             "Default to 100.")
    parser.add_argument("-interval", "--incremental_interval", type=int, default=5,
                        help="Incremental interval of the number of neurons in the hidden layer."
                             "Default to 1.")
    parser.add_argument("-d", "--dataset", type=str, default="d1",
                        help="Select the data set to be used this time."
                             "Default to d1, and optional are [d1, d2, d3, d4, d5, d6].")
    parser.add_argument("-ac", "--activation", type=str, default="sigmoid",
                        help="Select the activation function to be used this time."
                             "Default to sigmoid, and optional are sigmoid and exp.")
    parser.add_argument("-rfc", "--regularization_factor_c", type=float, default=2e-5,
                        help="Setup the regularization factor c of model between 2e-10 to 2e10 with multiples of 10."
                             "Default to 2e-5.")
    parser.add_argument("-alpha", "--regulating_factors_alpha", type=float, default=0.5,
                        help="Setup the regularization factor alpha of model between 0 to 1 with 0.1 stride."
                             "Default to 2e-5.")

    return parser.parse_args()


def train(cfg, items):
    # load target dataset
    data_select = DataSelect(
        data_name=args.dataset,
        data_root="D:/QYZ/Code/random_latent_factor/dataset/",
        logger=logger
    )
    data_set = getattr(data_select, f"pty{args.dataset[-1]}rating")()

    # processing dataset
    x, y, x_trust, y_trust, x_mistrust, y_mistrust = \
        data_set.x, data_set.y, data_set.x_trust, data_set.y_trust, data_set.x_mistrust, data_set.y_mistrust
    logger.info(f"Loading {args.dataset.upper()} dataset and the shape of dataset is: {x.shape}")

    # setting up variant
    result_list = []

    logger.info(f"Loading {args.model.upper()} model")

    # parameter training
    for num_hidden_dim in range(cfg.hidden_dim_lb, cfg.hidden_dim_ub, cfg.incremental_interval):
        try:
            model = ModelSelect(num_input_dim=x.shape[1],
                                num_hidden_dim=num_hidden_dim,
                                num_output_dim=y.shape[1],
                                activation=args.activation,
                                regularization_factor_c=args.regularization_factor_c,
                                regulating_factors_alpha=args.regulating_factors_alpha,
                                model_name=args.model)
            slfn = getattr(model, args.model + "_model" if "model" not in args.model else args.model)()
            _ = slfn.fix(x, y, x_trust, y_trust, x_mistrust, y_mistrust)
        except np.linalg.LinAlgError as e:
            logger.info(f"{e}")
            continue

        # model predict
        predict = slfn(x)
        # computing error of predict between trust
        rmse = metrics.mean_squared_error(y, predict) ** 0.5
        mae = metrics.mean_absolute_error(y, predict)
        # record the rmse and mae
        result_list.append([num_hidden_dim, rmse, mae])
        logger.info(f"number of layer {num_hidden_dim}, rmse: {rmse}, mae: {mae}")
        # update the best performance of model
        if items['hidden'] is not None and items['rmse'] > rmse:
            items['factor'] = args.regularization_factor_c
            items['hidden'] = num_hidden_dim
            items['rmse'], items['mae'] = rmse, mae
        else:
            items['factor'] = args.regularization_factor_c
            items['hidden'] = num_hidden_dim
            items['rmse'], items['mae'] = rmse, mae

    logger.info("save result with to D:/QYZ/Code/random_latent_factor/output\n")
    if len(result_list) >= 2:
        save_result_with_excel(
            result_list,
            f"{args.model}_c_{args.regularization_factor_c}_alpha_{args.regulating_factors_alpha}_{timestamp}.xlsx",
            f"D:/QYZ/Code/random_latent_factor/output/excel/{args.model}/{args.dataset}/"
            f"alpha_{args.regulating_factors_alpha}")


def main():
    # record iteration time
    tic = time.time()
    # select dataset
    for dataset in ["d1", "d2", "d3", "d4", "d5", "d6"]:
        logger.info(f"Start {args.model} model training in {args.dataset} dataset...")
        # setup alpha range
        for alpha in np.arange(0, 1.1, 0.1):
            alpha = 0.5 if args.model in ["elm", "rlfn"] else alpha
            # reset the parameter indicators
            best_result_items = {
                'alpha': alpha,
                'factor': None,
                'hidden': None,
                'rmse': None,
                'mae': None
            }
            # setup regularization factor range
            for rfc in np.logspace(-10, 10, base=2., num=21):
                rfc = 1e-5 if args.model in ["elm", "rlfn"] else rfc
                # reset hyper configs
                args.regularization_factor_c = rfc
                args.regulating_factors_alpha = round(alpha, 2)
                args.dataset = dataset
                # train model
                train(args, best_result_items)
                # elm and rlfn just one train
                if args.model in ["elm", "rlfn"]:
                    break

            logger.info(f"the best performance with {best_result_items['hidden']} hidden dim, "
                        f"regular factor {best_result_items['factor']} and alpha {alpha} of model, "
                        f"rmse: {best_result_items['rmse']}, mae: {best_result_items['mae']}")
            save_best_result(best_result_items,
                             f"best_{args.model}_result_{timestamp}.xlsx",
                             f"D:/QYZ/Code/random_latent_factor/output/excel/{args.model}/{args.dataset}/"
                             f"alpha_{args.regulating_factors_alpha}")
            if args.model in ["elm", "rlfn"]:
                break
        # show cost time of one iteration
        logger.info(f"Time cost in a dataset is :{time.time() - tic}\n")


if __name__ == '__main__':
    args = command_line()
    timestamp = str(time.strftime('%Y%m%d%M%S', time.localtime()))
    logger_path = "D:/QYZ/Code/random_latent_factor/logs"
    if not osp.exists(logger_path):
        os.mkdir(logger_path)
    logger = get_root_logger(log_file=osp.join(logger_path, f"out_{timestamp}.txt"))
    main()
