import os
import random
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path as osp
from utlis.logger import get_root_logger
from utlis.tools import outliers_replaced_with_mean_values
from scipy.interpolate import make_interp_spline

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

markers = ["o", "v", "1", "s", "*", "D", "+"]
# xy_range_dict = {
#     "D1-MAE": [0.05, 0.5], "D1-RMSE": [0.05, 0.8],
#     "D2-MAE": [0, 2], "D2-RMSE": [0, 1.8],
#     "D3-MAE": [0, 2], "D3-RMSE": [0, 2],
#     "D4-MAE": [0, 2], "D4-RMSE": [0, 3],
#     "D5-MAE": [0.0, 1.3], "D5-RMSE": [0.0, 1.6],
#     "D6-MAE": [-0.1, 1], "D6-RMSE": [-0.1, 2]
# }

val_c_dict = {str(rfc): f"2$^{{{c}}}$" for rfc, c in zip(np.logspace(-10, 11, base=2., num=22), range(10, -12, -1))}


def draw_picture(data_path):
    # get each result excel tables
    excel_lists = [
        osp.join(data_path, 'alpha_0.5', excel_res)
        for excel_res in os.listdir(osp.join(data_path, 'alpha_0.5')) if re.search(r"alpha", excel_res)
    ]
    # sort list
    excel_lists.sort(key=lambda x: float(factor_re.search(osp.basename(x))[0].
                                         replace("pty_model_c_", "").replace("_alpha", "")))
    # setup x-label
    x_label = ["%.0e" % float(factor_re.search(osp.basename(label))[0].
                              replace("pty_model_c_", "").replace("_alpha", "")) for label in excel_lists]
    x_index = [x.replace("+0", "").replace("+00", "").replace("-0", "-").replace("+", "").replace("e", "$^{") + "}$"
               for x in x_label]
    # draw picture
    fig1, fig2 = plt.figure(), plt.figure()
    ax1, ax2 = fig1.add_subplot(1, 1, 1), fig2.add_subplot(1, 1, 1)
    for idx, hidden in enumerate([5, 20, 40, 80, 120, 160, 200]):
        # setup indicators
        draw_data = []
        # parsing data
        for excel_path in excel_lists:
            result = pd.read_excel(excel_path, sheet_name="Sheet1")
            draw_data.extend(result.iloc[result[result["hidden_dim"] == hidden].index.to_list(), 1:].values.tolist())
        # draw each line
        ax1.plot(x_index, np.array(draw_data)[:, 0].tolist(), markers[idx], ms=4, linestyle='-')
        ax2.plot(x_index, np.array(draw_data)[:, 1].tolist(), markers[idx], ms=4, linestyle='-')
    # set up drawing
    ax1.set_title(f"{osp.basename(data_path).upper()}")
    ax1.set_xlabel('Value of C', fontsize=18)
    ax1.set_ylabel('RMSE', fontsize=18)
    ax1.legend(loc='upper left', labels=["f=5", "f=20", "f=40", "f=80", "f=120", "f=160", "f=200"])
    ax1.xaxis.set_ticks(x_index[::2])
    # ax1.set_ylim(xy_range_dict[f"{osp.basename(data_path).upper()}-RMSE"])
    ax2.set_title(f"{osp.basename(data_path).upper()}")
    ax2.set_xlabel('Value of C', fontsize=18)
    ax2.set_ylabel('MAE', fontsize=18)
    ax2.legend(loc='upper left', labels=["f=5", "f=20", "f=40", "f=80", "f=120", "f=160", "f=200"])
    ax2.xaxis.set_ticks(x_index[::2])
    # ax2.set_ylim(xy_range_dict[f"{osp.basename(data_path).upper()}-MAE"])
    fig1.show()
    fig2.show()
    # save fig
    fig1.savefig(f'{osp.join(output_fig_path, f"{osp.basename(data_path).upper()}_RMSE_inference_c.pdf")}')
    fig2.savefig(f'{osp.join(output_fig_path, f"{osp.basename(data_path).upper()}_MAE_inference_c.pdf")}')


def draw_inference_of_c_in_f_200(data_path, alpha=0.5, f_dim=150):
    # get each result excel tables
    excel_lists = [
        osp.join(data_path, f'alpha_{alpha}', excel_res)
        for excel_res in os.listdir(osp.join(data_path, f'alpha_{alpha}')) if re.search(r"alpha", excel_res)
    ]
    # sort list
    excel_lists.sort(key=lambda x: float(factor_re.search(osp.basename(x))[0].
                                         replace("pty_model_c_", "").replace("_alpha", "")), reverse=True)
    # setup x-label
    x_label = [v for k, v in val_c_dict.items()]
    # draw picture
    rmse_fig, mae_fig = plt.figure(), plt.figure()
    rmse_ax, mae_ax = rmse_fig.add_subplot(1, 1, 1), mae_fig.add_subplot(1, 1, 1)
    # setup indicators
    draw_data = []
    # parsing data
    for excel_path in excel_lists:
        result = pd.read_excel(excel_path, sheet_name="Sheet1")
        draw_data.extend(result.iloc[result[result["hidden_dim"] == f_dim].index.to_list(), 1:].values.tolist())
    # draw each line
    random_idx = random.randint(0, len(markers) - 1)
    # curve smoothing
    # x_label_ = [float(k) for k, v in val_c_dict.items()]
    # model0 = make_interp_spline(np.array(x_label_), np.array(draw_data)[:, 0])
    # model1 = make_interp_spline(np.array(x_label_), np.array(draw_data)[:, 1])
    # ys0 = model0(np.logspace(-10, 11, base=2., num=100))
    # rmse_ax.plot(np.logspace(-10, 11, base=2., num=100), ys0, markers[random_idx], ms=4, linestyle='-')
    # mae_ax.plot(np.logspace(-10, 11, base=2., num=100), ys0, markers[random_idx], ms=4, linestyle='-')
    # replace outliers with_mean_values
    process_rmse_data = outliers_replaced_with_mean_values(np.array(draw_data)[:, 0].tolist())
    process_mae_data = outliers_replaced_with_mean_values(np.array(draw_data)[:, 1].tolist())
    rmse_ax.plot(x_label[::-1], process_rmse_data, markers[random_idx], ms=4, linestyle='-')
    mae_ax.plot(x_label[::-1], process_mae_data, markers[random_idx], ms=4, linestyle='-')
    # set up drawing
    rmse_ax.set_title(f"{osp.basename(data_path).upper()}")
    rmse_ax.set_xlabel('Value of C', fontsize=18)
    rmse_ax.set_ylabel('RMSE', fontsize=18)
    rmse_ax.legend(loc='upper left', labels=[f"f={f_dim}"])
    rmse_ax.xaxis.set_ticks(x_label[::-2])
    # ax1.set_ylim(xy_range_dict[f"{osp.basename(data_path).upper()}-RMSE"])
    mae_ax.set_title(f"{osp.basename(data_path).upper()}")
    mae_ax.set_xlabel('Value of C', fontsize=18)
    mae_ax.set_ylabel('MAE', fontsize=18)
    mae_ax.legend(loc='upper left', labels=["f=5", "f=20", "f=40", "f=80", "f=120", "f=160", "f=200"])
    mae_ax.xaxis.set_ticks(x_label[::-2])
    # ax2.set_ylim(xy_range_dict[f"{osp.basename(data_path).upper()}-MAE"])
    rmse_fig.show()
    mae_fig.show()
    # save fig
    output_path = output_fig_path + f"_alpha{alpha}_d{f_dim}"
    if not osp.exists(output_path):
        os.makedirs(output_path)
    rmse_fig.savefig(f'{osp.join(output_path, f"{osp.basename(data_path).upper()}_RMSE_inference_c.pdf")}')
    mae_fig.savefig(f'{osp.join(output_path, f"{osp.basename(data_path).upper()}_MAE_inference_c.pdf")}')


def main():
    dataset_lists = [osp.join(root_path, sub_dir) for sub_dir in os.listdir(root_path)
                     if osp.isdir(osp.join(root_path, sub_dir))]

    for sub_dir in dataset_lists:
        logger.info(f"Start drawing picture from {sub_dir} datasets")
        draw_inference_of_c_in_f_200(sub_dir, 0.5, 60)
        # draw_picture(sub_dir)


if __name__ == '__main__':
    output_fig_path = "D:/QYZ/Code/random_latent_factor/output/figs/inference_c"
    timestamp = str(time.strftime('%Y%m%d%M%S', time.localtime()))
    logger_path = "D:/QYZ/Code/random_latent_factor/logs"
    if not osp.exists(logger_path):
        os.makedirs(logger_path)
    if not osp.exists(output_fig_path):
        os.makedirs(output_fig_path)
    logger = get_root_logger(log_file=osp.join(logger_path, f"out_{timestamp}.txt"))
    root_path = "D:/QYZ/Code/random_latent_factor/output/excel/pty_model"
    factor_re = re.compile(r"pty_model_c_(.+)_alpha")
    main()
