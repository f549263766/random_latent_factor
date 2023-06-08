import os
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path as osp
from utlis.logger import get_root_logger

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


def main():
    dataset_lists = [osp.join(root_path, sub_dir) for sub_dir in os.listdir(root_path)]
    logger.info(f"Start drawing picture from {dataset_lists} datasets")

    for sub_dir in dataset_lists:
        draw_picture(sub_dir)


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
