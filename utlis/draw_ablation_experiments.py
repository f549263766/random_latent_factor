import os
import os.path as osp
import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
from utlis.logger import get_root_logger
from collections import defaultdict

methods_model_dict = {
    "M1": "NLF", "M2": "ELM", "M3": "RLF", "M4": "STE", "M5": "ENLF", "M6": "ISoRec", "M7": "pty_model"
}
model_method_dict = {v: k for k, v in methods_model_dict.items()}
methods_name_list = [v for _, v in methods_model_dict.items()]
data_name_list = ["D1", "D2", "D3", "D4", "D5", "D6"]
dim_name_list = ["f=5", "f=20", "f=40", "f=80", "f=120", "f=160", "f=200"]
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

markers = ["o", "v", "1", "s", "*", "D", "+"]


class DataProcess:

    def __init__(self, root_path):
        self.root_path = root_path

        self.processing()

    def processing(self):
        for method_name in methods_name_list:
            method_name = method_name.lower()
            if not osp.exists(osp.join(self.root_path, method_name, f"{method_name}_mae_summary.xlsx")) and \
                    not osp.exists(osp.join(self.root_path, method_name, f"{method_name}_rmse_summary.xlsx")):
                if method_name in ['elm', "rlf", "pty_model"]:
                    self.processing_current_method(method_name)
                else:
                    self.processing_historical_method(method_name)

    def processing_current_method(self, method_name):
        # setup root path for method
        data_root_path = osp.join(self.root_path, method_name)
        logger.info(f"Processing result data with {method_name} method...")
        # setup diff alpha sub dir
        sub_dir_lists = [osp.join(data_root_path, sub_dir) for sub_dir in os.listdir(data_root_path)]
        # create data summary table
        pd_data_summary_table_rmse = pd.DataFrame(columns=data_name_list, index=dim_name_list)
        pd_data_summary_table_mae = pd.DataFrame(columns=data_name_list, index=dim_name_list)
        # record best rmse and mae
        best_rmse, best_mae = [None] * 7, [None] * 7
        # traversal sub dirs
        for idx, sub_dir in enumerate(sub_dir_lists):
            # get excel data fro sub dir path
            excel_lists = [osp.join(sub_dir, alpha_dir, excel_data) for alpha_dir in os.listdir(sub_dir)
                           for excel_data in os.listdir(osp.join(sub_dir, alpha_dir))
                           if re.search(r"alpha", excel_data)]
            # read each excel data
            for excel_path in excel_lists:
                logger.info(f"Currently deal with excel {osp.basename(excel_path)}")
                sub_excel_data = pd.read_excel(excel_path, sheet_name="Sheet1")
                # rmse val and mae val
                rmse_mae_val_list = sub_excel_data.iloc[
                                    sub_excel_data[sub_excel_data["hidden_dim"].isin(
                                        [5, 20, 40, 80, 120, 160, 200])].index.to_list(), 1:]
                best_rmse = [round(v1, 4) if v1 is not None and v1 < v2 else round(v2, 4)
                             for v1, v2 in zip(best_rmse, rmse_mae_val_list.loc[:, "rmse"].tolist())]
                best_mae = [round(v1, 4) if v1 is not None and v1 < v2 else round(v2, 4)
                            for v1, v2 in zip(best_mae, rmse_mae_val_list.loc[:, "mae"].tolist())]
            pd_data_summary_table_rmse.loc[:, data_name_list[idx]] = best_rmse
            pd_data_summary_table_mae.loc[:, data_name_list[idx]] = best_mae
        # save data summary result
        pd_data_summary_table_rmse.to_excel(osp.join(data_root_path, f"{method_name}_rmse_summary.xlsx"))
        pd_data_summary_table_mae.to_excel(osp.join(data_root_path, f"{method_name}_mae_summary.xlsx"))

    def processing_historical_method(self, method_name):
        # setup root path for method
        data_root_path = osp.join(self.root_path, method_name)
        logger.info(f"Processing result data with {method_name} method...")
        # get diff data result in method
        sub_dir_lists = [osp.join(data_root_path, sub_dir) for sub_dir in os.listdir(data_root_path)]
        # create data summary table
        pd_data_summary_table_rmse = pd.DataFrame(columns=data_name_list, index=dim_name_list)
        pd_data_summary_table_mae = pd.DataFrame(columns=data_name_list, index=dim_name_list)
        # traversal sub dirs
        for idx, sub_dir in enumerate(sub_dir_lists):
            # get excel data fro sub dir path
            excel_lists = [osp.join(sub_dir, excel_data) for excel_data in os.listdir(sub_dir)]
            # read each excel data
            for excel_path in excel_lists:
                logger.info(f"Currently deal with excel {osp.basename(excel_path)}")
                # rmse and mae data
                rmse_data = pd.read_excel(excel_path, sheet_name="rmse", header=None)
                mae_data = pd.read_excel(excel_path, sheet_name="mae", header=None)
                # rmse val and mae val
                rmse_val = round(rmse_data.loc[rmse_data.index[-1], rmse_data.columns[-1]], 4)
                mae_val = round(mae_data.loc[mae_data.index[-1], mae_data.columns[-1]], 4)
                # get current dimension
                dim_val = re.search(r"d\d+", osp.basename(excel_path))[0]
                assert dim_val is not None, "please check excel name"
                pd_data_summary_table_rmse.loc[dim_val.replace("d", "f="), data_name_list[idx]] = rmse_val
                pd_data_summary_table_mae.loc[dim_val.replace("d", "f="), data_name_list[idx]] = mae_val
        # save data summary result
        pd_data_summary_table_rmse.to_excel(osp.join(data_root_path, f"{method_name}_rmse_summary.xlsx"))
        pd_data_summary_table_mae.to_excel(osp.join(data_root_path, f"{method_name}_mae_summary.xlsx"))


def draw_ablation_picture():
    # setup root path
    output_fig_path = osp.join(r"D:\QYZ\Code\random_latent_factor\output\figs", "inference_f")
    if not osp.exists(output_fig_path):
        os.makedirs(output_fig_path)

    # setup x and y label
    x_label = [20, 40, 60, 80, 120, 160, 200]
    # traver rmse data
    for idx, (data_name, data_list) in enumerate(rmse_data_to_method_dict.items()):
        # draw picture
        rmse_fig = plt.figure()
        rmse_ax = rmse_fig.add_subplot(1, 1, 1)
        for data_item in data_list:
            # draw each line
            rmse_ax.plot(x_label, data_item, markers[idx], ms=4, linestyle='-')
        # set up drawing
        rmse_ax.xaxis.set_ticks(x_label)
        rmse_ax.set_xlabel('Value of f', fontsize=18)
        rmse_ax.set_ylabel('RMSE', fontsize=18)
        rmse_ax.legend(loc='upper center', labels=methods_name_list)
        rmse_fig.show()
        # save fig
        rmse_fig.savefig(f'{osp.join(output_fig_path, f"RMSE_on_{data_name}_inference_f.pdf")}')
    # traver mae data
    for idx, (data_name, data_list) in enumerate(mae_data_to_method_dict.items()):
        # draw picture
        mae_fig = plt.figure()
        mae_ax = mae_fig.add_subplot(1, 1, 1)
        for data_item in data_list:
            # draw each line
            mae_ax.plot(x_label, data_item, markers[idx], ms=4, linestyle='-')
        mae_ax.xaxis.set_ticks(x_label)
        mae_ax.set_xlabel('Value of f', fontsize=18)
        mae_ax.set_ylabel('MAE', fontsize=18)
        mae_ax.legend(loc='upper center', labels=methods_name_list)
        mae_fig.show()
        # save fig
        mae_fig.savefig(f'{osp.join(output_fig_path, f"MAE_on_{data_name}_inference_f.pdf")}')


def deal_with_summary_data(root_path):
    # setup method path list
    method_path_list = [osp.join(root_path, method_path.lower()) for method_path in methods_name_list]

    # traversal path
    for idx, model_path in enumerate(method_path_list):
        # setup data path
        rmse_table_path = osp.join(model_path, f"{osp.basename(model_path)}_rmse_summary.xlsx")
        mae_table_path = osp.join(model_path, f"{osp.basename(model_path)}_mae_summary.xlsx")
        if not osp.exists(rmse_table_path) or not osp.exists(mae_table_path):
            raise ValueError("please check summary table")
        # read rmse and mae data
        rmse_data = pd.read_excel(rmse_table_path, sheet_name="Sheet1", index_col=0)
        mae_data = pd.read_excel(mae_table_path, sheet_name="Sheet1", index_col=0)
        # recode to dict
        for data_name in data_name_list:
            rmse_data_to_method_dict[data_name].append(rmse_data.loc[:, data_name].tolist())
            mae_data_to_method_dict[data_name].append(mae_data.loc[:, data_name].tolist())


if __name__ == '__main__':
    logger = get_root_logger()
    rmse_data_to_method_dict, mae_data_to_method_dict = defaultdict(list), defaultdict(list)
    data_process = DataProcess(root_path=r"D:\QYZ\Code\random_latent_factor\output\excel")
    deal_with_summary_data(root_path=r"D:\QYZ\Code\random_latent_factor\output\excel")
    draw_ablation_picture()
