import os.path as osp
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def save_result_with_excel(data_list,
                           file_name,
                           save_root_path):
    assert isinstance(data_list, list)
    assert isinstance(file_name, str)
    assert isinstance(save_root_path, str)

    if not osp.exists(save_root_path):
        os.makedirs(save_root_path)

    np_data = np.array(data_list)
    df_data = {
        'hidden_dim': np_data[:, 0],
        'rmse': np_data[:, 1],
        'mae': np_data[:, 2]
    }
    df = pd.DataFrame(df_data)
    df.to_excel(osp.join(save_root_path, file_name), index=False)


def save_best_result(data_item,
                     file_name,
                     save_root_path):
    assert isinstance(data_item, dict)
    assert isinstance(file_name, str)
    assert isinstance(save_root_path, str)

    if not osp.exists(save_root_path):
        os.makedirs(save_root_path)

    df = pd.DataFrame.from_dict(data_item, orient='index').T
    df.to_excel(osp.join(save_root_path, file_name), index=False)


def outliers_replaced_with_mean_values(data_list):
    assert isinstance(data_list, list), "Must be a list"
    assert len(np.array(data_list).shape) == 1, "Must be a 1D-array"

    data_array = np.asarray(data_list, dtype=float)
    data_mean = np.mean(data_array, axis=0)
    data_std = np.std(data_array, axis=0)

    floor = data_mean - 3 * data_std
    upper = data_mean + 3 * data_std

    output_list = [float(np.where(((val < floor) | (val > upper)), data_mean, val)) for val in data_array]

    return output_list
