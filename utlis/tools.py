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


