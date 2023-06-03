import pandas as pd
import os.path as osp
import os
import numpy as np
from utlis.logger import get_root_logger


class Pty1Rating:
    """

    """

    def __init__(self,
                 root="D:/QYZ/Code/random_latent_factor/dataset/",
                 user_rating_data_path="D1-movie-ratings.xlsx",
                 user_trust_data_path="D1-trust-data.xlsx",
                 logger=get_root_logger()):
        self.root_path = root
        self.user_rating_data_path = osp.join(root, "raw", user_rating_data_path)
        self.user_trust_data_path = osp.join(root, "raw", user_trust_data_path)
        self.logger = logger
        self.x, self.y, self.x_trust, self.y_trust, self.x_mistrust, self.y_mistrust = [None] * 6
        self.processing()

    def download(self):
        self.logger.info("Loading processed and saved dataset..")
        # Set the saved file path
        rating_file = os.path.join(self.root_path, "processing", "D1-pty1ratings.xlsx")
        trust_file = os.path.join(self.root_path, "processing", "D1-pty1trust.xlsx")
        mistrust_file = os.path.join(self.root_path, "processing", "D1-pty1mistrust.xlsx")
        # Load dataset from save excel
        if os.path.exists(rating_file) and os.path.exists(trust_file) and os.path.exists(mistrust_file):
            dataset = pd.read_excel(rating_file, sheet_name="Sheet1", header=None)
            self.x = dataset.iloc[:, :].to_numpy()
            self.y = dataset.iloc[:, :].to_numpy()
            dataset = pd.read_excel(trust_file, sheet_name="Sheet1", header=None)
            self.x_trust = dataset.iloc[:, :].to_numpy()
            self.y_trust = dataset.iloc[:, :].to_numpy()
            dataset = pd.read_excel(mistrust_file, sheet_name="Sheet1", header=None)
            self.x_mistrust = dataset.iloc[:, :].to_numpy()
            self.y_mistrust = dataset.iloc[:, :].to_numpy()

            return True

        return False

    def processing_ratings(self):
        self.logger.info("During processing user ratings dataset..")
        # pre-processing user ratings dataset
        user_ratings_dataset = pd.read_excel(self.user_rating_data_path, sheet_name="Sheet2")
        # Only 1508 items remain rated
        user_ratings_dataset = user_ratings_dataset.drop(
            user_ratings_dataset[user_ratings_dataset['moving'] > 2070].index, axis=0)
        # Build user item rating pivot table
        user_item_pivot = pd.pivot_table(user_ratings_dataset[['user', 'moving', 'rating']],
                                         columns=['moving'],
                                         index=['user'],
                                         values="rating",
                                         fill_value=0)
        # Completing the missing users
        user_item_pivot = user_item_pivot.combine_first(pd.DataFrame(
            data=0,
            dtype=float,
            columns=user_item_pivot.columns.tolist(),
            index=[i for i in range(1, user_item_pivot.shape[1] + 1)]
        ))
        # Count the number of non-zero elements in rows and columns
        static_nonzero = (user_item_pivot != 0).astype(int)
        count_row_nonzero = static_nonzero.sum(axis=1)
        count_col_nonzero = static_nonzero.sum(axis=0)
        # Discard rows and columns whose number of non-zero elements is below the threshold
        drop_row_index_list = list(count_row_nonzero[count_row_nonzero <= 2].index)
        drop_col_index_list = list(count_col_nonzero[count_col_nonzero <= 2].index)
        user_item_pivot = user_item_pivot.drop(drop_row_index_list, axis=0).drop(drop_col_index_list, axis=1)
        # Expand the matrix to a square matrix
        num_row, num_col = user_item_pivot.shape
        user_item_expand = pd.DataFrame(
            data=[[0] * (num_row - num_col)] * num_row,
            index=user_item_pivot.index,
            columns=[i for i in range(user_item_pivot.columns[-1] + 1,
                                      user_item_pivot.columns[-1] + 1 + (num_row - num_col))])
        user_item_pivot = pd.concat([user_item_pivot, user_item_expand], axis=1)
        # Export data to excel sheet
        user_item_pivot.to_excel(os.path.join(self.root_path, "processing", "D1-pty1ratings.xlsx"),
                                 index=False,
                                 header=False)

        self.x = user_item_pivot.iloc[:, :].to_numpy()
        self.y = user_item_pivot.iloc[:, :].to_numpy()

        return drop_row_index_list

    def processing_trust(self, drop_row_index_list):
        self.logger.info("During processing user trust dataset..")
        # Update the discard index list
        drop_row_index_list = (np.array(drop_row_index_list) - 1).tolist()
        # pre-processing user trust dataset
        user_trust_dataset = pd.read_excel(self.user_trust_data_path, sheet_name="sheet1", header=None)
        # Discard rows and columns whose number of non-zero elements is below the threshold
        user_trust_dataset = user_trust_dataset.drop(drop_row_index_list, axis=0).drop(drop_row_index_list, axis=1)
        # Export data to excel sheet
        user_trust_dataset.to_excel(os.path.join(self.root_path, "processing", "D1-pty1trust.xlsx"),
                                    index=False, header=False)

        self.x_trust = user_trust_dataset.iloc[:, :].to_numpy()
        self.y_trust = user_trust_dataset.iloc[:, :].to_numpy()

        return user_trust_dataset

    def processing_mistrust(self, user_trust_dataset):
        self.logger.info("During processing user mistrust dataset..")
        # Update user mistrust dataset
        user_mistrust_dataset = (user_trust_dataset - 1) * -1
        # Export data to excel sheet
        user_mistrust_dataset.to_excel(os.path.join(self.root_path, "processing", "D1-pty1mistrust.xlsx"),
                                       index=False, header=False)

        self.x_mistrust = user_mistrust_dataset.iloc[:, :].to_numpy()
        self.y_mistrust = user_mistrust_dataset.iloc[:, :].to_numpy()

    def processing(self):
        if not self.download():
            drop_rows_list = self.processing_ratings()
            user_trust_dataset = self.processing_trust(drop_rows_list)
            self.processing_mistrust(user_trust_dataset)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


if __name__ == '__main__':
    pty = Pty1Rating()
    print(pty.x.shape)
    print(pty.y.shape)
    print(pty.x_trust.shape)
    print(pty.y_trust.shape)
    print(pty.x_mistrust.shape)
    print(pty.y_mistrust.shape)
