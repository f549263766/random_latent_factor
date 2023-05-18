import pandas as pd
import os
import numpy as np


class Pty3Rating:

    def __init__(self,
                 rating_path="D:/QYZ/Code/random_latent_factor/dataset/D3-ratings.xlsx",
                 trust_path="D:/QYZ/Code/random_latent_factor/dataset/data3trust.xlsx"):
        self.rating_path = rating_path
        self.trust_path = trust_path
        self.root_path = os.path.dirname(rating_path)
        self.x, self.y, self.x_trust, self.y_trust = None, None, None, None
        self.processing()

    def download(self):
        rating_file = os.path.join(self.root_path, "pty3ratings.xlsx")
        trust_file = os.path.join(self.root_path, "pty3trust.xlsx")
        if os.path.exists(rating_file) and os.path.exists(trust_file):
            dataset = pd.read_excel(rating_file, sheet_name="Sheet1", header=None)
            self.x = dataset.iloc[:, :].to_numpy()
            self.y = dataset.iloc[:, :].to_numpy()
            dataset = pd.read_excel(trust_file, sheet_name="Sheet1", header=None)
            self.x_trust = dataset.iloc[:, :].to_numpy()
            self.y_trust = dataset.iloc[:, :].to_numpy()

            return True

        return False

    def processing_ratings(self):
        print('during processing data rating...')
        # pre-processing ratings dataset
        dataset = pd.read_excel(self.rating_path, sheet_name="user-item-ratings")
        dataset = dataset.drop(dataset[dataset['movie'] > 1508].index, axis=0)
        output_pd = pd.pivot_table(dataset[['user', 'movie', 'ratings']],
                                   columns=['movie'],
                                   index=['user'],
                                   values="ratings",
                                   fill_value=0)
        output_pd = output_pd.combine_first(pd.DataFrame(
            data=0,
            dtype=float,
            columns=output_pd.columns.tolist(),
            index=[i for i in range(1, output_pd.shape[1] + 1)]
        ))
        drop_row = (output_pd != 0).astype(int).sum(axis=1)
        drop_row = list(drop_row[drop_row <= 30].index)
        drop_col = (output_pd != 0).astype(int).sum(axis=0)
        drop_col = list(drop_col[drop_col <= 4].index)
        output_pd = output_pd.drop(drop_row, axis=0)
        output_pd = output_pd.drop(drop_col, axis=1)
        num_row, num_col = output_pd.shape
        output_append = pd.DataFrame(data=[[0] * (num_row - num_col)] * num_row,
                                     index=output_pd.index,
                                     columns=[i for i in range(output_pd.columns[-1] + 1,
                                                               output_pd.columns[-1] + 1 + (num_row - num_col))])
        output_pd = pd.concat([output_pd, output_append], axis=1)
        output_pd.to_excel(os.path.join(self.root_path, "pty3ratings.xlsx"), index=False, header=False)
        self.x = output_pd.iloc[:, :].to_numpy()
        self.y = output_pd.iloc[:, :].to_numpy()

        return drop_row

    def processing_trust(self, rows):
        print('during processing trust dataset...')
        rows = (np.array(rows) - 1).tolist()
        dataset = pd.read_excel(self.trust_path, sheet_name="sheet1", header=None)
        dataset = dataset.drop(rows, axis=0)
        dataset = dataset.drop(rows, axis=1)
        dataset.to_excel(os.path.join(self.root_path, "pty3trust.xlsx"), index=False, header=False)
        self.x_trust = dataset.iloc[:, :].to_numpy()
        self.y_trust = dataset.iloc[:, :].to_numpy()

    def processing(self):
        if not self.download():
            drop_rows_list = self.processing_ratings()
            self.processing_trust(drop_rows_list)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


if __name__ == '__main__':
    pty = Pty3Rating()
    print(pty.x.shape)
    print(pty.y.shape)
    print(pty.x_trust.shape)
    print(pty.y_trust.shape)
