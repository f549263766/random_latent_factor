import pandas as pd
import os


class Pty3Rating:

    def __init__(self, data_path="D:/QYZ/Code/random_latent_factor/dataset/D3-ratings.xlsx"):
        self.data_path = data_path
        self.root_path = os.path.dirname(data_path)
        self.x, self.y = None, None
        self.processing()

    def download(self):
        if os.path.exists(os.path.join(self.root_path, "pty3ratings.xlsx")):
            dataset = pd.read_excel(os.path.join(self.root_path, "pty3ratings.xlsx"), sheet_name="Sheet1")
            self.x = dataset.iloc[:, 0:2].to_numpy()
            self.y = dataset.iloc[:, 2].to_numpy().reshape(-1, 1)

            return True

        return False

    def processing(self):
        if not self.download():
            print('during processing...')
            # pre-processing dataset
            dataset = pd.read_excel(self.data_path, sheet_name="user-item-ratings")
            index = dataset[dataset['user'] == 601].index.tolist()[0]
            dataset = dataset.iloc[:index, :]
            dataset = dataset.drop(index=dataset[dataset['movie'] > 600].index)
            # generate new table
            output_pd = pd.DataFrame(data=0,
                                     columns=[i for i in range(600)],
                                     index=[i for i in range(600)],
                                     dtype=float)
            for _, row in dataset.iterrows():
                idx_x, idx_y, score = int(row[0]), int(row[1]), float(row[2])
                output_pd.loc[idx_x - 1][idx_y - 1] = score
            output_pd.to_excel(os.path.join(self.root_path, "pty3ratings.xlsx"), index=False, header=False)

            self.x = output_pd.iloc[:, 0:2].to_numpy()
            self.y = output_pd.iloc[:, 2].to_numpy().reshape(-1, 1)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


class Pty3RatingConfidence:

    def __init__(self, data_path="D:/QYZ/Code/random_latent_factor/dataset/data3trust.xlsx"):
        self.data_path = data_path
        self.root_path = os.path.dirname(data_path)
        self.x, self.y = None, None
        self.processing()

    def download(self):
        if os.path.exists(os.path.join(self.root_path, "pty3trust.xlsx")):
            dataset = pd.read_excel(os.path.join(self.root_path, "pty3trust.xlsx"), sheet_name="Sheet1")
            self.x = dataset.iloc[:, 0:2].to_numpy()
            self.y = dataset.iloc[:, 2].to_numpy().reshape(-1, 1)

            return True

        return False

    def processing(self):
        if not self.download():
            print('during processing...')
            dataset = pd.read_excel(self.data_path, sheet_name="sheet1", header=None)
            col = 600 if len(dataset.columns) > 600 else len(dataset.columns)
            data_set = [[i, j, dataset.iloc[i - 1, j - 1]]
                        for i in range(1, col + 1) for j in range(1, col + 1)]
            df = pd.DataFrame(data_set, columns=['user1', 'user2', 'trust'])
            df.to_excel(os.path.join(self.root_path, "pty3trust.xlsx"), index=False)
            self.x = df.iloc[:, 0:2].to_numpy()
            self.y = df.iloc[:, 2].to_numpy().reshape(-1, 1)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


if __name__ == '__main__':
    pty = Pty3Rating()
    print(pty.x.shape)
    print(pty.y.shape)

    pty2 = Pty3RatingConfidence()
    print(pty2.x.shape)
    print(pty2.y.shape)
