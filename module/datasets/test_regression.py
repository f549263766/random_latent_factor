import numpy as np


class RegressionDataset:
    def __init__(self):
        x = np.linspace(0, 20, 200)
        noise = np.random.normal(0, 0.08, 200)
        y = np.sin(x) + np.cos(0.5) * x + noise
        self.x = np.array(x).reshape(-1, 1)
        self.y = np.array(y).reshape(-1, 1)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


if __name__ == '__main__':
    reg = RegressionDataset()
    print(reg.x.shape)
    print(reg.y.shape)
