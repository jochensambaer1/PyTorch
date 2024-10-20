import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import math

class WineDataset(Dataset):

    def __init__(self):
        # Initialize data, download, etc.
        # read with numpy or pandas
        xy = np.loadtxt('data/wine/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]

        # here the first column is the class label, the rest are the features
        self.x_data = torch.from_numpy(xy[:, 1:]) # size [n_samples, n_features]
        self.y_data = torch.from_numpy(xy[:, [0]]) # size [n_samples, 1]

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # return the size of the dataset
    def __len__(self):
        return self.n_samples

if __name__ == '__main__':
    # Create dataset
    dataset = WineDataset()
    train_loader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=2)

    # Iterate through the dataset
    dataiter = iter(train_loader)
    data = next(dataiter)
    print(data)
