import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from config import *

path = '../data/semi_simulationMotion/'
train_input = np.load(path + 'train_input.npy')
train_output = np.load(path + 'train_output.npy')
train_location = np.load(path + 'train_location.npy', allow_pickle=True)

val_input = np.load(path + 'val_input.npy')
val_output = np.load(path + 'val_output.npy')
val_location = np.load(path + 'val_location.npy')

test_input = np.load(path + 'test_input.npy')
test_output = np.load(path + 'test_output.npy')
test_location = np.load(path + 'test_location.npy')


class my_dataset(Dataset):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.z[idx]


train_dataset = my_dataset(train_input, train_output, train_location)
val_dataset = my_dataset(val_input, val_output, val_location)
test_dataset = my_dataset(test_input, test_output, test_location)

# 利用DataLoader加载数据
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=int(test_output.shape[0]/11), shuffle=False, drop_last=False)