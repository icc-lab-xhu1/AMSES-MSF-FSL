import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import collections
import numbers
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle





class SMDSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        # self.scaler = StandardScaler()
        data = np.load(data_path + "/SMD_train.npy")
        # self.scaler.fit(data)
        # data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMD_test.npy")
        # self.test = self.scaler.transform(test_data)
        self.test = test_data
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(data_path + "/SMD_test_label.npy")

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.double(self.train[index:index + self.win_size]), np.double(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.double(self.val[index:index + self.win_size]), np.double(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.double(self.test[index:index + self.win_size]), np.double(
                self.test_labels[index:index + self.win_size])
        else:
            return np.double(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.double(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
        
        
class SWaTSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        # self.scaler = StandardScaler()
        data = np.load(data_path + "/swat_x_train_de5_d10.npy")
        # self.scaler.fit(data)
        # data = self.scaler.transform(data)
        test_data = np.load(data_path + "/swat_x_test_de5_d10.npy")
        # self.test = self.scaler.transform(test_data)
        self.test = test_data
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(data_path + "/swat_y_test_de5_d10.npy")

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.double(self.train[index:index + self.win_size]), np.double(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.double(self.val[index:index + self.win_size]), np.double(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.double(self.test[index:index + self.win_size]), np.double(
                self.test_labels[index:index + self.win_size])
        else:
            return np.double(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.double(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


def get_loader_segment(data_path, batch_size, win_size=6, step=1, mode='train', dataset='SMD'):
    if (dataset == 'SMD'):
        dataset = SMDSegLoader(data_path, win_size, step, mode)
    elif (dataset == 'SWaT'):
        dataset = SWaTSegLoader(data_path, win_size, 1, mode)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader

