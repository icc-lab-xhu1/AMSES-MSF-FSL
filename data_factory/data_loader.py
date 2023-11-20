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
from sklearn.preprocessing import MinMaxScaler
import pickle


class dataSegLoader(object):
    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1  - self.horizon
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1 - self.horizon

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.double(self.train[index:index + self.win_size]), np.double(self.train[index + self.win_size:index + self.win_size + self.horizon])
        elif (self.mode == 'test'):
            return np.double(self.test[index:index + self.win_size]), np.double(self.test[index + self.win_size:index + self.win_size + self.horizon])

class cpuSegLoader(dataSegLoader):
    def __init__(self, data_path, win_size, step, mode="train", horizon=1):
        self.data_path = data_path
        self.horizon = horizon
        self.mode = mode
        self.step = step
        self.win_size = win_size
        name = self.data_path.split('/')[1]
        # name = name + '_latency'

        train_data = pd.read_csv(self.data_path + "/" + name + "_train.csv")
        for i in train_data.columns:
            if i.split('_')[-1] != 'cpu':
                train_data = train_data.drop(i, axis=1)
        train_latency = np.array(train_data)
        # train_data = np.array(train_data)
        # train_data = np.load(self.data_path + "/" + name + "_train.npy")
        # test_data = np.load(self.data_path +  "/"+name+"_test.npy")

        self.train = train_latency

        # self.test_labels = np.load(self.data_path + "/"+name+"_label.npy").astype(int)
        print(self.train.shape)

class memSegLoader(dataSegLoader):
    def __init__(self, data_path, win_size, step, mode="train", horizon=1):
        self.data_path = data_path
        self.horizon = horizon
        self.mode = mode
        self.step = step
        self.win_size = win_size
        name = self.data_path.split('/')[1]
        # name = name + '_latency'

        train_data = pd.read_csv(self.data_path + "/" + name + "_train.csv")
        for i in train_data.columns:
            if i.split('_')[-1] != 'mem':
                train_data = train_data.drop(i, axis=1)
        train_latency = np.array(train_data)
        # train_data = np.array(train_data)
        # train_data = np.load(self.data_path + "/" + name + "_train.npy")
        # test_data = np.load(self.data_path +  "/"+name+"_test.npy")

        self.train = train_latency

        # self.test_labels = np.load(self.data_path + "/"+name+"_label.npy").astype(int)
        print(self.train.shape)

class latencySegLoader(dataSegLoader):
    def __init__(self, data_path, win_size, step, mode="train", horizon=1):
        self.data_path = data_path
        self.horizon = horizon
        self.mode = mode
        self.step = step
        self.win_size = win_size
        name = self.data_path.split('/')[1]
        # name = name + '_latency'

        train_data = pd.read_csv(self.data_path + "/" + name + "_train.csv")
        for i in train_data.columns:
            if i.split('_')[-1] != 'latency':
                train_data = train_data.drop(i, axis=1)
        train_latency = np.array(train_data)
        # train_data = np.array(train_data)
        # train_data = np.load(self.data_path + "/" + name + "_train.npy")
        # test_data = np.load(self.data_path +  "/"+name+"_test.npy")

        self.train = train_latency

        # self.test_labels = np.load(self.data_path + "/"+name+"_label.npy").astype(int)
        print(self.train.shape)

# def get_loader_segment_memory(data_path, batch_size, win_size=6, step=1, mode='train', dataset='mem', horizon=1):
#     dataset = eval(dataset+'SegLoader')(data_path, win_size, step, mode, horizon=1)
#
#     data_loader = DataLoader(dataset=dataset,
#                              batch_size=batch_size,
#                              shuffle=False,
#                              num_workers=8)
#     return dataset, data_loader
#
# def get_loader_segment_latency(data_path, batch_size, win_size=6, step=1, mode='train', dataset='latency', horizon=1):
#     dataset = eval(dataset+'SegLoader')(data_path, win_size, step, mode, horizon=1)
#     data_loader = DataLoader(dataset=dataset,
#                              batch_size=batch_size,
#                              shuffle=False,
#                              num_workers=8)
#     return dataset, data_loader
#
# def get_loader_segment_cpu(data_path, batch_size, win_size=6, step=1, mode='train', dataset='cpu', horizon=1):
#     dataset = eval(dataset+'SegLoader')(data_path, win_size, step, mode, horizon=1)
#     data_loader = DataLoader(dataset=dataset,
#                              batch_size=batch_size,
#                              shuffle=False,
#                              num_workers=8)
#     return dataset, data_loader

def get_loader_segment(data_path, batch_size, win_size=6, step=1, mode='train', dataset='cpu', horizon=1):
    dataset = eval(dataset+'SegLoader')(data_path, win_size, step, mode, horizon=1)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=8)
    return dataset, data_loader

