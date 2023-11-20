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
    def __init__(self, data_path,  anomaly, win_size, step, mode="train", horizon=1):
        self.data_path = data_path
        self.horizon = horizon
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.anomaly = anomaly
        name = self.data_path.split('/')[1]
        # name = name + '_latency'

        train_data = pd.read_csv(self.data_path + "/" + name + "_train.csv")
        anomaly_data = train_data[train_data['label'] == 1]
        random_train = random.randint(0, len(anomaly_data) - self.anomaly)
        anomaly_data = anomaly_data.drop(anomaly_data.iloc[random_train: random_train + self.anomaly, :].index, axis=0)
        train_data = train_data.drop(anomaly_data.index, axis=0)
        label = np.array(train_data.label.values)
        two_label1 = []
        for i in label:
            if i == 1:
                a = 1
                two_label1.append(a)
            else:
                a = 0
                two_label1.append(a)
        two_label1 = np.array(two_label1)
        for i in train_data.columns:
            if 'cpu' not in i:
                train_data = train_data.drop(i, axis=1)
        train_data['label'] = two_label1
        # print(train_data)
        train_latency = np.array(train_data)
        # train_data = np.array(train_data)
        # train_data = np.load(self.data_path + "/" + name + "_train.npy")
        # test_data = np.load(self.data_path +  "/"+name+"_test.npy")

        self.train = train_latency

        # self.test_labels = np.load(self.data_path + "/"+name+"_label.npy").astype(int)
        print('train shape:', self.train.shape)

class rxSegLoader(dataSegLoader):
    def __init__(self, data_path, anomaly, win_size, step, mode="train", horizon=1):
        self.data_path = data_path
        self.horizon = horizon
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.anomaly = anomaly
        name = self.data_path.split('/')[1]
        # name = name + '_latency'

        train_data = pd.read_csv(self.data_path + "/" + name + "_train.csv")
        anomaly_data = train_data[train_data['label'] == 2]
        random_train = random.randint(0, len(anomaly_data) - self.anomaly)
        anomaly_data = anomaly_data.drop(anomaly_data.iloc[random_train: random_train + self.anomaly, :].index, axis=0)
        train_data = train_data.drop(anomaly_data.index, axis=0)
        label = np.array(train_data.label.values)
        two_label1 = []
        for i in label:
            if i == 2:
                a = 1
                two_label1.append(a)
            else:
                a = 0
                two_label1.append(a)
        two_label1 = np.array(two_label1)
        for i in train_data.columns:
            if 'rx' not in i:
                train_data = train_data.drop(i, axis=1)
        train_data['label'] = two_label1
        # print(train_data)
        train_latency = np.array(train_data)
        # train_data = np.array(train_data)
        # train_data = np.load(self.data_path + "/" + name + "_train.npy")
        # test_data = np.load(self.data_path +  "/"+name+"_test.npy")

        self.train = train_latency

        # self.test_labels = np.load(self.data_path + "/"+name+"_label.npy").astype(int)
        print(self.train.shape)

class txSegLoader(dataSegLoader):
    def __init__(self, data_path, anomaly, win_size, step, mode="train", horizon=1):
        self.data_path = data_path
        self.horizon = horizon
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.anomaly = anomaly
        name = self.data_path.split('/')[1]
        # name = name + '_latency'
        # print('anomaly', self.anomaly)
        train_data = pd.read_csv(self.data_path + "/" + name + "_train.csv")
        anomaly_data = train_data[train_data['label'] == 3]
        random_train = random.randint(0, len(anomaly_data) - self.anomaly)
        # print(random_train)
        anomaly_data = anomaly_data.drop(anomaly_data.iloc[random_train: random_train + self.anomaly, :].index, axis=0)
        train_data = train_data.drop(anomaly_data.index, axis=0)
        label = np.array(train_data.label.values)
        two_label1 = []
        for i in label:
            if i == 3:
                a = 1
                two_label1.append(a)
            else:
                a = 0
                two_label1.append(a)
        two_label1 = np.array(two_label1)
        for i in train_data.columns:
            if 'tx' not in i:
                train_data = train_data.drop(i, axis=1)
        train_data['label'] = two_label1
        # print(train_data)
        train_latency = np.array(train_data)
        # train_data = np.array(train_data)
        # train_data = np.load(self.data_path + "/" + name + "_train.npy")
        # test_data = np.load(self.data_path +  "/"+name+"_test.npy")

        self.train = train_latency

        # self.test_labels = np.load(self.data_path + "/"+name+"_label.npy").astype(int)
        print('train shape', self.train.shape)

class memSegLoader(dataSegLoader):
    def __init__(self, data_path, anomaly, win_size, step, mode="train", horizon=1):
        self.data_path = data_path
        self.horizon = horizon
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.anomaly = anomaly
        name = self.data_path.split('/')[1]
        # name = name + '_latency'

        train_data = pd.read_csv(self.data_path + "/" + name + "_train.csv")
        anomaly_data = train_data[train_data['label'] == 2]
        random_train = random.randint(0, len(anomaly_data) - self.anomaly)
        anomaly_data = anomaly_data.drop(anomaly_data.iloc[random_train: random_train + self.anomaly, :].index, axis=0)
        train_data = train_data.drop(anomaly_data.index, axis=0)
        label = np.array(train_data.label.values)
        two_label1 = []
        for i in label:
            if i == 2:
                a = 1
                two_label1.append(a)
            else:
                a = 0
                two_label1.append(a)
        two_label1 = np.array(two_label1)
        for i in train_data.columns:
            if 'mem' not in i:
                train_data = train_data.drop(i, axis=1)
        train_data['label'] = two_label1
        # print(train_data)
        train_latency = np.array(train_data)
        # train_data = np.array(train_data)
        # train_data = np.load(self.data_path + "/" + name + "_train.npy")
        # test_data = np.load(self.data_path +  "/"+name+"_test.npy")

        self.train = train_latency

        # self.test_labels = np.load(self.data_path + "/"+name+"_label.npy").astype(int)
        print(self.train.shape)

class latencySegLoader(dataSegLoader):
    def __init__(self, data_path, anomaly, win_size, step, mode="train", horizon=1):
        self.data_path = data_path
        self.horizon = horizon
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.anomaly = anomaly
        name = self.data_path.split('/')[1]
        # name = name + '_latency'
        # print('anomaly', self.anomaly)
        train_data = pd.read_csv(self.data_path + "/" + name + "_train.csv")
        anomaly_data = train_data[train_data['label'] == 3]
        random_train = random.randint(0, len(anomaly_data) - self.anomaly)
        # print(random_train)
        anomaly_data = anomaly_data.drop(anomaly_data.iloc[random_train: random_train + self.anomaly, :].index, axis=0)
        train_data = train_data.drop(anomaly_data.index, axis=0)
        label = np.array(train_data.label.values)
        two_label1 = []
        for i in label:
            if i == 3:
                a = 1
                two_label1.append(a)
            else:
                a = 0
                two_label1.append(a)
        two_label1 = np.array(two_label1)
        for i in train_data.columns:
            if 'latency' not in i:
                train_data = train_data.drop(i, axis=1)
        train_data['label'] = two_label1
        # print(train_data)
        train_latency = np.array(train_data)
        # train_data = np.array(train_data)
        # train_data = np.load(self.data_path + "/" + name + "_train.npy")
        # test_data = np.load(self.data_path +  "/"+name+"_test.npy")

        self.train = train_latency

        # self.test_labels = np.load(self.data_path + "/"+name+"_label.npy").astype(int)
        print('train shape', self.train.shape)

def get_loader_segment(data_path, batch_size, anomaly, win_size=6, step=2, mode='train', dataset='cpu', horizon=2, ):
    dataset = eval(dataset+'SegLoader')(data_path, anomaly, win_size, step, mode,  horizon=2)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=8)
    return dataset, data_loader

