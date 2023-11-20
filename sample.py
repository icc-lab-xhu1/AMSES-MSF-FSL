import random

import pandas as pd
from sklearn import preprocessing, metrics
import numpy as np
from args import get_parser

import pickle as pk

def chouyang(data_name, sample_rca):

    dict_values = {'cpu': 1, 'mem': 2, 'latency': 3}
    parser = get_parser()
    config = parser.parse_args()
    name = config.data_path.split('/')[1]
    # args = vars(config)
    data = pd.read_csv(config.data_path + "/" + name + ".csv")
    # data = pd.read_csv(r'C:\pycharmwork\data_supplement\shipping.csv')
    inx = data[data['Unnamed: 0'] == 0].index
    cpu = data.iloc[: inx[1],:]
    memory = data.iloc[inx[1]: inx[2],:]
    latency = data.iloc[inx[2]: ,:]
    # cpu
    # values_cpu = [v for k, v in dict_values.items() if k == 'cpu'][0]
    train_cpu = cpu.iloc[:int(cpu.shape[0]*0.7),:]
    train_memory = memory.iloc[:int(memory.shape[0] * 0.7), :]
    train_latency = latency.iloc[:int(latency.shape[0] * 0.7), :]

    if sample_rca:
        rca_train = pd.concat([train_cpu, train_memory, train_latency])
        rca_data = rca_train[rca_train['label'] == [v for k, v in dict_values.items() if k == data_name][0]]

        return rca_data
    else:
        train_cpu_anomaly = train_cpu[train_cpu['label'] == 1]
        random_train = np.random.randint(low=0, high=len(train_cpu_anomaly.index), size=(1), dtype='int')
        # random_train = random.randint(0, len(train_cpu_anomaly.index))
        print('cpu random', random_train)
        index = train_cpu_anomaly.index[random_train[0]]
        # index = train_cpu_anomaly.index[random_train]
        first_index = index - 50
        last_index = index + 50 - 1
        if first_index < train_cpu.index[0]:
            f_index = train_cpu.index[0]
            l_index = last_index + abs(train_cpu.index[0] - first_index)
        elif train_cpu.index[-1] - last_index < 0:
            l_index = train_cpu.index[-1]
            f_index = first_index - abs(train_cpu.index[-1] - last_index)
        else:
            f_index = first_index
            l_index = last_index


        sample_cpu = train_cpu.loc[f_index:l_index, :]
        # sample_cpu.label.unique()

        no_sample_cpu = train_cpu.drop(list(range(f_index, l_index+1)))
        no_sample_cpu = no_sample_cpu[no_sample_cpu['label'] == 0]

        # memory

        # values_mem = [v for k, v in dict_values.items() if k == 'mem'][0]
        train_memory_anomaly = train_memory[train_memory['label'] == 2]
        random_train = np.random.randint(low=0, high=len(train_memory_anomaly.index), size=(1), dtype='int')
        # random_train = random.randint(0, len(train_memory_anomaly.index))
        print('mem random', random_train)
        index = train_memory_anomaly.index[random_train[0]]
        # index = train_memory_anomaly.index[random_train]
        first_index = index - 50
        last_index = index + 50 - 1
        if first_index < train_memory.index[0]:
            f_index = train_memory.index[0]
            l_index = last_index + abs(train_memory.index[0] - first_index)
        elif train_memory.index[-1] - last_index < 0:
            l_index = train_memory.index[-1]
            f_index = first_index - abs(train_memory.index[-1] - last_index)
        else:
            f_index = first_index
            l_index = last_index

        sample_memory = train_memory.loc[f_index:l_index, :]
        sample_memory.label.unique()
        no_sample_memory = train_memory.drop(list(range(f_index, l_index)))
        no_sample_memory = no_sample_memory[no_sample_memory['label'] == 0]

        # latency
        # values_latency = [v for k, v in dict_values.items() if k == 'latency'][0]

        train_latency_anomaly = train_latency[train_latency['label'] == 3]
        # random_train = random.randint(0, len(train_latency_anomaly.index))
        random_train = np.random.randint(low=0, high=len(train_latency_anomaly.index), size=(1), dtype='int')
        print('latency random', random_train)
        index = train_latency_anomaly.index[random_train[0]]
        # index = train_latency_anomaly.index[random_train]

        first_index = index - 50
        last_index = index + 50 - 1
        if first_index < train_latency.index[0]:
            f_index = train_latency.index[0]
            l_index = last_index + abs(train_latency.index[0] - first_index)
        elif train_latency.index[-1] - last_index < 0:
            l_index = train_latency.index[-1]
            f_index = first_index - abs(train_latency.index[-1] - last_index)
        else:
            f_index = first_index
            l_index = last_index

        sample_latency = train_latency.loc[f_index:l_index, :]
        # sample_latency.label.unique()
        no_sample_latency = train_latency.drop(list(range(f_index,l_index)))
        no_sample_latency = no_sample_latency[no_sample_latency['label'] == 0]





        # concat
        sample_test = pd.concat([sample_cpu, sample_memory,  sample_latency])
        print('test label:', sample_test.label.unique())
        train = pd.concat([no_sample_cpu, no_sample_memory, no_sample_latency])
        train = train.drop(['Unnamed: 0', 'label', 'time'],axis = 1)
        MinMaxScaler = preprocessing.MinMaxScaler()
        user_train = pd.DataFrame(MinMaxScaler.fit_transform(train))
        user_train.columns = train.columns
        label = sample_test.label.values
        test_data=sample_test.drop(['Unnamed: 0', 'label', 'time'],axis = 1)
        MinMaxScaler = preprocessing.MinMaxScaler()
        sample_test = pd.DataFrame(MinMaxScaler.fit_transform(test_data))
        sample_test.columns = test_data.columns

        # rca_data = sample_test
        # rca_data['label'] = label

        # random_train = np.random.randint(low=0, high=len(user_train) - 500, size=(1), dtype='int')
        # user_train = user_train.iloc[random_train[0]:random_train[0] + 500, :]
        # pre_label = np.array(test_data['label'].values)
        for i in user_train.columns:
            if i.split('_')[-1] != data_name:
                user_train = user_train.drop(i, axis=1)
        for i in sample_test.columns:
            if i.split('_')[-1] != data_name:
                sample_test = sample_test.drop(i, axis=1)
        user_train = np.array(user_train)
        test_cpu = np.array(sample_test)
        input_cpu = user_train.shape[1]
        values_1 = [v for k, v in dict_values.items() if k == data_name]
        return user_train, test_cpu, label, input_cpu, values_1
    # return user_train, test_cpu, label, input_cpu, values_1, rca_data
