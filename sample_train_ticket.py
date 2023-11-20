import random

import pandas as pd
from sklearn import preprocessing, metrics
import numpy as np
from args import get_parser

import pickle as pk

def chouyang(data_name):
    dict_values = {'cpu': 1, 'rx': 2, 'tx': 3}
    parser = get_parser()
    config = parser.parse_args()
    name = config.data_path.split('/')[1]
    # args = vars(config)
    data = pd.read_csv(config.data_path + "/" + name + "_train.csv")
    # data = pd.read_csv(r'C:\pycharmwork\data_supplement\shipping.csv')
    # inx = data[data['index'] == 0].index
    # cpu = data.iloc[: inx[1],:]
    # memory = data.iloc[inx[1]: inx[2],:]
    # latency = data.iloc[inx[2]: ,:]
    # # cpu
    # # values_cpu = [v for k, v in dict_values.items() if k == 'cpu'][0]
    # train_cpu = cpu.iloc[:int(cpu.shape[0]*0.7),:]
    train_cpu_anomaly = data[data['label'] == 1]
    random_train = np.random.randint(low=0, high=len(train_cpu_anomaly.index), size=(1), dtype='int')
    # random_train = random.randint(0, len(train_cpu_anomaly.index))
    print('cpu random', random_train)
    index = train_cpu_anomaly.index[random_train[0]]
    # index = train_cpu_anomaly.index[random_train]
    first_index = index - 50
    last_index = index + 50 - 1
    if first_index < data.index[0]:
        fcpu_index = data.index[0]
        lcpu_index = last_index + abs(data.index[0] - first_index)
    elif data.index[-1] - last_index < 0:
        lcpu_index = data.index[-1]
        fcpu_index = first_index - abs(data.index[-1] - last_index)
    else:
        fcpu_index = first_index
        lcpu_index = last_index


    sample_cpu = data.loc[fcpu_index:lcpu_index, :]
    # sample_cpu.label.unique()

    no_sample_cpu = data.drop(list(range(fcpu_index, lcpu_index+1)))
    # no_sample_cpu = no_sample_cpu.reset_index()
    # no_sample_cpu.reset_index().drop(['index'], axis = 1, inplace=True)
    # no_sample_cpu = no_sample_cpu[no_sample_cpu['label'] == 0]

    # memory
    # train_memory = memory.iloc[:int(memory.shape[0] * 0.7), :]
    # values_mem = [v for k, v in dict_values.items() if k == 'mem'][0]
    train_memory_anomaly = data[data['label'] == 2]
    random_train = np.random.randint(low=0, high=len(train_memory_anomaly.index), size=(1), dtype='int')
    # random_train = random.randint(0, len(train_memory_anomaly.index))
    print('rx random', random_train)
    index = train_memory_anomaly.index[random_train[0]]
    # index = train_memory_anomaly.index[random_train]
    first_index = index - 50
    last_index = index + 50 - 1
    if first_index < data.index[0]:
        fmem_index = data.index[0]
        lmem_index = last_index + abs(data.index[0] - first_index)
    elif data.index[-1] - last_index < 0:
        lmem_index = data.index[-1]
        fmem_index = first_index - abs(data.index[-1] - last_index)
    else:
        fmem_index = first_index
        lmem_index = last_index

    sample_memory = data.loc[fmem_index:lmem_index, :]
    # sample_memory.label.unique()
    no_sample_memory = no_sample_cpu.drop(list(range(fmem_index, lmem_index)))
    # no_sample_memory.reset_index().drop(['index'], axis = 1, inplace=True)
    # no_sample_memory = no_sample_memory[no_sample_memory['label'] == 0]

    # latency
    # values_latency = [v for k, v in dict_values.items() if k == 'latency'][0]
    # train_latency = latency.iloc[:int(latency.shape[0]*0.7),:]
    train_latency_anomaly = data[data['label'] == 3]
    # random_train = random.randint(0, len(train_latency_anomaly.index))
    random_train = np.random.randint(low=0, high=len(train_latency_anomaly.index), size=(1), dtype='int')
    print('tx random', random_train)
    index = train_latency_anomaly.index[random_train[0]]
    # index = train_latency_anomaly.index[random_train]

    first_index = index - 50
    last_index = index + 50 - 1
    if first_index < data.index[0]:
        flat_index = data.index[0]
        llat_index = last_index + abs(data.index[0] - first_index)
    elif data.index[-1] - last_index < 0:
        llat_index = data.index[-1]
        flat_index = first_index - abs(data.index[-1] - last_index)
    else:
        flat_index = first_index
        llat_index = last_index

    sample_latency = data.loc[flat_index:llat_index, :]
    # sample_latency.label.unique()
    no_sample_latency =  no_sample_memory.drop(list(range(flat_index,llat_index)))
    no_sample_latency = no_sample_latency[no_sample_latency['label'] == 0]

    # no_sample_latency.drop(['index'], axis = 1, inplace=True)
    # print(no_sample_latency.iloc[: 30, :])





    # concat
    sample_test = pd.concat([sample_cpu, sample_memory,  sample_latency])
    print('test label:', sample_test.label.unique())
    sample_train = no_sample_latency
    # train_label = sample_train.label.values
    # train = sample_train.drop([ 'label'],axis = 1)
    # MinMaxScaler = preprocessing.MinMaxScaler()
    # user_train = pd.DataFrame(MinMaxScaler.fit_transform(train))
    # user_train['label'] = train_label
    # user_train.columns = sample_train.columns


    label = sample_test.label.values
    # test_data = sample_test.drop(['label'],axis = 1)
    # MinMaxScaler = preprocessing.MinMaxScaler()
    # user_test = pd.DataFrame(MinMaxScaler.fit_transform(test_data))
    # user_test['label'] = label
    # user_test.columns = sample_test.columns

    # rca_data = sample_test
    # rca_data['label'] = label

    # random_train = np.random.randint(low=0, high=len(user_train) - 500, size=(1), dtype='int')
    # user_train = user_train.iloc[random_train[0]:random_train[0] + 500, :]
    # pre_label = np.array(test_data['label'].values)
    user_train = sample_train
    user_test = sample_test
    for i in user_train.columns:
        if data_name not in i:
            user_train = user_train.drop(i, axis=1)
    for i in user_test.columns:
        if data_name not in i:
            user_test = user_test.drop(i, axis=1)
    user_train = np.array(user_train)
    test_cpu = np.array(user_test)
    input_cpu = user_train.shape[1]
    values_1 = [v for k, v in dict_values.items() if k == data_name]

    return user_train, test_cpu, label, input_cpu, values_1
