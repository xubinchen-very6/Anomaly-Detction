import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(file_name, sequence_length=10, split=1,train=True):
    df = pd.read_csv(file_name, sep=',', usecols=[3])
    data_all = np.array(df).astype(float)
    data_all = scaler(data_all)

    data = []

    for i in range(len(data_all) - sequence_length - 1):
        data.append(data_all[i: i + sequence_length + 1])
    reshaped_data = np.array(data).astype('float64')
    np.random.seed(0)
    if train:
        np.random.shuffle(reshaped_data)
    x = reshaped_data[:, :-1]
    y = reshaped_data[:, -1]
    split_boundary = int(reshaped_data.shape[0] * split)
    train_x = x[: split_boundary]
    train_y = y[: split_boundary]

    return train_x, train_y

def scaler(data):
    max_num = np.array([39197.])
    min_num = np.array([8.])
    mean = (max_num - min_num)/2
    interval = (max_num + min_num)/2
    return (data - mean) / interval

if __name__ == '__main__':

    df = pd.read_csv('/Users/macbookair/Downloads/nyc_taxi_train.csv', sep=',', usecols=[3])
    print(df)
    # train_x,train_y,scaler = load_data('/Users/macbookair/Downloads/nyc_taxi_train.csv',sequence_length=10,split=0.8)
    # print('=======%nyc_taxi.csv=======')
    # print('train_x.shape >>>', np.shape(train_x))
    # print('train_y.shape >>>', np.shape(train_y))
    # print('test_x.shape >>>', np.shape(test_x))
    # print('test_y.shape >>>', np.shape(test_y))
