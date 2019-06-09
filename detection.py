import numpy as np
import pandas as pd
from more_itertools import consecutive_groups
from sklearn.metrics import mean_squared_error,r2_score
from calcScore import calcScore
from pprint import pprint

def denorm(path,window):
    root = './result/'
    file_path = root+path
    predict = np.load(file_path)

    data = pd.read_csv('./data/nyc_taxi.csv')
    data = data['value'].values
    min_num = np.array([min(data)])
    max_num = np.array([max(data)])
    interval = (max_num-min_num) / 2
    mean = (max_num+min_num) / 2
    predict = predict * interval + mean

    test = pd.read_csv('./data/nyc_taxi_test.csv')
    index = test.shape[0] - window - 1
    ground_truth = test.values[::,3][-index:]

    return predict,ground_truth

def find_epsilon(e_s,r,limit):

    mean,std,temp = np.mean(e_s),np.std(e_s),0
    threshold = limit

    for sigma in np.arange(0.5,threshold,0.1):
        epsilon = mean + std* sigma
        '''
        记录正常点的value和index
        记录异常点的index
        '''
        normal_val,normal_idx,anomaly_idx = [],[],[]

        for idx,error in enumerate(e_s):
            # 小于阈值不认为是异常
            if error < epsilon:
                normal_val.append(error);normal_idx.append(idx)
            # 大于等于阈值定异常点
            if error > epsilon:
                for area in range(0,r):
                    if idx + area not in anomaly_idx and not idx + area >= len(e_s):
                        anomaly_idx.append(idx+area)
                    if idx - area not in anomaly_idx and not idx - area <= 0:
                        anomaly_idx.append(idx+area)
        # 非空异常集合
        if len(anomaly_idx) > 0:
            anomaly_idx = sorted(list(set(anomaly_idx)))
            errorGroups = [list(errorGroup) for errorGroup in consecutive_groups(anomaly_idx)]
            startAndEnds = [(group[0],group[-1]) for group in errorGroups if not group[0] == group[1]]

            mean_decrease = (mean - np.mean(normal_val)) / len(e_s)
            std_decrease = (std - np.std(normal_val)) / len(e_s)

            metric = (mean_decrease + std_decrease) / (len(startAndEnds)**2 + len(anomaly_idx))

            '''
            1.是否需要进行Error window个数制约
            2.异常值数量制约
            3.window limit如何进行制约限制
            '''
            window_limit = 5


            if metric > temp and len(anomaly_idx) < len(e_s)*0.5 and len(startAndEnds) <= window_limit:
                threshold = sigma
                temp = metric

    return threshold

def compare_to_epsilon(e_s,epsilon,inter_range,y_test_std,std,r,window):
    '''
    error与epsilon的关系判断是否outlier; window:处理的error_window个数
    '''
    anomaly_idx,error_seq,non_anomaly_max = [],[],0

    '''
    std : 193.29
    mean: 1706.89
    y_test std: 7457.43
    0.05y_test_std: 372.871
    0.05inter_range: 1028.31
    '''
    #没异常的情况
    # if not (std > 0.05 * y_test_std) or max(e_s) > 0.05 * inter_range or max(e_s) <= 0.05:
    #     return error_seq,anomaly_idx,non_anomaly_max

    window_size = 50
    num_to_ignore = window_size * 2
    for idx in range(len(e_s)):
        anom_flag = True
        if e_s[idx] <= epsilon or e_s[idx] < 0.1 * inter_range:
            anom_flag = False

        if anom_flag:
            for area in range(0,r):
                '''
                1.当前点未存
                2.当前点没出边界
                3.到最后一个窗口没找到异常
                '''
                if idx + area not in anomaly_idx and 0 < idx + area < len(e_s) and (idx + area >= len(e_s) - window_size or window == 0):
                    anomaly_idx.append(idx + area)
                if idx - area not in anomaly_idx and (idx - area >= len(e_s) - window_size or window == 0):
                    if not (window == 0 and idx - area < num_to_ignore):
                        anomaly_idx.append(idx - area)

    # 非异常的最大值
    min_anom = min(np.array(e_s)[anomaly_idx])
    for idx in range(0,len(e_s)):
        temp = e_s[idx]
        if temp < min_anom and temp > non_anomaly_max:
            non_anomaly_max = temp

    anomaly_idx = sorted(list(set(anomaly_idx)))
    startAndEnds = [list(startAndEnd) for startAndEnd in consecutive_groups(anomaly_idx)]
    error_seq = [(group[0],group[-1]) for group in startAndEnds if group[0] != group[1]]

    return error_seq,anomaly_idx,non_anomaly_max

def prune_anoms(error_seq,e_s,non_anomaly_max,anomaly_idx):
    error_seq_max,e_s_max,e_s_max_idx = [],[],[]
    for seq in error_seq:
        if len(e_s[seq[0]:seq[1]]) > 0:
            start,end = seq[0],seq[1]
            error_seq_max.append(max(e_s[start:end]))
            e_s_max.append(max(e_s[start:end]))

    e_s_max.append(non_anomaly_max)
    e_s_max.sort(reverse = True)

    remove_idx = []
    decay = 0.13

    for idx in range(0,len(e_s_max)):
        if idx+1 < len(e_s_max):
            if (e_s_max[idx] - e_s_max[idx+1]) / e_s_max[idx] < decay:
                remove_idx.append(error_seq_max.index(e_s_max[idx]))
            else:
                remove_idx = []

    for idx in sorted(remove_idx,reverse=True):
        del error_seq[idx]

    pruned_idx = []
    for idx in anomaly_idx:
        keep_anomaly = False
        for seq in error_seq:
            if idx >= seq[0] and idx<=seq[1]:
                keep_anomaly = True

        if keep_anomaly == True:
            pruned_idx.append(idx)

    return pruned_idx


def get_anomalies(e_s,y_test,z,window,r):
    perc_high,perc_low = np.percentile(y_test,[90,10])
    inter_range = perc_high - perc_low

    mean = np.mean(e_s)
    std = np.std(e_s)

    y_test_std = np.std(y_test)

    e_s_inv = mean + mean - e_s
    z_inv = find_epsilon(e_s_inv,r,limit=12)

    epsilon = mean + float(z) * std
    epsilon_inv = mean + float(z_inv) * std

    # 找大于epsilon的error
    error_sequence,anomaly_idx,non_anom_max = compare_to_epsilon(e_s,epsilon,inter_range,y_test_std,std,r,window)
    error_sequence_inv, anomaly_idx_inv, non_anom_max_inv = compare_to_epsilon(e_s_inv, epsilon_inv, inter_range, y_test_std, std, r, window)

    if len(error_sequence) > 0:
        anomaly_idx = prune_anoms(error_sequence,e_s,non_anom_max,anomaly_idx)
    if len(error_sequence_inv) > 0:
        anomaly_idx_inv = prune_anoms(error_sequence_inv,e_s_inv,non_anom_max_inv,anomaly_idx_inv)
    # anomaly_idx = list(set(anomaly_idx))
    anomaly_idx = list(set(anomaly_idx+anomaly_idx_inv))
    return anomaly_idx

if __name__ == '__main__':
    data = pd.read_csv('./result/results.csv')
    ground_truth = data['ground_truth']
    lstmEncDec = data['LSTM-Enc-Dnc']
    conv1d = data['CONV1D-DENSE']
    conv1d_lstm = data['CONV1D-LSTM']
    attnLSTM = data['AttnLSTM']
    time = pd.read_csv('./data/nyc_taxi_test.csv')['1'][-(1000 - 50 - 1):].values
    time = pd.to_datetime(time.tolist())
    print('>>> LSTM-Enc-Dnc %.3f(RMSE) %.3f(R^2)' % (
    np.sqrt(mean_squared_error(ground_truth, lstmEncDec)), r2_score(ground_truth, lstmEncDec)))
    print('>>> CONV1D-DENSE %.3f(RMSE) %.3f(R^2)' % (
    np.sqrt(mean_squared_error(ground_truth, conv1d)), r2_score(ground_truth, conv1d)))
    print('>>> CONV1D-LSTM  %.3f(RMSE) %.3f(R^2)' % (
    np.sqrt(mean_squared_error(ground_truth, conv1d_lstm)), r2_score(ground_truth, conv1d_lstm)))
    print('>>> AttnLSTM     %.3f(RMSE) %.3f(R^2)' % (
    np.sqrt(mean_squared_error(ground_truth, attnLSTM)), r2_score(ground_truth, attnLSTM)))

    name = ['attn','LSTMEncDec','conv1d','conv1dLSTM']
    preds = [attnLSTM,lstmEncDec,conv1d,conv1d_lstm]
    for n,pred in zip(name,preds):
        e = np.abs(ground_truth-pred)
        e_s = pd.DataFrame(e).ewm(span=40).mean().values.flatten().tolist()
        # e_s = pd.DataFrame(e_s).ewm(span=60).mean().values.flatten().tolist()
        threshold = find_epsilon(e_s,70,10)
        anomaly_idx = get_anomalies(e_s,ground_truth,threshold,0,70)
        np.save('./result/anomaly_idx_'+ n +'.npy',np.array(anomaly_idx))

        print('>>> [Model]: %s'%n)
        pprint(calcScore(anomaly_idx))
        print('-----------------------')

