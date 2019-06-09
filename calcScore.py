from more_itertools import consecutive_groups
import pandas as pd
import numpy as np
from pprint import pprint


def calcScore(anomalyList):
    label = ["2015-01-24 20:30:00.000000",
             "2015-01-29 03:30:00.000000"]
    label = pd.to_datetime(label)
    time = pd.read_csv('./data/nyc_taxi_test.csv')['1'][-(1000 - 50 - 1):].values
    time = pd.to_datetime(time.tolist())

    time_idx = []
    for tid, t in enumerate(time):
        if label[0] <= t <= label[1]:
            time_idx.append(tid)
    ground_truth = [list(g) for g in consecutive_groups(time_idx)]
    ground_truth = [[g[0], g[-1]] for g in ground_truth]
    # 初始化统计
    # every point in a window is a false negative
    # every point outside a window is a true negtive

    anom = {}
    anom['false_positives'] = 0
    anom['false_negatives'] = 0
    anom['true_positives'] = 0
    anom['fp_sequences'] = []
    anom['tp_sequences'] = []
    anom['ground_truth'] = ground_truth
    anom['num_anoms'] = len(ground_truth)

    startAndEnds = [list(g) for g in consecutive_groups(anomalyList)]
    startAndEnds = [[g[0], g[-1]] for g in startAndEnds]
    error_sequence = [(group[0], group[1]) for group in startAndEnds if group[0] != group[1]]
    if len(error_sequence) > 0:
        matched_error_sequence_test = []
        for startEnd in error_sequence:
            valid = False
            for idx, a in enumerate(ground_truth):
                condition1 = (startEnd[0] >= a[0] and startEnd[0] <= a[1])  # 异常区间的开始点在标准区间内
                condition2 = (startEnd[1] >= a[0] and startEnd[1] <= a[1])  # 异常区间在结束点在标准区间内
                condition3 = (startEnd[0] <= a[0] and startEnd[1] >= a[1])  # 标准区间在异常区间内
                condition4 = (a[0] <= startEnd[0] and a[1] >= startEnd[1])  # 异常区间在标准区间内
                if condition1 or condition2 or condition3 or condition4:
                    anom['tp_sequences'].append(startEnd)
                    valid = True

                    if idx not in matched_error_sequence_test:
                        anom['true_positives'] += 1
                        matched_error_sequence_test.append(idx)

            if valid == False:
                anom['false_positives'] += 1
                anom['fp_sequences'].append([startEnd[0], startEnd[1]])

        anom['false_negatives'] += len(ground_truth) - len(matched_error_sequence_test)

    else:
        anom['false_negatives'] += len(ground_truth)

    anom['RECALL'] = anom['true_positives'] / (anom['true_positives'] + anom['false_negatives'])
    anom['PRECISION'] = anom['true_positives'] / (anom['true_positives'] + anom['false_positives'])
    anom['F1-SCORE'] = (2 * anom['RECALL'] * anom['PRECISION']) / (anom['RECALL'] + anom['PRECISION'])

    return anom

if __name__ == '__main__':
    name = ['attn','LSTMEncDec','conv1d','conv1dLSTM']
    for n in name:
        anomaly_idx = np.load('./result/anomaly_idx_'+ n +'.npy')
        print('>>> [Model]:%s'%n)
        pprint(calcScore(anomaly_idx))
        print('----------------------')