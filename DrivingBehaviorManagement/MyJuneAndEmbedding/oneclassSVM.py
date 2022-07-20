import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
root = "D:/RX-105/wakeup/MyJuneAndEmbedding/"
datasetpath = "D:/RX-105/wakeup/MyJuneAndEmbedding/dataset/"
datapath = 'D:/RX-105/wakeup/MyJuneAndEmbedding/data/'
eventpath = "D:/RX-105/wakeup/MyJuneAndEmbedding/event/"

data = pd.read_csv(root + 'allevents_washed_outlier.csv', encoding='gbk')
data2 = pd.read_csv(root + 'allevents_washed_outlier2.csv', encoding='gbk')
data.loc[data2['异常标签2']==-1, ['异常标签']] = -2
outlier = data[data['异常标签']==-2]
mider = data[data['异常标签']==-1]
inner = data[data['异常标签']==1]
midrisk_and_safe = data[(data['异常标签']==1) | (data['异常标签']==-1)]
highrisk_and_midrisk = data[(data['异常标签']==-1) | (data['异常标签']==-2)]

# columns = ['Unnamed: 0', '车辆编号', '开始时间', '结束时间', '持续时间', '加加速度最大值', '加加速度最小值' ,'加加速度极差', '加加速度标准差', '加加速度均值',
#                      '加加速度首尾和', '速度变异系数', '速度时变波动性', '事件类型']
columns = ['Unnamed: 0', '车辆编号', '开始时间', '结束时间', '持续时间', '加加速度最大值', '加加速度最小值' ,'加加速度极差', '加加速度标准差', '加加速度均值',
                     '加加速度首尾和', '事件类型']
outlier.drop(labels=columns, axis=1, inplace=True)
mider.drop(labels=columns, axis=1, inplace=True)
inner.drop(labels=columns, axis=1, inplace=True)

midrisk_and_safe.drop(labels=columns, axis=1, inplace=True)
highrisk_and_midrisk.drop(labels=columns, axis=1, inplace=True)

outlier.reset_index(drop=True,inplace=True)
mider.reset_index(drop=True,inplace=True)
inner.reset_index(drop=True,inplace=True)
highrisk_and_midrisk.reset_index(drop=True,inplace=True)
midrisk_and_safe.reset_index(drop=True,inplace=True)

midrisk_and_safe.loc[midrisk_and_safe['异常标签']==-1, ['异常标签']] = 1
outlier.loc[outlier['异常标签']==-2, ['异常标签']] = -1
highrisk_and_midrisk.loc[highrisk_and_midrisk['异常标签']==-2, ['异常标签']] = -1
def load_data(raw_data):
    label = raw_data.iloc[:, -1:]
    data = raw_data.iloc[:, :-1]
    return data, label
data_midrisk_and_safe, label_midrisk_and_safe = load_data(midrisk_and_safe)
data_highrisk, label_highrisk = load_data(outlier)
data_safe, label_safe = load_data(inner)
data_highrisk_and_midrisk, label_highrisk_and_midrisk = load_data(highrisk_and_midrisk)

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
def minmax(trainset):
    """归一化"""
    scaler = MinMaxScaler()
    trainset_new = scaler.fit_transform(trainset)
    return trainset_new
def standard(trainset):
    """标准化"""
    scaler = StandardScaler()
    trainset_new = scaler.fit_transform(trainset)
    return trainset_new

data_midrisk_and_safe = minmax(data_midrisk_and_safe)
data_highrisk = minmax(data_highrisk)
data_safe = minmax(data_safe)
data_highrisk_and_midrisk = minmax(data_highrisk_and_midrisk)

X_highrisk = data_highrisk
X_midrisk_and_safe = data_midrisk_and_safe
y_highrisk = label_highrisk.values
y_midrisk_and_safe = label_midrisk_and_safe.values
X_safe = data_safe
y_safe = label_safe.values
X_highrisk_and_midrisk = data_highrisk_and_midrisk
y_highrisk_and_midrisk = label_highrisk_and_midrisk.values


from sklearn.svm import OneClassSVM
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import accuracy_score
model = OneClassSVM(kernel='rbf', gamma='auto', nu=0.2)
model.fit(X_safe)
# 训练模型
# 预测测试集
labels_pred = model.predict(X_highrisk)
# 计算指标
#f1_score_fold = f1_score(y_test_fold, test_labels_predminmax(), average="weighted")
acc = accuracy_score(y_highrisk, labels_pred)
print('平均准确率为：')
print(acc)