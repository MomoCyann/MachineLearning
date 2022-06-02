import pandas as pd
import numpy as np
from numpy import where
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_samples, silhouette_score
import sklearn.cluster as sc
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt


root = "E:/wakeup/volatility_data/"
datasetpath = "E:/wakeup/volatility_data/dataset/"
datapath = 'E:/wakeup/volatility_data/data/'
eventpath = "E:/wakeup/volatility_data/event/"
# data = pd.read_csv(root + 'allevents.csv', encoding='gbk')
data = pd.read_csv(root + 'allevents_washed.csv', encoding='gbk')
print("删除前")
print(data.shape[0])
# #删除持续时间123的短行为
# data = data[(data['持续时间'].isin([1,2,3])&data['事件类型']=)]

# #删除最大和最小速度都为0的事件
# data.drop(index=data[(data['最大速度']==0)&(data['最小速度']==0)].index,inplace=True)
# #删除极差为0和1的事件，可以看做是平稳驾驶，正常波动。
# data.drop(index=data[data['速度极差']==1].index,inplace=True)
# data.drop(index=data[data['速度极差']==0].index,inplace=True)
#删除持续时间为4S内且极差也在4内的“加速过程”，踩油门如果只让速度变化为4以内的话，可以看做平稳驾驶。至于“刹车事件“确实时间持续较短,
data.drop(index=data[(data['速度极差']<=4)&(data['事件类型']=='accel')].index,inplace=True)
print("删除后")
print(data.shape[0])
print(data['事件类型'].value_counts())
input()
data.sort_values(by='开始时间', inplace=True)
data_new = data.reset_index(drop=True)
data_new.drop('Unnamed: 0', axis=1, inplace=True)
data_new.to_csv(root + 'allevents_washed.csv',
                   encoding='gbk')
print("complete")