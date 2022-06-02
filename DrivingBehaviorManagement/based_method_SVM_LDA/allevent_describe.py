# coding:utf-8
import pandas as pd
root = "E:/wakeup/"
datasetpath = "E:/wakeup/dataset/"
datapath = 'E:/wakeup/data/'
eventpath = "E:/wakeup/event/"

data = pd.read_csv(root + 'allevents_washed_outlier.csv', encoding='gbk')
data2 = pd.read_csv(root + 'allevents_washed_outlier2.csv', encoding='gbk')
# data.loc[data2['异常标签2']==-1, ['异常标签']] = -2
# outlier = data[data['异常标签']==-2]
mider = data[data['异常标签']==-1]
inner = data[data['异常标签']==1]
print("正常行为的速度标准差的平均")
print(inner.mean())
print(mider.mean())
# print(outlier.mean())
# print(data['异常标签'].value_counts())
# accel = data[data['事件类型']=='accel']
# brake = data[data['事件类型']=='brake']
# turn = data[data['事件类型']=='turn']
# print(accel['异常标签'].value_counts())
# print(brake['异常标签'].value_counts())
# print(turn['异常标签'].value_counts())