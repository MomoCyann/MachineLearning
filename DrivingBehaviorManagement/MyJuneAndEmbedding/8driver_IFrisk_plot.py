import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
root = "D:/RX-105/wakeup/MyJuneAndEmbedding/8car/"
datasetpath = "D:/RX-105/wakeup/MyJuneAndEmbedding/dataset/"
datapath = 'D:/RX-105/wakeup/MyJuneAndEmbedding/data/'
eventpath = "D:/RX-105/wakeup/MyJuneAndEmbedding/event/"

data = pd.read_csv(root + 'allevents_washed_outlier.csv', encoding='gbk')
data2 = pd.read_csv(root + 'allevents_washed_outlier2.csv', encoding='gbk')
data.loc[data2['异常标签2']==-1, ['异常标签']] = -2
outlier = data[data['异常标签']==-2]
mider = data[data['异常标签']==-1]
inner = data[data['异常标签']==1]

highrisk = data[data['异常标签']==-2]['车辆编号'].value_counts().to_dict()
midrisk = data[data['异常标签']==-1]['车辆编号'].value_counts().to_dict()
car_no = list(highrisk.keys())
highrisk_count = list(highrisk.values())
midrisk_count = list(midrisk.values())
print(car_no)

#plot
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 添加地名坐标
# x=np.arange(8)
# plt.bar(x, height=highrisk_count, width=0.5, color='salmon', tick_label=car_no)
# #添加数据标签
# for a, b in zip(x, highrisk_count):
#     plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=10)
# #坐标轴
# x_name = '车辆编号'
# y_name = '高风险行为次数'
# title = '8辆车高风险行为次数'
# plt.gca().set(xlabel=x_name, ylabel=y_name)

#中风险
# 添加地名坐标
x=np.arange(8)
plt.bar(x, height=midrisk_count, width=0.5, color='orange', tick_label=car_no)
#添加数据标签
for a, b in zip(x, midrisk_count):
    plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=10)
#坐标轴
x_name = '车辆编号'
y_name = '中风险行为次数'
title = '8辆车中风险行为次数'
plt.gca().set(xlabel=x_name, ylabel=y_name)

plt.legend()
plt.show()