import pandas as pd
import numpy as np
from numpy import where
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_samples, silhouette_score
import sklearn.cluster as sc
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt
# from pycluster import KMedoids
plt.rcParams['font.family'] = 'simhei'
root = "E:/wakeup/volatility_data/"
datasetpath = "E:/wakeup/volatility_data/dataset/"
datapath = 'E:/wakeup/volatility_data/data/'
eventpath = "E:/wakeup/volatility_data/event/"

data_row = pd.read_csv(root + 'allevents_washed.csv', encoding='gbk')
data_accel_row=data_row[data_row['事件类型']=='accel']
data_brake_row=data_row[data_row['事件类型']=='brake']
data_turn_row=data_row[data_row['事件类型']=='turn']
data_accel_row = data_accel_row.reset_index(drop=True)
data_brake_row = data_brake_row.reset_index(drop=True)
data_turn_row = data_turn_row.reset_index(drop=True)
data = data_row[['最大速度',
                '最小速度',
                '速度极差',
                '速度标准差',
                '速度均值',
                '临界速度',
                '最大加速度',
                '最小加速度',
                '加速度极差',
                '加速度标准差',
                '加速度均值',
                '首尾加速度和',
                ]]
data_accel = data_accel_row[['最大速度',
                '最小速度',
                '速度极差',
                '速度标准差',
                '速度均值',
                '临界速度',
                '最大加速度',
                '最小加速度',
                '加速度极差',
                '加速度标准差',
                '加速度均值',
                '首尾加速度和',
                ]]
data_brake = data_brake_row[['最大速度',
                '最小速度',
                '速度极差',
                '速度标准差',
                '速度均值',
                '临界速度',
                '最大加速度',
                '最小加速度',
                '加速度极差',
                '加速度标准差',
                '加速度均值',
                '首尾加速度和',
                ]]
data_turn = data_turn_row[['最大速度',
                '最小速度',
                '速度极差',
                '速度标准差',
                '速度均值',
                '临界速度',
                '最大加速度',
                '最小加速度',
                '加速度极差',
                '加速度标准差',
                '加速度均值',
                '首尾加速度和',
                ]]

# #标准化
# std = StandardScaler()
# data_std = std.fit_transform(data.values)
# #训练
# model = KMeans(n_clusters=2)
# model.fit(data_std)
# # 简单打印结果
# r1 = pd.Series(model.labels_).value_counts()  # 统计各个类别的数目
# r2 = pd.DataFrame(model.cluster_centers_)  # 找出聚类中心
# # 所有簇中心坐标值中最大值和最小值
# max = r2.values.max()
# min = r2.values.min()
#
# r = pd.concat([r2, r1], axis=1)  # 横向连接（0是纵向），得到聚类中心对应的类别下的数目
# r.columns = list(data.columns) + [u'类别数目']  # 重命名表头
#
# #详细输出原始数据及其类别
# r = pd.concat([data_row, pd.Series(model.labels_, index = data.index)], axis = 1)  #详细输出每个样本对应的类别
# r.columns = list(data_row.columns) + [u'聚类类别'] #重命名表头
# r.to_csv(root + 'allevents_cluster_label.csv', encoding='gbk') #保存结果
#



#孤立森林
from sklearn.ensemble import IsolationForest

model_isof = IsolationForest(contamination=0.10, max_features=12,n_estimators=200)
# model_isof_accel = IsolationForest(contamination=0.10, max_features=12,n_estimators=200)
# model_isof_brake = IsolationForest(contamination=0.10, max_features=12,n_estimators=200)
# model_isof_turn = IsolationForest(contamination=0.2, max_features=12,n_estimators=50)
outlier_label = model_isof.fit_predict(data) #原本的
# outlier_label_accel = model_isof_accel.fit_predict(data_accel)
# outlier_label_brake = model_isof_brake.fit_predict(data_brake)
# outlier_label_turn = model_isof_turn.fit_predict(data_turn)
outlier_pd = pd.DataFrame(outlier_label, columns=[u'异常标签'])
# outlier_pd_accel = pd.DataFrame(outlier_label_accel, columns=[u'异常标签'])
# outlier_pd_brake = pd.DataFrame(outlier_label_brake, columns=[u'异常标签'])
# outlier_pd_turn = pd.DataFrame(outlier_label_turn, columns=[u'异常标签'])
data_merge = pd.concat((data_row, outlier_pd), axis=1)
# data_merge_accel = pd.concat((data_accel_row, outlier_pd_accel), axis=1)
# data_merge_brake = pd.concat((data_brake_row, outlier_pd_brake), axis=1)
# data_merge_turn = pd.concat((data_turn_row, outlier_pd_turn), axis=1)
print(data_merge['异常标签'].value_counts())
# print(data_merge_accel['异常标签'].value_counts())
# print(data_merge_brake['异常标签'].value_counts())
# print(data_merge_turn['异常标签'].value_counts())

#用TSNE进行数据降维并展示聚类结果
from sklearn.manifold import TSNE
# tsne = TSNE()
# tsne.fit_transform(data) #进行数据降维,并返回结果
# tsne = pd.DataFrame(tsne.embedding_, index = data.index) #转换数据格式

import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
#
# # #不同类别用不同颜色和样式绘图
# # d = tsne[r[u'聚类类别'] == 0]     #找出聚类类别为0的数据对应的降维结果
# # plt.plot(d[0], d[1], 'r.')
# # d = tsne[r[u'聚类类别'] == 1]
# # plt.plot(d[0], d[1], 'go')
# # d = tsne[r[u'聚类类别'] == 2]
# # plt.plot(d[0], d[1], 'b*')
# # plt.show()
#
# #不同类别用不同颜色和样式绘图
# d = tsne[data_merge['outlier_label'] == -1]     #找出聚类类别为0的数据对应的降维结果
# plt.plot(d[0], d[1], 'r.')
# d = tsne[data_merge['outlier_label'] == 1]
# plt.plot(d[0], d[1], 'b*')
# plt.show()

def plot_embedding_2d(X, title=None):
    #坐标缩放到[0,1]区间
    x_min, x_max = np.min(X,axis=0), np.max(X,axis=0)
    X = (X - x_min) / (x_max - x_min)

    #降维后的坐标为（X[i, 0], X[i, 1]），在该位置画出对应的digits
    label = data_merge['outlier_label']
    #找出不同类别数据对应的降维结果
    plt.scatter(X[:,0], X[:,1], c=label, cmap=plt.cm.Spectral, alpha=0.5)
    if title is not None:
        plt.title(title)
    plt.show()

#将降维后的数据可视化,3维
def plot_embedding_3d(X, title=None):
    #坐标缩放到[0,1]区间
    x_min, x_max = np.min(X,axis=0), np.max(X,axis=0)
    X = (X - x_min) / (x_max - x_min)

    #降维后的坐标为（X[i, 0], X[i, 1],X[i,2]），在该位置画出对应的digits
    ax = plt.subplot(111, projection='3d')
    outlier = X[data_merge['异常标签'] == -1]
    iner = X[data_merge['异常标签'] == 1]
    ax.scatter(outlier[:,0], outlier[:,1], outlier[:,2], c="deepskyblue", alpha=0.5)
    ax.scatter(iner[:,0], iner[:,1], iner[:,2], c="salmon", alpha=0.1)


    ax.set_zlabel('Z')
    ax.set_ylabel('Y')
    ax.set_xlabel('X')


    if title is not None:
        plt.title(title)
    plt.show()

def plot_embedding_3d_k(X, title=None):
    #坐标缩放到[0,1]区间
    x_min, x_max = np.min(X,axis=0), np.max(X,axis=0)
    X = (X - x_min) / (x_max - x_min)

    #降维后的坐标为（X[i, 0], X[i, 1],X[i,2]），在该位置画出对应的digits
    ax = plt.subplot(111, projection='3d')
    one = X[data_outlier_merge[u'cluster标签'] == 0]
    two = X[data_outlier_merge[u'cluster标签'] != 0]
    ax.scatter(one[:,0], one[:,1], one[:,2], c="orange", alpha=0.5, label='0类')
    ax.scatter(two[:,0], two[:,1], two[:,2], c="salmon", alpha=0.5, label='1类')
    ax.legend()

    ax.set_zlabel('Z')
    ax.set_ylabel('Y')
    ax.set_xlabel('X')


    if title is not None:
        plt.title(title)
    plt.show()

from time import time
print("Computing t-SNE embedding")
# tsne = TSNE(n_components=3, init='pca', random_state=0)
# t0 = time()
# X_tsne = tsne.fit_transform(data)
# X_tsne_accel = tsne.fit_transform(data_accel)
# X_tsne_brake = tsne.fit_transform(data_brake)
# X_tsne_turn = tsne.fit_transform(data_turn)
#plot_embedding_2d(X_tsne[:,0:2],"t-SNE 2D")
# 3d图
#plot_embedding_3d(X_tsne,"t-SNE 3D (time %.2fs)" %(time() - t0))
# #plot_embedding_3d(X_tsne_accel,data_merge_accel,"t-SNE 3D (time %.2fs)" %(time() - t0))
# #plot_embedding_3d(X_tsne_brake,data_merge_brake,"t-SNE 3D (time %.2fs)" %(time() - t0))
# plot_embedding_3d(X_tsne_turn,data_merge_turn,"t-SNE 3D (time %.2fs)" %(time() - t0))

# data_merge.drop('Unnamed: 0', axis=1, inplace=True)
# data_merge_accel.drop('Unnamed: 0', axis=1, inplace=True)
# data_merge_brake.drop('Unnamed: 0', axis=1, inplace=True)
# data_merge_turn.drop('Unnamed: 0', axis=1, inplace=True)

# data_merge.to_csv(root + 'allevents_washed_outlier.csv',
#                    encoding='gbk')
# data_merge_accel.to_csv(root + 'allevents_washed_outlier_accel.csv',
#                    encoding='gbk')
# data_merge_brake.to_csv(root + 'allevents_washed_outlier_brake.csv',
#                    encoding='gbk')
# data_merge_turn.to_csv(root + 'allevents_washed_outlier_turn.csv',
#                    encoding='gbk')
print("complete")

#聚类危险行为
data_outlier_row = pd.read_csv(root + 'allevents_washed_outlier.csv', encoding='gbk')
# data_outlier_row_accel = pd.read_csv(root + 'allevents_washed_outlier_accel.csv', encoding='gbk')
# data_outlier_row_brake = pd.read_csv(root + 'allevents_washed_outlier_brake.csv', encoding='gbk')
# data_outlier_row_turn = pd.read_csv(root + 'allevents_washed_outlier_turn.csv', encoding='gbk')
data_outlier_row = data_outlier_row[data_outlier_row['异常标签'] == -1]
# data_outlier_row_accel = data_outlier_row_accel[data_outlier_row_accel['异常标签'] == -1]
# data_outlier_row_brake = data_outlier_row_brake[data_outlier_row_brake['异常标签'] == -1]
# data_outlier_row_turn = data_outlier_row_turn[data_outlier_row_turn['异常标签'] == -1]
data_outlier = data_outlier_row[['最大速度',
                            '最小速度',
                            '速度极差',
                            '速度标准差',
                            '速度均值',
                            '临界速度',
                            '最大加速度',
                            '最小加速度',
                            '加速度极差',
                            '加速度标准差',
                            '加速度均值',
                            '首尾加速度和',
         ]]
# data_outlier_accel = data_outlier_row_accel[['最大速度',
#                             '最小速度',
#                             '速度极差',
#                             '速度标准差',
#                             '速度均值',
#                             '临界速度',
#                             '最大加速度',
#                             '最小加速度',
#                             '加速度极差',
#                             '加速度标准差',
#                             '加速度均值',
#                             '首尾加速度和',
#          ]]
# data_outlier_brake = data_outlier_row_brake[['最大速度',
#                             '最小速度',
#                             '速度极差',
#                             '速度标准差',
#                             '速度均值',
#                             '临界速度',
#                             '最大加速度',
#                             '最小加速度',
#                             '加速度极差',
#                             '加速度标准差',
#                             '加速度均值',
#                             '首尾加速度和',
#          ]]
# data_outlier_turn = data_outlier_row_turn[['最大速度',
#                             '最小速度',
#                             '速度极差',
#                             '速度标准差',
#                             '速度均值',
#                             '临界速度',
#                             '最大加速度',
#                             '最小加速度',
#                             '加速度极差',
#                             '加速度标准差',
#                             '加速度均值',
#                             '首尾加速度和',
#          ]]

data_outlier_row = data_outlier_row.reset_index(drop=True)
data_outlier = data_outlier.reset_index(drop=True)
# cmodel = KMeans(n_clusters=2)
cmodel = DBSCAN(eps = 5, min_samples = 15)
cmodel.fit(data_outlier)
outlier_cluster = pd.DataFrame(cmodel.labels_, columns=[u'cluster标签'])
data_outlier_merge = pd.concat((data_outlier_row, outlier_cluster), axis=1)
print(pd.Series(cmodel.labels_).value_counts())
# 异常聚类tsne降维可视化
from time import time
print("Computing t-SNE embedding")
tsne = TSNE(n_components=3, init='pca', random_state=0)
t0 = time()
X_tsne = tsne.fit_transform(data_outlier)
#plot_embedding_2d(X_tsne[:,0:2],"t-SNE 2D")
plot_embedding_3d_k(X_tsne,"t-SNE 3D (time %.2fs)" %(time() - t0))

# input()
# data_outlier_merge.loc[data_outlier_merge['cluster标签'] != 0, 'cluster标签'] = -1
# # data_outlier_merge.set_index('Unnamed：0')
# # data_outlier_merge.drop('Unnamed: 0', axis=1, inplace=True)
# data_outlier_merge.to_csv(root + 'allevents_washed_outlier_cluster.csv',index=False,
#                    encoding='gbk')
# print("complete")


