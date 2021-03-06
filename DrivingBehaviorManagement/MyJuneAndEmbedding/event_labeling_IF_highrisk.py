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
root = "D:/RX-105/wakeup/MyJuneAndEmbedding/8car/"
datasetpath = "D:/RX-105/wakeup/MyJuneAndEmbedding/dataset/"
datapath = 'D:/RX-105/wakeup/MyJuneAndEmbedding/data/'
eventpath = "D:/RX-105/wakeup/MyJuneAndEmbedding/event/"

#data_row = pd.read_csv(root + 'allevents_washed.csv', encoding='gbk')
data_row = pd.read_csv(root + 'allevents_washed_outlier.csv', encoding='gbk')
#之前已经分好了一个outlier，好像是5%的，不是进一步处理，只是再重新生成一个异常标签2，结合异常标签1，筛选
data_accel_row=data_row[data_row['事件类型']=='accel']
data_brake_row=data_row[data_row['事件类型']=='brake']
data_turn_row=data_row[data_row['事件类型']=='turn']
data_accel_row = data_accel_row.reset_index(drop=True)
data_brake_row = data_brake_row.reset_index(drop=True)
data_turn_row = data_turn_row.reset_index(drop=True)

def get_attribute(data_row):
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
    return data


data = get_attribute(data_row)
data_accel = get_attribute(data_accel_row)
data_brake = get_attribute(data_brake_row)
data_turn = get_attribute(data_turn_row)


def minmax(data):
    """归一化"""
    scaler = MinMaxScaler()
    data_new = scaler.fit_transform(data)
    return data_new


#孤立森林
from sklearn.ensemble import IsolationForest


def get_isolation(data_row, data, model):
    outlier_label = model.fit_predict(data)
    outlier_pd = pd.DataFrame(outlier_label, columns=[u'异常标签2'])
    data_merge = pd.concat((data_row, outlier_pd), axis=1)
    print(data_merge['异常标签2'].value_counts())
    return data_merge

model_isof = IsolationForest(contamination=0.01, max_features=5,n_estimators=250, max_samples=125000, random_state=1)
data_merge = get_isolation(data_row, data, model_isof)
# model_isof_accel = IsolationForest(contamination=0.10, max_features=12,n_estimators=200)
# model_isof_brake = IsolationForest(contamination=0.10, max_features=12,n_estimators=200)
# model_isof_turn = IsolationForest(contamination=0.2, max_features=12,n_estimators=50)

# outlier_label_accel = model_isof_accel.fit_predict(data_accel)
# outlier_label_brake = model_isof_brake.fit_predict(data_brake)
# outlier_label_turn = model_isof_turn.fit_predict(data_turn)

# outlier_pd_accel = pd.DataFrame(outlier_label_accel, columns=[u'异常标签'])
# outlier_pd_brake = pd.DataFrame(outlier_label_brake, columns=[u'异常标签'])
# outlier_pd_turn = pd.DataFrame(outlier_label_turn, columns=[u'异常标签'])

# data_merge_accel = pd.concat((data_accel_row, outlier_pd_accel), axis=1)
# data_merge_brake = pd.concat((data_brake_row, outlier_pd_brake), axis=1)
# data_merge_turn = pd.concat((data_turn_row, outlier_pd_turn), axis=1)

# print(data_merge_accel['异常标签'].value_counts())
# print(data_merge_brake['异常标签'].value_counts())
# print(data_merge_turn['异常标签'].value_counts())

#用TSNE进行数据降维并展示聚类结果
from sklearn.manifold import TSNE
# tsne = TSNE()
# tsne.fit_transform(data) #进行数据降维,并返回结果
# tsne = pd.DataFrame(tsne.embedding_, index = data.index) #转换数据格式

import matplotlib.pyplot as plt


def plot_embedding_2d(X, title=None):
    #坐标缩放到[0,1]区间
    x_min, x_max = np.min(X,axis=0), np.max(X,axis=0)
    X = (X - x_min) / (x_max - x_min)

    #降维后的坐标为（X[i, 0], X[i, 1]），在该位置画出对应的digits
    # label = data_merge['outlier_label']
    outlier = X[data_merge['异常标签2'] == -1]
    iner = X[data_merge['异常标签2'] == 1]
    #找出不同类别数据对应的降维结果
    plt.scatter(outlier[:,0], outlier[:,1], c="deepskyblue", alpha=0.5)
    plt.scatter(iner[:,0], iner[:,1], c="salmon", alpha=0.1)
    if title is not None:
        plt.title(title)
    plt.show()


# 将降维后的数据可视化,3维
def plot_embedding_3d(X, title=None):
    # 坐标缩放到[0,1]区间
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X = (X - x_min) / (x_max - x_min)

    # 降维后的坐标为（X[i, 0], X[i, 1],X[i,2]），在该位置画出对应的digits
    ax = plt.subplot(111, projection='3d')
    outlier = X[data_merge['异常标签2'] == -1]
    iner = X[data_merge['异常标签2'] == 1]
    ax.scatter(outlier[:,0], outlier[:,1], outlier[:,2], c="deepskyblue", alpha=0.5)
    ax.scatter(iner[:,0], iner[:,1], iner[:,2], c="salmon", alpha=0.1)
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

# # # X_tsne_accel = tsne.fit_transform(data_accel)
# # # X_tsne_brake = tsne.fit_transform(data_brake)
# # # X_tsne_turn = tsne.fit_transform(data_turn)
# # #plot_embedding_2d(X_tsne[:,0:2],"t-SNE 2D")

# 3d图
#plot_embedding_3d(X_tsne,"t-SNE 3D (time %.2fs)" %(time() - t0))

# #plot_embedding_3d(X_tsne_accel,data_merge_accel,"t-SNE 3D (time %.2fs)" %(time() - t0))
# #plot_embedding_3d(X_tsne_brake,data_merge_brake,"t-SNE 3D (time %.2fs)" %(time() - t0))
# plot_embedding_3d(X_tsne_turn,data_merge_turn,"t-SNE 3D (time %.2fs)" %(time() - t0))


def outlier_process(data_merge):
    data_merge.drop('Unnamed: 0', axis=1, inplace=True)
# data_merge_accel.drop('Unnamed: 0', axis=1, inplace=True)
# data_merge_brake.drop('Unnamed: 0', axis=1, inplace=True)
# data_merge_turn.drop('Unnamed: 0', axis=1, inplace=True)
    data_merge.to_csv(root + 'allevents_washed_outlier2.csv', encoding='gbk')
# data_merge_accel.to_csv(root + 'allevents_washed_outlier_accel.csv',
#                    encoding='gbk')
# data_merge_brake.to_csv(root + 'allevents_washed_outlier_brake.csv',
#                    encoding='gbk')
# data_merge_turn.to_csv(root + 'allevents_washed_outlier_turn.csv',
#                    encoding='gbk')
    print("complete")

outlier_process(data_merge)


