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

class Cluster:

    def __init__(self):
        self.X = pd.read_csv("E:/wakeup/abandoned/data_factors.csv", encoding='gbk')
        self.X = self.X.values
        # 定义数据集

    def minmax(self):
        #归一化
        min_max_scaler = MinMaxScaler()
        self.X = min_max_scaler.fit_transform(self.X)
    def standard(self):
        #标准化
        std = StandardScaler()
        self.X = std.fit_transform(self.X)

    def Elbow_Method(self):
        """利用SSE选择k"""
        SSE = []  # 存放每次结果的误差平方和
        for k in range(1,9):
            estimator = KMeans(n_clusters=k)  # 构造聚类器
            estimator.fit(self.X)
            SSE.append(estimator.inertia_) # estimator.inertia_获取聚类准则的总和
        x = range(1,9)
        plt.xlabel('k')
        plt.ylabel('SSE')
        plt.plot(x,SSE,'o-')
        plt.show()

    def Silhouette_Coefficient(self):
        Scores = []  # 存放轮廓系数
        for k in range(2, 9):
            estimator = KMeans(n_clusters=k)  # 构造聚类器
            estimator.fit(self.X)
            Scores.append(silhouette_score(self.X, estimator.labels_, metric='euclidean'))
        x = range(2, 9)
        plt.xlabel('k')
        plt.ylabel('轮廓系数')
        plt.plot(x, Scores, 'o-')
        plt.show()

    def fit(self):
        # 定义模型
        #常见聚类模型 以下10个聚类方法需要用谁把谁注释掉即可
        # model = sc.AffinityPropagation(damping=0.9)#亲和力传播（运行太慢）
        # model = sc.AgglomerativeClustering(n_clusters=10)# 聚合聚类
        # model = sc.Birch(threshold=0.01, n_clusters=10)# birch聚类
        # model = sc.DBSCAN(eps=0.30, min_samples=10)# dbscan 聚类
        model = sc.KMeans(n_clusters=3)# k-means 聚类
        # model = sc.MiniBatchKMeans(n_clusters=3)# mini-batch k均值聚类
        # model = sc.MeanShift()# 均值漂移聚类（运行较慢）
        # model = sc.OPTICS(eps=0.8, min_samples=10)#optics聚类
        # model = sc.SpectralClustering(n_clusters=10)# spectral clustering（速度较慢，结果较散）
        #model = GaussianMixture(n_components=3)#高斯混合模型

        # 模型拟合与聚类预测
        # 模型拟合
        # 为每个示例分配一个集群
        yhat = model.fit_predict(self.X)
        #new1查看各个类数量
        print(np.unique(yhat,return_counts=True))
        # 检索唯一群集
        clusters = np.unique(yhat)
        # 为每个群集的样本创建散点图

        for cluster in clusters:
            # 获取此群集的示例的行索引
            row_ix = where(yhat == cluster)

            # 创建这些样本的散布
            plt.scatter(self.X[row_ix, 0], self.X[row_ix, 1])
        # 绘制散点图
        plt.show()
        np.set_printoptions(suppress=True)
        print(model.cluster_centers_)

if __name__ == '__main__':
    cluster = Cluster()
    #cluster.minmax()
    cluster.standard()
    cluster.Elbow_Method()
    cluster.Silhouette_Coefficient()
    cluster.fit()
