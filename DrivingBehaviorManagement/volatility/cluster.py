# -*- coding : utf-8 -*-
# coding: utf-8
import os
import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
import matplotlib
from math import sqrt,pow,acos
from pandas.core.frame import DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import sklearn.cluster as sklearn
from numpy import where
from matplotlib import pyplot as plt


class CLUSTER:

    def __init__(self):
        self.cars_num = ["031267", "077102", "078351", "078837", "080913", "082529",
                         "090798", "098840", "108140", "112839"]
        self.filename_extenstion = '.csv'
        self.root = "E:/wakeup/"
        self.datasetpath = "E:/wakeup/dataset/"
        self.datapath = 'E:/wakeup/data/'
        self.eventpath = "E:/wakeup/event/"

        self.features_data = pd.read_csv(self.root + 'all_features.csv', encoding='gbk')

    def fit(self):
        dataset = self.features_data
        X = dataset[['安全事件比例', '中风险事件', '高风险事件',
                     '高分Pattern', '低分Pattern',
                     '高风险时长Pattern', '低风险时长Pattern', '无风险时长Pattern']]

        std = StandardScaler()
        X = std.fit_transform(X)

        model = sklearn.KMeans(n_clusters=3)

        yhat = model.fit_predict(X)
        # new1查看各个类数量
        print(np.unique(yhat, return_counts=True))
        # 检索唯一群集
        clusters = np.unique(yhat)
        # 为每个群集的样本创建散点图

        for cluster in clusters:
            # 获取此群集的示例的行索引
            row_ix = where(yhat == cluster)

            # 创建这些样本的散布
            plt.scatter(X[row_ix, 0], X[row_ix, 1])
        # 绘制散点图
        plt.show()
        np.set_printoptions(suppress=True)
        print(model.cluster_centers_)

if __name__ == '__main__':
    clst = CLUSTER()
    clst.fit()


