import numpy as np
import pandas as pd


class DecisionTree:

    def __init__(self):
        self.X_train = None
        self.y_train = None


    def calEnt(self):
        n = self.X_train.shape[0]  # 数据集总行数
        iset = self.y_train.value_counts()  # 标签的所有类别
        p = iset / n  # 每一类标签所占比
        ent = (-p * np.log2(p)).sum()  # 计算信息熵
        return ent

    def fit(self,X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train