# -*- coding : utf-8 -*-
# coding: utf-8
import os
import pandas as pd
import numpy as np
import datetime
import matplotlib
from math import sqrt,pow,acos
from pandas.core.frame import DataFrame
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from KFOLD.KFOLD import KFOLD
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone, BaseEstimator, TransformerMixin



class Eventfit:

    def __init__(self):
        self.folder_name = ["031267", "077102", "078351", "078837", "080913", "082529",
                            "090798", "098840", "108140", "112839"]
        self.filename_extenstion = '.csv'
        self.root = "E:/wakeup/"
        self.datasetpath = "E:/wakeup/dataset/"
        self.datapath = 'E:/wakeup/data/'
        self.eventpath = "E:/wakeup/event/"

        self.dataset = pd.read_csv(self.root + 'allevents.csv', encoding='gbk')
        self.data = pd.read_csv(self.root + 'allevents_label.csv', encoding='gbk')
        self.std = StandardScaler()

        self.f1_score = []
        self.acc = []


    def fit(self):
        data_train_set = self.data
        data_train_set.loc[data_train_set['风险等级'] == '低', '风险等级'] = 1
        data_train_set.loc[data_train_set['风险等级'] == '中', '风险等级'] = 2
        data_train_set.loc[data_train_set['风险等级'] == '高', '风险等级'] = 3

        y = data_train_set['风险等级']
        y = y.astype('int')
        X = data_train_set[['最大速度',
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
                              '首尾加速度和']]

        self.std = StandardScaler().fit(X)
        X = self.std.transform(X)
        clf = svm.SVC(decision_function_shape='ovo', probability=True, class_weight = 'balanced')

        skf = StratifiedKFold(n_splits=5, shuffle=True)
        for train_index, test_index in skf.split(X, y):
            clone_model = clone(clf)

            X_train_folds = X[train_index]
            y_train_folds = y[train_index]
            X_test_fold = X[test_index]
            y_test_fold = y[test_index]

            clone_model.fit(X_train_folds, y_train_folds)
            test_labels_pred = clone_model.predict(X_test_fold)
            f1_score_fold = f1_score(y_test_fold, test_labels_pred, average="weighted")
            acc_fold = accuracy_score(y_test_fold, test_labels_pred, normalize=True)
            self.f1_score.append(f1_score_fold)
            self.acc.append(acc_fold)
            print(classification_report(y_test_fold, test_labels_pred))

        print('平均准确率为：')
        print(np.array(self.acc).mean())
        print('平均f1 score为：')
        print(np.array(self.f1_score).mean())


if __name__ == '__main__':
    event_fitter = Eventfit()
    event_fitter.fit()




