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

class Eventfit:

    def __init__(self):
        self.folder_name = ["031267", "077102", "078351", "078837", "080913", "082529",
                            "090798", "098840", "108140", "112839"]
        self.filename_extenstion = '.csv'
        self.root = "E:/wakeup/"
        self.datasetpath = "E:/wakeup/dataset/"
        self.datapath = 'E:/wakeup/data/'
        self.eventpath = "E:/wakeup/event/"

        self.data = pd.read_csv(self.root + 'allevents_label.csv', encoding='gbk')
        self.accel_data = self.data[(self.data['事件类型'] == 'accel')]
        self.brake_data = self.data[(self.data['事件类型'] == 'brake')]
        self.turn_data = self.data[(self.data['事件类型'] == 'turn')]

        self.y_test_all = pd.DataFrame([])
        self.y_pred_all = pd.DataFrame([])
        self.y_pred_all_proba = pd.DataFrame([])

        self.acc = 0

    def sampler(self):
        accel_normal = self.accel_data[(self.accel_data['风险等级'] == '低')]
        accel_medium = self.accel_data[(self.accel_data['风险等级'] == '中')]
        accel_high = self.accel_data[(self.accel_data['风险等级'] == '高')]
        brake_normal = self.brake_data[(self.brake_data['风险等级'] == '低')]
        brake_medium = self.brake_data[(self.brake_data['风险等级'] == '中')]
        brake_high = self.brake_data[(self.brake_data['风险等级'] == '高')]
        turn_normal = self.turn_data[(self.turn_data['风险等级'] == '低')]
        turn_medium = self.turn_data[(self.turn_data['风险等级'] == '中')]
        turn_high = self.turn_data[(self.turn_data['风险等级'] == '高')]

        accel_high_sample = accel_high.sample(n=500,axis=0,random_state=None,replace=False)
        accel_medium_sample = accel_medium.sample(n=500, axis=0, random_state=None, replace=False)
        accel_normal_sample = accel_normal.sample(n=500, axis=0, random_state=None, replace=False)
        brake_high_sample = brake_high.sample(n=500, axis=0, random_state=None, replace=False)
        brake_medium_sample = brake_medium.sample(n=500, axis=0, random_state=None, replace=False)
        brake_normal_sample = brake_normal.sample(n=500, axis=0, random_state=None, replace=False)
        turn_high_sample = turn_high.sample(n=500, axis=0, random_state=None, replace=False)
        turn_medium_sample = turn_medium.sample(n=500, axis=0, random_state=None, replace=False)
        turn_normal_sample = turn_normal.sample(n=500, axis=0, random_state=None, replace=False)

        accel_train_set = pd.concat([accel_high_sample, accel_medium_sample, accel_normal_sample])
        accel_train_set.sort_values(by='Unnamed: 0', inplace=True)
        accel_train_set.drop(labels=['Unnamed: 0'], axis=1, inplace=True)
        accel_train_set = accel_train_set.reset_index(drop=True)

        brake_train_set = pd.concat([brake_high_sample, brake_medium_sample, brake_normal_sample])
        brake_train_set.sort_values(by='Unnamed: 0', inplace=True)
        brake_train_set.drop(labels=['Unnamed: 0'], axis=1, inplace=True)
        brake_train_set = brake_train_set.reset_index(drop=True)

        turn_train_set = pd.concat([turn_high_sample, turn_medium_sample, turn_normal_sample])
        turn_train_set.sort_values(by='Unnamed: 0', inplace=True)
        turn_train_set.drop(labels=['Unnamed: 0'], axis=1, inplace=True)
        turn_train_set = turn_train_set.reset_index(drop=True)

        accel_train_set.loc[accel_train_set['风险等级'] == '低', '风险等级'] = 1
        accel_train_set.loc[accel_train_set['风险等级'] == '中', '风险等级'] = 2
        accel_train_set.loc[accel_train_set['风险等级'] == '高', '风险等级'] = 3
        brake_train_set.loc[brake_train_set['风险等级'] == '低', '风险等级'] = 1
        brake_train_set.loc[brake_train_set['风险等级'] == '中', '风险等级'] = 2
        brake_train_set.loc[brake_train_set['风险等级'] == '高', '风险等级'] = 3
        turn_train_set.loc[turn_train_set['风险等级'] == '低', '风险等级'] = 1
        turn_train_set.loc[turn_train_set['风险等级'] == '中', '风险等级'] = 2
        turn_train_set.loc[turn_train_set['风险等级'] == '高', '风险等级'] = 3

        return accel_train_set,brake_train_set,turn_train_set

    def fit(self):
        accel_train_set, brake_train_set, turn_train_set = self.sampler()
        for item in [accel_train_set, brake_train_set, turn_train_set]:

            y = item['风险等级']
            y = y.astype('int')
            X = item[['最大速度',
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
            std = StandardScaler()
            cache = std.fit_transform(X)
            X = pd.DataFrame(cache)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=1)
            if item['事件类型'][0] == 'accel':
                self.y_test_all = pd.concat([self.y_test_all, y_test])
                clf_accel = svm.SVC()
                clf_accel.fit(X_train, y_train)
                # 交叉验证
                kf_accel = KFOLD(X, y, 10)
                accel_scores = []
                clf_accel = svm.SVC(decision_function_shape='ovr', probability=True)
                clf_accel.fit(X_train, y_train)
                accel_scores.append(kf_accel.cross_validation(clf_accel))
                accel_score = np.mean(accel_scores)
                # pred
                y_pred_accel = pd.DataFrame(clf_accel.predict(X_test))
                y_pred_accel_proba = pd.DataFrame(clf_accel.predict_proba(X_test))
                self.y_pred_all_proba = pd.concat([self.y_pred_all_proba, y_pred_accel_proba])
                self.y_pred_all = pd.concat([self.y_pred_all, y_pred_accel])



            if item['事件类型'][0] == 'brake':
                self.y_test_all = pd.concat([self.y_test_all, y_test])
                clf_brake = svm.SVC()
                clf_brake.fit(X_train, y_train)
                # 交叉验证
                kf_brake = KFOLD(X, y, 10)
                brake_scores = []
                clf_brake = svm.SVC(decision_function_shape='ovr', probability=True)
                clf_brake.fit(X_train, y_train)
                brake_scores.append(kf_brake.cross_validation(clf_brake))
                brake_score = np.mean(brake_scores)
                # pred
                y_pred_brake = pd.DataFrame(clf_brake.predict(X_test))
                y_pred_brake_proba = pd.DataFrame(clf_brake.predict_proba(X_test))
                self.y_pred_all_proba = pd.concat([self.y_pred_all_proba, y_pred_brake_proba])
                self.y_pred_all = pd.concat([self.y_pred_all, y_pred_brake])

            if item['事件类型'][0] == 'turn':
                self.y_test_all = pd.concat([self.y_test_all, y_test])
                clf_turn = svm.SVC()
                clf_turn.fit(X_train, y_train)
                # 交叉验证
                kf_turn = KFOLD(X, y, 10)
                turn_scores = []
                clf_turn = svm.SVC(decision_function_shape='ovr', probability=True)
                clf_turn.fit(X_train, y_train)
                turn_scores.append(kf_turn.cross_validation(clf_turn))
                turn_score = np.mean(turn_scores)
                # pred
                y_pred_turn = pd.DataFrame(clf_turn.predict(X_test))
                y_pred_turn_proba = pd.DataFrame(clf_turn.predict_proba(X_test))
                self.y_pred_all_proba = pd.concat([self.y_pred_all_proba, y_pred_turn_proba])
                self.y_pred_all = pd.concat([self.y_pred_all, y_pred_turn])

        self.acc = (accel_score + brake_score + turn_score) / 3
        # print(self.acc)

    def get_acc(self):
        print(accuracy_score(self.y_test_all, self.y_pred_all, normalize=True))

    def get_f1_score(self):
        print(classification_report(self.y_test_all, self.y_pred_all))

    def get_auc(self):
        print(roc_auc_score(self.y_test_all, self.y_pred_all_proba, multi_class='ovr'))

if __name__ == '__main__':
    event_fitter = Eventfit()
    # for i in range(10):
    #     event_fitter.fit()
    #     acc = event_fitter.get_acc()
    #     accs.append(acc)
    event_fitter.fit()
    event_fitter.get_acc()
    event_fitter.get_f1_score()
    event_fitter.get_auc()