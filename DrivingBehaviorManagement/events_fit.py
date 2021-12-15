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

        self.dataset = pd.read_csv(self.root + 'allevents.csv', encoding='gbk')
        self.accel_dataset = self.dataset[(self.dataset['事件类型'] == 'accel')]
        self.brake_dataset = self.dataset[(self.dataset['事件类型'] == 'brake')]
        self.turn_dataset = self.dataset[(self.dataset['事件类型'] == 'turn')]

        self.data = pd.read_csv(self.root + 'allevents_label.csv', encoding='gbk')
        self.accel_data = self.data[(self.data['事件类型'] == 'accel')]
        self.brake_data = self.data[(self.data['事件类型'] == 'brake')]
        self.turn_data = self.data[(self.data['事件类型'] == 'turn')]

        self.std = StandardScaler()

        self.y_test_all = pd.DataFrame([])
        self.y_pred_all = pd.DataFrame([])
        self.y_pred_all_proba = pd.DataFrame([])

        self.acc = 0

    def sampler_om(self):
        data_normal = self.data[(self.data['风险等级'] == '低')]
        data_medium = self.data[(self.data['风险等级'] == '中')]
        data_high = self.data[(self.data['风险等级'] == '高')]

        data_high_sample = data_high.sample(n=2500, axis=0, random_state=None, replace=False)
        data_medium_sample = data_medium.sample(n=7500, axis=0, random_state=None, replace=False)
        data_normal_sample = data_normal.sample(n=20000, axis=0, random_state=None, replace=False)

        data_train_set = pd.concat([data_high_sample, data_medium_sample, data_normal_sample])
        data_train_set.sort_values(by='Unnamed: 0', inplace=True)
        data_train_set.drop(labels=['Unnamed: 0'], axis=1, inplace=True)
        data_train_set = data_train_set.reset_index(drop=True)

        data_train_set.loc[data_train_set['风险等级'] == '低', '风险等级'] = 1
        data_train_set.loc[data_train_set['风险等级'] == '中', '风险等级'] = 2
        data_train_set.loc[data_train_set['风险等级'] == '高', '风险等级'] = 3

        return data_train_set

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

    def fit_om(self):
        data_train_set = self.sampler_om()

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

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=1)
        self.std = StandardScaler().fit(X_train)
        X_train = self.std.transform(X_train)  # 讲规则应用到训练集
        X_test = self.std.transform(X_test)

        self.y_test_all = pd.concat([self.y_test_all, y_test])
        clf = svm.SVC(decision_function_shape='ovo', probability=True, class_weight = 'balanced')
        clf.fit(X_train, y_train)

        # # 交叉验证
        # kf_accel = KFOLD(X, y, 10)
        # accel_scores = []
        # accel_scores.append(kf_accel.cross_validation(clf_accel))
        # accel_score = np.mean(accel_scores)

        # pred
        y_pred_accel = pd.DataFrame(clf.predict(X_test))
        y_pred_accel_proba = pd.DataFrame(clf.predict_proba(X_test))
        self.y_pred_all_proba = pd.concat([self.y_pred_all_proba, y_pred_accel_proba])
        self.y_pred_all = pd.concat([self.y_pred_all, y_pred_accel])

        return clf

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


            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=1)
            self.std = StandardScaler().fit(X_train)
            X_train = self.std.transform(X_train)  # 讲规则应用到训练集
            X_test = self.std.transform(X_test)
            if item['事件类型'][0] == 'accel':
                self.y_test_all = pd.concat([self.y_test_all, y_test])
                clf_accel = svm.SVC(decision_function_shape='ovo', probability=True)
                clf_accel.fit(X_train, y_train)

                # # 交叉验证
                # kf_accel = KFOLD(X, y, 10)
                # accel_scores = []
                # accel_scores.append(kf_accel.cross_validation(clf_accel))
                # accel_score = np.mean(accel_scores)

                # pred
                y_pred_accel = pd.DataFrame(clf_accel.predict(X_test))
                y_pred_accel_proba = pd.DataFrame(clf_accel.predict_proba(X_test))
                self.y_pred_all_proba = pd.concat([self.y_pred_all_proba, y_pred_accel_proba])
                self.y_pred_all = pd.concat([self.y_pred_all, y_pred_accel])



            if item['事件类型'][0] == 'brake':
                self.y_test_all = pd.concat([self.y_test_all, y_test])
                clf_brake = svm.SVC(decision_function_shape='ovo', probability=True)
                clf_brake.fit(X_train, y_train)

                # # 交叉验证
                # kf_brake = KFOLD(X, y, 10)
                # brake_scores = []
                # brake_scores.append(kf_brake.cross_validation(clf_brake))
                # brake_score = np.mean(brake_scores)

                # pred
                y_pred_brake = pd.DataFrame(clf_brake.predict(X_test))
                y_pred_brake_proba = pd.DataFrame(clf_brake.predict_proba(X_test))
                self.y_pred_all_proba = pd.concat([self.y_pred_all_proba, y_pred_brake_proba])
                self.y_pred_all = pd.concat([self.y_pred_all, y_pred_brake])

            if item['事件类型'][0] == 'turn':
                self.y_test_all = pd.concat([self.y_test_all, y_test])
                clf_turn = svm.SVC(decision_function_shape='ovo', probability=True)
                clf_turn.fit(X_train, y_train)

                ## 交叉验证
                # kf_turn = KFOLD(X, y, 10)
                # turn_scores = []
                # turn_scores.append(kf_turn.cross_validation(clf_turn))
                # turn_score = np.mean(turn_scores)

                # pred
                y_pred_turn = pd.DataFrame(clf_turn.predict(X_test))
                y_pred_turn_proba = pd.DataFrame(clf_turn.predict_proba(X_test))
                self.y_pred_all_proba = pd.concat([self.y_pred_all_proba, y_pred_turn_proba])
                self.y_pred_all = pd.concat([self.y_pred_all, y_pred_turn])

        # self.acc = (accel_score + brake_score + turn_score) / 3
        # print(self.acc)

        return clf_accel, clf_brake, clf_turn

    def get_acc(self):
        print('Accuracy is:')
        print(accuracy_score(self.y_test_all, self.y_pred_all, normalize=True))

    def get_f1_score(self):
        print('')
        print(classification_report(self.y_test_all, self.y_pred_all))

    def get_auc(self):
        print('AUC is:')
        print(roc_auc_score(self.y_test_all, self.y_pred_all_proba, multi_class='ovr'))

    def get_std(self):
        return self.std


    # def fit_one_clf(self):
    #     accel_train_set, brake_train_set, turn_train_set = self.sampler()
    #     for item in [accel_train_set, brake_train_set, turn_train_set]:
    #         data_set
    #         y = item['风险等级']
    #         y = y.astype('int')
    #         X = item[['最大速度',
    #                   '最小速度',
    #                   '速度极差',
    #                   '速度标准差',
    #                   '速度均值',
    #                   '临界速度',
    #                   '最大加速度',
    #                   '最小加速度',
    #                   '加速度极差',
    #                   '加速度标准差',
    #                   '加速度均值',
    #                   '首尾加速度和']]

class EventpredMM:

    def __init__(self, accel_model, brake_model, turn_model):
        self.folder_name = ["031267", "077102", "078351", "078837", "080913", "082529",
                            "090798", "098840", "108140", "112839"]
        self.filename_extenstion = '.csv'
        self.root = "E:/wakeup/"
        self.datasetpath = "E:/wakeup/dataset/"
        self.datapath = 'E:/wakeup/data/'
        self.eventpath = "E:/wakeup/event/"

        #model
        self.accel_model, self.brake_model, self.turn_model = accel_model, brake_model, turn_model

        self.dataset = pd.read_csv(self.root + 'allevents_label.csv', encoding='gbk')

        self.y_all = pd.DataFrame([])
        self.y_pred_all = pd.DataFrame([])
        self.y_pred_all_proba = pd.DataFrame([])

    def label_transform(self):

        self.dataset.loc[self.dataset['风险等级'] == '低', '风险等级'] = 1
        self.dataset.loc[self.dataset['风险等级'] == '中', '风险等级'] = 2
        self.dataset.loc[self.dataset['风险等级'] == '高', '风险等级'] = 3

        # self.accel_dataset.loc[self.accel_dataset['风险等级'] == '低', '风险等级'] = 1
        # self.accel_dataset.loc[self.accel_dataset['风险等级'] == '中', '风险等级'] = 2
        # self.accel_dataset.loc[self.accel_dataset['风险等级'] == '高', '风险等级'] = 3
        # self.brake_dataset.loc[self.brake_dataset['风险等级'] == '低', '风险等级'] = 1
        # self.brake_dataset.loc[self.brake_dataset['风险等级'] == '中', '风险等级'] = 2
        # self.brake_dataset.loc[self.brake_dataset['风险等级'] == '高', '风险等级'] = 3
        # self.turn_dataset.loc[self.turn_dataset['风险等级'] == '低', '风险等级'] = 1
        # self.turn_dataset.loc[self.turn_dataset['风险等级'] == '中', '风险等级'] = 2
        # self.turn_dataset.loc[self.turn_dataset['风险等级'] == '高', '风险等级'] = 3

        accel_dataset = self.dataset[(self.dataset['事件类型'] == 'accel')]
        brake_dataset = self.dataset[(self.dataset['事件类型'] == 'brake')]
        turn_dataset = self.dataset[(self.dataset['事件类型'] == 'turn')]

        return accel_dataset, brake_dataset, turn_dataset

    def predict(self):
        accel_dataset, brake_dataset, turn_dataset = self.label_transform()
        for item in [accel_dataset, brake_dataset, turn_dataset]:
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
            print(item.iloc[0].loc['事件类型'])
            if item.iloc[0].loc['事件类型'] == 'accel':
                self.y_all = pd.concat([self.y_all, y])
                # pred
                y_pred_accel = pd.DataFrame(self.accel_model.predict(X))
                y_pred_accel_proba = pd.DataFrame(self.accel_model.predict_proba(X))
                self.y_pred_all_proba = pd.concat([self.y_pred_all_proba, y_pred_accel_proba])
                self.y_pred_all = pd.concat([self.y_pred_all, y_pred_accel])

            if item.iloc[0].loc['事件类型'] == 'brake':
                self.y_all = pd.concat([self.y_all, y])
                # pred
                y_pred_brake = pd.DataFrame(self.brake_model.predict(X))
                y_pred_brake_proba = pd.DataFrame(self.brake_model.predict_proba(X))
                self.y_pred_all_proba = pd.concat([self.y_pred_all_proba, y_pred_brake_proba])
                self.y_pred_all = pd.concat([self.y_pred_all, y_pred_brake])

            if item.iloc[0].loc['事件类型'] == 'turn':
                self.y_all = pd.concat([self.y_all, y])
                # pred
                y_pred_turn = pd.DataFrame(self.turn_model.predict(X))
                y_pred_turn_proba = pd.DataFrame(self.turn_model.predict_proba(X))
                self.y_pred_all_proba = pd.concat([self.y_pred_all_proba, y_pred_turn_proba])
                self.y_pred_all = pd.concat([self.y_pred_all, y_pred_turn])

    def get_acc(self):
        print('Accuracy is:')
        print(accuracy_score(self.y_all, self.y_pred_all, normalize=True))

    def get_f1_score(self):
        print('')
        print(classification_report(self.y_all, self.y_pred_all))

    def get_auc(self):
        print('AUC is:')
        print(roc_auc_score(self.y_all, self.y_pred_all_proba, multi_class='ovr'))


class EventpredOM:

    def __init__(self, model):
        self.folder_name = ["031267", "077102", "078351", "078837", "080913", "082529",
                            "090798", "098840", "108140", "112839"]
        self.filename_extenstion = '.csv'
        self.root = "E:/wakeup/"
        self.datasetpath = "E:/wakeup/dataset/"
        self.datapath = 'E:/wakeup/data/'
        self.eventpath = "E:/wakeup/event/"

        # model
        self.model = model

        self.dataset = pd.read_csv(self.root + 'allevents_label.csv', encoding='gbk')

        self.y_all = pd.DataFrame([])
        self.y_pred_all = pd.DataFrame([])
        self.y_pred_all_proba = pd.DataFrame([])

    def label_transform_om(self):
        self.dataset.loc[self.dataset['风险等级'] == '低', '风险等级'] = 1
        self.dataset.loc[self.dataset['风险等级'] == '中', '风险等级'] = 2
        self.dataset.loc[self.dataset['风险等级'] == '高', '风险等级'] = 3

        return self.dataset

    def predict_om(self):
        dataset = self.label_transform_om()
        y = dataset['风险等级']
        y = y.astype('int')
        X = dataset[['最大速度',
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

        self.y_all = pd.concat([self.y_all, y])
        # pred
        y_pred_accel = pd.DataFrame(self.model.predict(X))
        y_pred_accel_proba = pd.DataFrame(self.model.predict_proba(X))
        self.y_pred_all_proba = pd.concat([self.y_pred_all_proba, y_pred_accel_proba])
        self.y_pred_all = pd.concat([self.y_pred_all, y_pred_accel])

    def get_acc(self):
        print('Accuracy is:')
        print(accuracy_score(self.y_all, self.y_pred_all, normalize=True))

    def get_f1_score(self):
        print('')
        print(classification_report(self.y_all, self.y_pred_all))

    def get_auc(self):
        print('AUC is:')
        print(roc_auc_score(self.y_all, self.y_pred_all_proba, multi_class='ovr'))


if __name__ == '__main__':
    event_fitter = Eventfit()
    # for i in range(10):
    #     event_fitter.fit()
    #     acc = event_fitter.get_acc()
    #     accs.append(acc)
    # clf_accel_model, clf_brake_model, clf_turn_model = event_fitter.fit()
    clf_model = event_fitter.fit_om()
    event_fitter.get_acc()
    event_fitter.get_f1_score()
    event_fitter.get_auc()

    # event_predictor = EventpredMM(clf_accel_model, clf_brake_model, clf_turn_model)
    event_predictor = EventpredOM(clf_model)
    event_predictor.predict_om()
    event_predictor.get_acc()
    event_predictor.get_f1_score()
    event_predictor.get_auc()



