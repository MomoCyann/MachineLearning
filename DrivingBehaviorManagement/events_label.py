# -*- coding : utf-8 -*-
# coding: utf-8
import os
import pandas as pd
import numpy as np
import datetime
import matplotlib
from math import sqrt,pow,acos
from pandas.core.frame import DataFrame

class EventLabel:

    def __init__(self):
        self.folder_name = ["031267", "077102", "078351", "078837", "080913", "082529",
                            "090798", "098840", "108140", "112839"]
        self.filename_extenstion = '.csv'
        self.root = "E:/wakeup/"
        self.datasetpath = "E:/wakeup/dataset/"
        self.datapath = 'E:/wakeup/data/'
        self.eventpath = "E:/wakeup/event/"

        self.data = pd.read_csv(self.root + 'allevents.csv', encoding='gbk')
        self.accel_data = self.data[(self.data['事件类型'] == 'accel')]
        self.brake_data = self.data[(self.data['事件类型'] == 'brake')]
        self.turn_data = self.data[(self.data['事件类型'] == 'turn')]

    def label_accel(self):
        accel_bins = [-0.1,1,1.5,999]
        self.accel_data = self.accel_data.copy()
        self.accel_data['风险等级']=''
        self.accel_data['风险等级'] = pd.cut(x=self.accel_data['最大加速度'],bins=accel_bins,
                        labels=['低','中','高'])
        print('accel event label')
        print(self.accel_data['风险等级'].value_counts(normalize=True)
              .mul(100)
              .rename_axis('risk level')
              .reset_index(name='percentage'))

    def label_brake(self):
        brake_bins = [-999, -2, -1, 1, 2, 999]
        self.brake_data = self.brake_data.copy()
        self.brake_data['风险等级'] = ''
        self.brake_data['风险等级'] = pd.cut(x=self.brake_data['最大加速度'], bins=brake_bins, ordered=False,
                                      labels=['高','中','低','中','高'])
        print('brake event label')
        print(self.brake_data['风险等级'].value_counts(normalize=True)
              .mul(100)
              .rename_axis('risk level')
              .reset_index(name='percentage'))

    def label_turn(self):
        turn_bins = [-999,-1.5,-1,1,1.5,999]
        self.turn_data = self.turn_data.copy()
        self.turn_data['风险等级'] = ''
        self.turn_data['风险等级'] = pd.cut(x=self.turn_data['最大加速度'], bins=turn_bins, ordered=False,
                                      labels=['高','中','低','中','高'])
        print('turn event label')
        print(self.turn_data['风险等级'].value_counts(normalize=True)
              .mul(100)
              .rename_axis('risk level')
              .reset_index(name='percentage'))

    def data_bind(self):
        data = pd.concat([self.accel_data, self.brake_data, self.turn_data])
        data.sort_values(by='Unnamed: 0', inplace=True)
        data.drop(labels=['Unnamed: 0'], axis=1, inplace=True)
        print('data bind complete')
        print(data.groupby('事件类型')['风险等级'].value_counts())
        data.to_csv(self.root + 'allevents_label' + self.filename_extenstion,
                   encoding='gbk')



if __name__ == '__main__':
    event_labeler = EventLabel()
    event_labeler.label_accel()
    event_labeler.label_brake()
    event_labeler.label_turn()
    event_labeler.data_bind()


