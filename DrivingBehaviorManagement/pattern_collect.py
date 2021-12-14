# -*- coding : utf-8 -*-
# coding: utf-8
import os
import pandas as pd
import numpy as np
import datetime
import matplotlib
from math import sqrt,pow,acos
from pandas.core.frame import DataFrame

class PATTERNCOLLECT:

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

        self.timegap = 40

    def label_transform(self):
        # self.accel_data.loc[self.accel_data['风险等级'] == '低', '风险等级'] = 'a'
        # self.accel_data.loc[self.accel_data['风险等级'] == '中', '风险等级'] = 'b'
        # self.accel_data.loc[self.accel_data['风险等级'] == '高', '风险等级'] = 'c'
        # self.brake_data.loc[self.brake_data['风险等级'] == '低', '风险等级'] = 'h'
        # self.brake_data.loc[self.brake_data['风险等级'] == '中', '风险等级'] = 'i'
        # self.brake_data.loc[self.brake_data['风险等级'] == '高', '风险等级'] = 'j'
        # self.turn_data.loc[self.turn_data['风险等级'] == '低', '风险等级'] = 'o'
        # self.turn_data.loc[self.turn_data['风险等级'] == '中', '风险等级'] = 'p'
        # self.turn_datas.loc[self.turn_data['风险等级'] == '高', '风险等级'] = 'q'

        pattern_data = self.data
        pattern_data.loc[(pattern_data['事件类型'] == 'accel') & (pattern_data['风险等级'] == '低'), 'pattern_label'] = 'a'
        pattern_data.loc[(pattern_data['事件类型'] == 'accel') & (pattern_data['风险等级'] == '中'), 'pattern_label'] = 'b'
        pattern_data.loc[(pattern_data['事件类型'] == 'accel') & (pattern_data['风险等级'] == '高'), 'pattern_label'] = 'c'
        pattern_data.loc[(pattern_data['事件类型'] == 'brake') & (pattern_data['风险等级'] == '低'), 'pattern_label'] = 'h'
        pattern_data.loc[(pattern_data['事件类型'] == 'brake') & (pattern_data['风险等级'] == '中'), 'pattern_label'] = 'i'
        pattern_data.loc[(pattern_data['事件类型'] == 'brake') & (pattern_data['风险等级'] == '高'), 'pattern_label'] = 'j'
        pattern_data.loc[(pattern_data['事件类型'] == 'turn') & (pattern_data['风险等级'] == '低'), 'pattern_label'] = 'o'
        pattern_data.loc[(pattern_data['事件类型'] == 'turn') & (pattern_data['风险等级'] == '中'), 'pattern_label'] = 'p'
        pattern_data.loc[(pattern_data['事件类型'] == 'turn') & (pattern_data['风险等级'] == '高'), 'pattern_label'] = 'q'



    def pattern_generate(self):


