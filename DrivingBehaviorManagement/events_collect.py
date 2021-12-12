# -*- coding : utf-8 -*-
# coding: utf-8
import os
import pandas as pd
import numpy as np
import datetime
from math import sqrt,pow,acos
from pandas.core.frame import DataFrame

class EventArrange:

    def __init__(self):
        self.folder_name = ["031267", "077102", "078351", "078837", "080913", "082529",
                            "090798", "098840", "108140", "112839"]
        self.filename_extenstion = '.csv'
        self.root = "E:/wakeup/"
        self.datasetpath = "E:/wakeup/dataset/"
        self.datapath = 'E:/wakeup/data/'
        self.eventpath = "E:/wakeup/event/"
        self.day = 20200902
        self.first_csv_switch = False

    def event_collection(self):
        # 创建目录
        for folder in self.folder_name:
            isExists = os.path.exists(self.eventpath + folder)
            if not isExists:
                os.makedirs(self.eventpath + folder)
            else:
                continue

        for folder in self.folder_name:
            while self.day < 20200931:
                if not self.first_csv_switch:
                    df1 = pd.read_csv(
                        self.eventpath + '031267' + '/' + '20200901' + 'event' + self.filename_extenstion,
                        encoding='gbk')
                    self.first_csv_switch = True
                else:
                    eventisExists = os.path.exists(
                        self.eventpath + folder + '/' + str(self.day) + 'event' +self.filename_extenstion)
                    if eventisExists:
                        df2 = pd.read_csv(
                            self.eventpath + folder + '/' + str(self.day) + 'event' + self.filename_extenstion,
                            encoding='gbk')
                        df1 = pd.concat([df1, df2], ignore_index=True)
                        self.day += 1
                    else:
                        self.day += 1
                print("1 day collected")
            self.day = 20200901
            print("1 car collected")
        df1.drop('Unnamed: 0', axis=1, inplace=True)
        df1.to_csv(self.root + 'allevents' + self.filename_extenstion,
                   encoding='gbk')


if __name__ == '__main__':
    event_collector = EventArrange()
    event_collector.event_collection()
    print('整合完毕')
