# -*- coding : utf-8 -*-
# coding: utf-8
import os
import pandas as pd
import numpy as np
import datetime
from math import sqrt,pow,acos
from pandas.core.frame import DataFrame
#这个代码是用来将所有的event合并为1个allevent文件的
class EventArrange:

    def __init__(self):
        # self.folder_name = ["031267"]
        self.folder_name = ["077102", "078351", "078837", "080913", "082529",
                            "090798", "098840", "108140"]
        self.filename_extenstion = '.csv'
        self.root = "D:/RX-105/wakeup/MyJuneAndEmbedding/8car/"
        self.datasetpath = "D:/RX-105/wakeup/dataset/"
        self.datapath = 'D:/RX-105/wakeup/data/'
        self.eventpath = "D:/RX-105/wakeup/MyJuneAndEmbedding/event/"
        self.day = 20200901
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
                        self.eventpath + folder + '/' + str(self.day) + 'event' + self.filename_extenstion,
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
