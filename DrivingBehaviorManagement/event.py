#-*- coding : utf-8 -*-
# coding: utf-8
import os
import pandas as pd
import numpy as np
import datetime
from pandas.core.frame import DataFrame

#TODO 特征向量，
# 驾驶行为数据里面，有急加速 急减速 转弯 疲劳驾驶
# 猛踩油门 空档滑行等等应该是油耗相关

'''
最大加速度
最小加速度
最大陀螺仪值？
速度差 
持续时间

range of accel 加速度差
加速度差 y轴
加速度标准差
加速度标准差y轴
加速度均值
加速度均值 y轴
gyroscope均值
速度均值

Axis direction
加速度最大 Y轴
the sum of the start and end values of the accel(首尾加速度和)

gyro标准差
gyro标准差y轴
gyto均值y轴
加速度最小y轴
速度标准差


把事件单独提出来做一个各自的表
序号 事件类型 采集时间 存储时间 持续时间 结束时间 风险等级 然后各特征向量

pattern的做法是 根据事件间隔 遍历所有事件  间隔内的就进入一个列表 这样就是按顺序的了

先提取出事件
再给标签100个训练

按间隔生成pattern
做lda

生成新得特征 包括scoreengine
做kmeans
'''


class EVENT:

    def __init__(self):
        self.folder_name = ["031267", "077102", "078351", "078837", "080913", "082529",
                            "090798", "098840", "108140", "112839"]
        self.filename_extenstion = '.csv'
        self.datasetpath = "D:/wakeup/dataset/"
        self.datapath = 'D:/wakeup/data/'
        self.eventpath = "D:/wakeup/event/"
        self.day = 20200901

        self.start = []
        self.endt = []
        self.durat = []
        self.spddif = []
        self.spdstd = []
        self.spdmea = []
        self.amax = []
        self.amin = []
        self.adif = []
        self.astd = []
        self.amea = []
        self.ahead = []
        self.type = []

        self.accelsave = []
        self.spdsave = []


    def cal_accel(self, i, spdobj, cltobj):
        t1 = cltobj[i-1]
        t2 = cltobj[i]
        duraa = self.cal_timeduration(t1,t2)
        v1 = spdobj[i-1]
        v2 = spdobj[i]
        return (v2-v1) / duraa


    def cal_timeduration(self, start, end):
        starttime = datetime.datetime.strptime(str(start),"%Y%m%d%H%M%S")
        endtime = datetime.datetime.strptime(str(end),"%Y%m%d%H%M%S")
        return (endtime -  starttime).seconds


    def checkemptyaccel(self, accelsave):
        if not accelsave:
            return True
        else:
            return False


    def clearevent(self):
        self.start.clear()
        self.endt.clear()
        self.durat.clear()
        self.spddif.clear()
        self.spdstd.clear()
        self.spdmea.clear()
        self.amax.clear()
        self.amin.clear()
        self.adif.clear()
        self.astd.clear()
        self.amea.clear()
        self.ahead.clear()
        self.type.clear()

    def eventresult(self, start, end, a):
        # 刹车结束 结算各项指标
        self.start.append(start)
        self.endt.append(end)
        # 持续时间
        self.durat.append(self.cal_timeduration(start, end))
        # 速度差
        self.spddif.append(max(self.spdsave) - min(self.spdsave))
        # 速度标准差
        self.spdstd.append(np.std(self.spdsave, ddof=1))
        # 速度均值
        self.spdmea.append(np.mean(self.spdsave))
        # 最大加速度
        self.amax.append(max(self.accelsave))
        self.amin.append(min(self.accelsave))
        self.adif.append(max(self.accelsave) - min(self.accelsave))
        # 加速度标准差
        self.astd.append(np.std(self.accelsave, ddof=1))
        # 加速度均值
        self.amea.append(np.mean(self.accelsave))
        # 首尾加速度和
        self.ahead.append(self.accelsave[0] + self.accelsave[-1])

        self.spdsave.clear()
        self.accelsave.clear()


    def accel_event(self, spdobj, cltobj):
        #事件：采集时间，结束时间，持续时间，速度差，速度标准差，速度均值
        # 最大加速度，最小加速度，加速度差，加速度标准差，加速度均值，首尾加速度和
        for i in range(len(spdobj)):
            # init
            if i == 0:
                isacceling = False
                continue

            # accel confirm
            if spdobj[i] > spdobj[i-1]:
                if not isacceling:
                    isacceling = True
                    #记录加速开始时间
                    start = cltobj[i-1]
                    self.spdsave.append(spdobj[i-1])

                a = self.cal_accel(i, spdobj, cltobj)
                self.accelsave.append(a)
                self.spdsave.append(spdobj[i])

            if spdobj[i] < spdobj[i-1]:
                if isacceling:
                    end = cltobj[i-1]
                    isacceling = False
                    self.eventresult(start, end, a)
                    #事件类型
                    self.type.append('a')
                else:
                    continue


    def brake_event(self, spdobj, cltobj, brkobj):
        for i in range(len(brkobj)):
            # init
            if i == 0:
                isbraking = False
                continue

            if not isbraking:
                if brkobj[i] == 1:
                    start = cltobj[i]
                    self.spdsave.append(spdobj[i])
                    isbraking = True
                    continue
                else:
                    continue
            #刹车中
            else:
                a = self.cal_accel(i, spdobj, cltobj)
                self.accelsave.append(a)
                self.spdsave.append(spdobj[i])
                if brkobj[i] == 0:
                    end = cltobj[i]
                    isbraking = False
                    self.eventresult(start, end, a)
                    # 事件类型
                    self.type.append('b')
                    self.spdsave.clear()
                    self.accelsave.clear()


    def turn_event(self, spdobj, cltobj, lefobj, rgtobj):
        for i in range(len(lefobj)):
            if i == 0:
                isturning = False
                continue

            if not isturning:
                if lefobj[i] == 1 and rgtobj[i] != lefobj[i]:
                    start = cltobj[i]
                    self.spdsave.append(spdobj[i])
                    isturning = True
                    continue
                else:
                    continue
            #转弯亮灯中
            else:
                a = self.cal_accel(i, spdobj, cltobj)
                self.accelsave.append(a)
                self.spdsave.append(spdobj[i])
                if lefobj[i] == 0:
                    end = cltobj[i]
                    isturning = False
                    self.eventresult(start, end, a)
                    # 事件类型
                    self.type.append('lt')
                    self.spdsave.clear()
                    self.accelsave.clear()

        for i in range(len(rgtobj)):
            if i == 0:
                isturning = False
                continue

            if not isturning:
                if rgtobj[i] == 1 and rgtobj[i] != lefobj[i]:
                    start = cltobj[i]
                    self.spdsave.append(spdobj[i])
                    isturning = True
                    continue
                else:
                    continue
            #转弯亮灯中
            else:
                a = self.cal_accel(i, spdobj, cltobj)
                self.accelsave.append(a)
                self.spdsave.append(spdobj[i])
                if rgtobj[i] == 0:
                    end = cltobj[i]
                    isturning = False
                    self.eventresult(start, end, a)
                    # 事件类型
                    self.type.append('rt')
                    self.spdsave.clear()
                    self.accelsave.clear()


    def eventprocess(self):
        # 创建目录
        for folder in self.folder_name:
            isExists = os.path.exists(self.eventpath + folder)
            if not isExists:
                os.makedirs(self.eventpath + folder)
            else:
                continue

        for folder in self.folder_name:
            while self.day < 20200931:
                dataisExists = os.path.exists(self.datasetpath + folder + '/' + str(self.day) + self.filename_extenstion)
                if dataisExists:
                    df = pd.read_csv(
                        self.datasetpath + folder + '/' + str(self.day) + self.filename_extenstion,
                        encoding='gbk')
                    df.rename(columns={u'脉冲车速(km/h)': 'spd', u'刹车': 'brk', u'采集时间': 'clt', u'存储时间': 'svt',
                                       u'左转向灯': 'lef', u'右转向灯': 'rgt'},
                              inplace=True)
                    spdobj = df['spd']
                    cltobj = df['clt']
                    brkobj = df['brk']
                    lefobj = df['lef']
                    rgtobj = df['rgt']
                    # # 列名占了一行 数据列从2开始
                    # # df['spd'][1] = spdobj[0]
                    # print(spdobj[0])
                    self.accel_event(spdobj, cltobj)
                    self.brake_event(spdobj, cltobj, brkobj)
                    self.turn_event(spdobj, cltobj, lefobj, rgtobj)

                    dic = {
                        'start' : self.start,
                        'end' : self.endt,
                        'durat' : self.durat,
                        'spddif' : self.spddif,
                        'spdstd' : self.spdstd,
                        'spdmea' : self.spdmea,
                        'amax' : self.amax,
                        'amin' : self.amin,
                        'adif' : self.adif,
                        'astd' : self.astd,
                        'amea' : self.amea,
                        'ahead' :self.ahead,
                        'type' : self.type
                    }
                    data = DataFrame(dic)
                    data.sort_values(by='start', inplace=True)
                    data_new = data.reset_index(drop=True)
                    data_new.to_csv(self.eventpath + folder + '/' + str(self.day) + 'event' + self.filename_extenstion,
                                encoding='gbk')
                    print("complete")
                    self.clearevent()
                    self.day += 1

                else:
                    self.day += 1
            self.day = 20200901

if __name__ == '__main__':
    eventdetection = EVENT()
    eventdetection.eventprocess()
    print('提取完毕')
    print('提取完毕!!!')






