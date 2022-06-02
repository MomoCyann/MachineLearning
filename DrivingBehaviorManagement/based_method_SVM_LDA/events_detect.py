# -*- coding : utf-8 -*-
# coding: utf-8
import os
import pandas as pd
import numpy as np
import datetime
from math import sqrt, pow, acos
from pandas.core.frame import DataFrame

# TODO 修改油门踏板为加速判定


'''

把事件单独提出来做一个各自的表
序号 事件类型 采集时间 存储时间 持续时间 结束时间 风险等级 然后各特征向量

pattern的做法是 根据事件间隔 遍历所有事件  间隔内的就进入一个列表 这样就是按顺序的了

按间隔生成pattern
做lda

生成新得特征 包括scoreengine
做kmeans
'''


def cal_timeduration(start, end):
    starttime = datetime.datetime.strptime(str(start), "%Y%m%d%H%M%S")
    endtime = datetime.datetime.strptime(str(end), "%Y%m%d%H%M%S")
    return (endtime - starttime).seconds


def cal_accel(i, spdobj, cltobj):
    t1 = cltobj[i - 1]
    t2 = cltobj[i]
    time_dura = cal_timeduration(t1, t2)
    v1 = spdobj[i - 1]
    v2 = spdobj[i]
    return ((v2 - v1) / 3.6) / time_dura


def get_timeduration(i, cltobj):
    t1 = cltobj[i - 1]
    t2 = cltobj[i]
    time_dura = cal_timeduration(t1, t2)
    return time_dura


def cal_angle(i, lgtobj, latobj):
    x1 = lgtobj.iloc[i - 1]
    x2 = lgtobj.iloc[i]
    x3 = lgtobj.iloc[i + 1]
    y1 = latobj.iloc[i - 1]
    y2 = latobj.iloc[i]
    y3 = latobj.iloc[i + 1]
    # 向量表示两段路程
    vx1 = (x2 - x1) * 100000  # 米
    vy1 = (y2 - y1) * 100000 * 1.1
    vx2 = (x3 - x2) * 100000
    vy2 = (y3 - y2) * 100000 * 1.1
    # 求向量余弦值再转为角度
    pi = 3.1415
    ab = vx1 * vx2 + vy1 * vy2
    mo = sqrt(pow(vx1, 2) + pow(vy1, 2)) * sqrt(pow(vx2, 2) + pow(vy2, 2))
    cos = ab * 1.0 / (mo * 1.0 + 1e-6)
    if cos == 0.0:
        return 0

    angle = (acos(cos) / pi) * 180
    return angle


def cal_jerk(i, accelsave, timeduration):
    a1 = accelsave[i]
    a2 = accelsave[i + 1]
    time = timeduration[i + 1]
    return (a2 - a1) / time


class EVENT:

    def __init__(self):
        self.folder_name = [#"031267",
                            "077102", "078351", "078837", "080913", "082529",
                            "090798", "098840", "108140",]
                            #"112839"]
        self.filename_extenstion = '.csv'
        self.datasetpath = "E:/wakeup/dataset/"
        self.datapath = 'E:/wakeup/data/'
        self.eventpath = "E:/wakeup/event/"
        self.day = 20200901

        self.start = []
        self.endt = []
        self.durat = []
        self.spdmax = []
        self.spdmin = []
        self.spddif = []
        self.spdstd = []
        self.spdmea = []
        self.amax = []
        self.spdcrt = []
        self.amin = []
        self.adif = []
        self.astd = []
        self.amea = []
        self.ahead = []
        self.type = []
        self.jerk = []
        self.jmea = []
        self.jmax = []
        self.jmin = []
        self.jstd = []
        self.jhead = []
        self.jdif = []

        self.accelsave = []
        self.timeduration = []
        self.spdsave = []

    def clearevent(self):
        self.start.clear()
        self.endt.clear()
        self.durat.clear()
        self.spdmax.clear()
        self.spdmin.clear()
        self.spddif.clear()
        self.spdstd.clear()
        self.spdmea.clear()
        self.amax.clear()
        self.spdcrt.clear()
        self.amin.clear()
        self.adif.clear()
        self.astd.clear()
        self.amea.clear()
        self.ahead.clear()
        self.type.clear()
        self.jerk.clear()
        self.jmea.clear()
        self.jmax.clear()
        self.jmin.clear()
        self.jstd.clear()
        self.jhead.clear()
        self.jdif.clear()

    def accel_eventresult(self, start, end):
        # 加速结束 结算各项指标
        self.start.append(start)
        self.endt.append(end)
        # 持续时间
        self.durat.append(cal_timeduration(start, end))
        # 速度最大最小值
        self.spdmax.append(max(self.spdsave))
        self.spdmin.append(min(self.spdsave))
        # 速度差
        self.spddif.append(max(self.spdsave) - min(self.spdsave))
        # 速度标准差
        self.spdstd.append(np.std(self.spdsave, ddof=1))
        # 速度均值
        self.spdmea.append(np.mean(self.spdsave))

        # 最大加速度
        amax = max(self.accelsave)
        # 达到最大加速度时的速度
        tar = self.accelsave.index(max(self.accelsave))
        crtspd = (self.spdsave[tar] + self.spdsave[tar + 1]) / 2
        self.spdcrt.append(crtspd)
        amin = min(self.accelsave)
        # amax = max(self.accelsave)
        # amin = min(self.accelsave)
        self.amax.append(amax)
        # 最小加速度
        self.amin.append(amin)
        # 加速度极差
        self.adif.append(amax - amin)
        # 加速度标准差
        if np.std(self.accelsave, ddof=1) != np.std(self.accelsave, ddof=1):
            self.astd.append(0)
        else:
            self.astd.append(np.std(self.accelsave, ddof=1))
        # 加速度均值
        self.amea.append(np.mean(self.accelsave))
        # 首尾加速度和
        self.ahead.append(self.accelsave[0] + self.accelsave[-1])

        if len(self.accelsave) > 1:
            for i in range(len(self.accelsave) - 1):
                j = cal_jerk(i, self.accelsave, self.timeduration)
                self.jerk.append(j)
            # jerk均值
            self.jmea.append(np.mean(self.jerk))
            # jerk最大值最小值
            jmax = max(self.jerk)
            jmin = min(self.jerk)
            self.jmax.append(jmax)
            self.jmin.append(jmin)
            self.jdif.append(jmax - jmin)
            # jerk标准差
            if np.std(self.jerk, ddof=1) != np.std(self.jerk, ddof=1):
                self.jstd.append(0)
            else:
                self.jstd.append(np.std(self.jerk, ddof=1))
            # 首尾加速度和
            self.jhead.append(self.jerk[0] + self.jerk[-1])
        else:
            self.jmea.append(0)
            self.jmax.append(0)
            self.jmin.append(0)
            self.jdif.append(0)
            self.jstd.append(0)
            self.jhead.append(0)

        self.timeduration.clear()
        self.spdsave.clear()
        self.accelsave.clear()
        self.jerk.clear()

    def brake_eventresult(self, start, end):
        # 刹车结束 结算各项指标
        self.start.append(start)
        self.endt.append(end)
        # 持续时间
        self.durat.append(cal_timeduration(start, end))
        # 速度最大最小值
        self.spdmax.append(max(self.spdsave))
        self.spdmin.append(min(self.spdsave))
        # 速度差
        self.spddif.append(max(self.spdsave) - min(self.spdsave))
        # 速度标准差
        self.spdstd.append(np.std(self.spdsave, ddof=1))
        # 速度均值
        self.spdmea.append(np.mean(self.spdsave))

        # 加速度取绝对值
        absolute_a = np.maximum(np.array(self.accelsave), -np.array(self.accelsave))
        # 最大加速度
        # amax = min(self.accelsave)
        # 达到最大加速度时的速度
        tar = self.accelsave.index(max(self.accelsave))
        crtspd = (self.spdsave[tar] + self.spdsave[tar + 1]) / 2
        self.spdcrt.append(crtspd)

        # amin = max(self.accelsave)
        amax = max(absolute_a)
        amin = min(absolute_a)
        self.amax.append(amax)
        # 最小加速度
        self.amin.append(amin)
        # 加速度极差
        self.adif.append(max(absolute_a) - min(absolute_a))
        # 加速度标准差
        if np.std(absolute_a, ddof=1) != np.std(absolute_a, ddof=1):
            self.astd.append(0)
        else:
            self.astd.append(np.std(absolute_a, ddof=1))
        # 加速度均值
        self.amea.append(np.mean(absolute_a))
        # 首尾加速度和
        # self.ahead.append(self.accelsave[0] + self.accelsave[-1])
        self.ahead.append(absolute_a[0] + absolute_a[-1])

        if len(absolute_a) > 1:
            for i in range(len(absolute_a) - 1):
                j = cal_jerk(i, absolute_a, self.timeduration)
                self.jerk.append(j)
            # jerk均值
            self.jmea.append(np.mean(self.jerk))
            # jerk最大值最小值
            jmax = max(self.jerk)
            jmin = min(self.jerk)
            self.jmax.append(jmax)
            self.jmin.append(jmin)
            self.jdif.append(jmax - jmin)
            # jerk标准差
            if np.std(self.jerk, ddof=1) != np.std(self.jerk, ddof=1):
                self.jstd.append(0)
            else:
                self.jstd.append(np.std(self.jerk, ddof=1))
            # 首尾加速度和
            self.jhead.append(self.jerk[0] + self.jerk[-1])
        else:
            self.jmea.append(0)
            self.jmax.append(0)
            self.jmin.append(0)
            self.jdif.append(0)
            self.jstd.append(0)
            self.jhead.append(0)

        self.spdsave.clear()
        self.jerk.clear()
        self.accelsave.clear()
        self.timeduration.clear()

    def turn_eventresult(self, start, end):
        # 转弯结束 结算各项指标
        self.start.append(start)
        self.endt.append(end)
        # 持续时间
        self.durat.append(cal_timeduration(start, end))
        # 速度最大最小值
        self.spdmax.append(max(self.spdsave))
        self.spdmin.append(min(self.spdsave))
        # 速度差
        self.spddif.append(max(self.spdsave) - min(self.spdsave))
        # 速度标准差
        self.spdstd.append(np.std(self.spdsave, ddof=1))
        # 速度均值
        self.spdmea.append(np.mean(self.spdsave))

        # 加速度取绝对值，我认为只需要表示速度变化趋势的大小。
        absolute_a = np.maximum(np.array(self.accelsave), -np.array(self.accelsave))
        # 最大加速度
        # amax = max(min(self.accelsave), max(self.accelsave), key=abs)
        # 达到最大加速度时的速度
        tar = self.accelsave.index(max(self.accelsave))
        crtspd = (self.spdsave[tar] + self.spdsave[tar + 1]) / 2
        self.spdcrt.append(crtspd)

        # amin = 999
        # for i in self.accelsave:
        #     if max(i, -i) < amin:
        #         amin = i
        amax = max(absolute_a)
        amin = min(absolute_a)
        self.amax.append(amax)
        # 最小加速度
        self.amin.append(amin)
        # 加速度极差
        self.adif.append(amax - amin)
        # 加速度标准差
        if np.std(absolute_a, ddof=1) != np.std(absolute_a, ddof=1):
            self.astd.append(0)
        else:
            self.astd.append(np.std(absolute_a, ddof=1))
        # 加速度均值
        self.amea.append(np.mean(absolute_a))
        # 首尾加速度和
        self.ahead.append(absolute_a[0] + absolute_a[-1])

        if len(absolute_a) > 1:
            for i in range(len(absolute_a) - 1):
                j = cal_jerk(i, absolute_a, self.timeduration)
                self.jerk.append(j)
            # jerk均值
            self.jmea.append(np.mean(self.jerk))
            # jerk最大值最小值
            jmax = max(self.jerk)
            jmin = min(self.jerk)
            self.jmax.append(jmax)
            self.jmin.append(jmin)
            self.jdif.append(jmax - jmin)
            # jerk标准差
            if np.std(self.jerk, ddof=1) != np.std(self.jerk, ddof=1):
                self.jstd.append(0)
            else:
                self.jstd.append(np.std(self.jerk, ddof=1))
            # 首尾加速度和
            self.jhead.append(self.jerk[0] + self.jerk[-1])
        else:
            self.jmea.append(0)
            self.jmax.append(0)
            self.jmin.append(0)
            self.jdif.append(0)
            self.jstd.append(0)
            self.jhead.append(0)

        self.spdsave.clear()
        self.timeduration.clear()
        self.accelsave.clear()
        self.jerk.clear()

    # def get_a(self):
    #     df = pd.read_csv(
    #         self.datasetpath + "031267" + '/' + str(self.day) + self.filename_extenstion,
    #         encoding='gbk')
    #     df.rename(columns={u'脉冲车速(km/h)': 'spd', u'刹车': 'brk', u'采集时间': 'clt', u'存储时间': 'svt',
    #                        u'左转向灯': 'lef', u'右转向灯': 'rgt'},
    #               inplace=True)
    #     spdobj = df['spd']
    #     cltobj = df['clt']
    #     time = []
    #     all_a = []
    #     for i in range(len(spdobj)):
    #         if i == 0:
    #             continue
    #         delta_v = (spdobj[i] - spdobj[i - 1]) / 3.6  # m/s
    #         delta_t = cal_timeduration(cltobj[i - 1], cltobj[i])
    #         a = delta_v / delta_t
    #         all_a.append(a)
    #         time.append(cltobj[i - 1])
    #     dic = {
    #         'start': time,
    #         'a': all_a
    #     }
    #     data = DataFrame(dic)
    #     data.sort_values(by='start', inplace=True)
    #     data_new = data.reset_index(drop=True)
    #     data_new.to_csv(self.eventpath + "031267" + '/' + str(self.day) + 'a' + self.filename_extenstion,
    #                     encoding='gbk')

    def accel_event(self, trgobj, spdobj, cltobj):
        # 事件：采集时间，结束时间，持续时间，速度差，速度标准差，速度均值
        # 最大加速度，最小加速度，加速度差，加速度标准差，加速度均值，首尾加速度和
        # 5/23 将指针指向转速。
        for i in range(len(trgobj)):
            # 初始化标记
            if i == 0:
                isacceling = False
                continue

            # 加速事件开始
            if trgobj[i] >= 20:
                if not isacceling:
                    isacceling = True
                    # 记录加速开始时间
                    start = cltobj[i-1]
                    self.spdsave.append(spdobj[i-1])

                a = cal_accel(i, spdobj, cltobj)
                t = get_timeduration(i, cltobj)

                self.timeduration.append(t)
                self.accelsave.append(a)
                self.spdsave.append(spdobj[i])

            # # 速度相同的时候 只录入速度
            # if spdobj[i] == spdobj[i-1]:
            #     self.spdsave.append(spdobj[i])

            # 加速事件停止条件的判定
            if trgobj[i] < 20 or i == len(trgobj):  # 最后一个强制结束
                if isacceling:
                    end = cltobj[i - 1]
                    isacceling = False
                    self.accel_eventresult(start, end)
                    # 事件类型
                    self.type.append('accel')
                else:
                    continue

    def brake_event(self, spdobj, cltobj, brkobj, trgobj):
        for i in range(len(brkobj)):
            # init
            if i == 0:
                isbraking = False
                continue

            # 减速事件开始
            if not isbraking:
                if brkobj[i] == 1:
                    start = cltobj[i]
                    self.spdsave.append(spdobj[i])
                    isbraking = True
                    continue
                else:
                    continue
            # 刹车中
            else:
                a = cal_accel(i, spdobj, cltobj)
                t = get_timeduration(i, cltobj)

                self.timeduration.append(t)
                self.accelsave.append(a)
                self.spdsave.append(spdobj[i])

                # 停止条件判定; 速度增加的时候也视为刹车停止
                if brkobj[i] == 0 or trgobj[i-1] < trgobj[i]:
                    end = cltobj[i]
                    isbraking = False
                    self.brake_eventresult(start, end)
                    # 事件类型
                    self.type.append('brake')

    # def turn_event(self, spdobj, cltobj, lefobj, rgtobj):
    #     for i in range(len(lefobj)):
    #         if i == 0:
    #             isturning = False
    #             continue
    #
    #         if not isturning:
    #             if lefobj[i] == 1 and rgtobj[i] != lefobj[i]:
    #                 start = cltobj[i]
    #                 self.spdsave.append(spdobj[i])
    #                 isturning = True
    #                 continue
    #             else:
    #                 continue
    #         # 转弯亮灯中
    #         else:
    #             a = cal_accel(i, spdobj, cltobj)
    #             self.accelsave.append(a)
    #             self.spdsave.append(spdobj[i])
    #             if lefobj[i] == 0:
    #                 end = cltobj[i]
    #                 isturning = False
    #                 self.eventresult(start, end)
    #                 # 事件类型
    #                 self.type.append('左转')
    #                 self.spdsave.clear()
    #                 self.accelsave.clear()
    #
    #     for i in range(len(rgtobj)):
    #         if i == 0:
    #             isturning = False
    #             continue
    #
    #         if not isturning:
    #             if rgtobj[i] == 1 and rgtobj[i] != lefobj[i]:
    #                 start = cltobj[i]
    #                 self.spdsave.append(spdobj[i])
    #                 isturning = True
    #                 continue
    #             else:
    #                 continue
    #         # 转弯亮灯中
    #         else:
    #             a = cal_accel(i, spdobj, cltobj)
    #             self.accelsave.append(a)
    #             self.spdsave.append(spdobj[i])
    #             if rgtobj[i] == 0:
    #                 end = cltobj[i]
    #                 isturning = False
    #                 self.eventresult(start, end)
    #                 # 事件类型
    #                 self.type.append('右转')
    #                 self.spdsave.clear()
    #                 self.accelsave.clear()

    def turn_event_gps(self, spdobj, cltobj, lgtobj, latobj):
        lgtobj = lgtobj.dropna()
        latobj = latobj.dropna()
        # print(lgtobj.iloc[0]) #第一个值
        # print(lgtobj.index[0]) #第一个位置的索引
        for i in range(len(lgtobj)):
            # 初始化标记
            if i == 0 or i == 1:
                # 不能等于1的原因：若在第二行 是作为中心点，那么第一行为起点，起点的速度要计算加速度，会再用到上一行的速度，第一行的上一行就没有了，
                # 就会报错
                isturning = False
                continue
            # 最后一个点是算不了角度的，直接结束。
            if i == len(lgtobj) - 1:
                continue

            # 计算角度 需要3个点，遍历的i是中间的那个点，所以计算转弯起点是前一个点。转弯的加速度可能有减有加，算数学指标的时候记得区分。
            angle = cal_angle(i, lgtobj, latobj)
            if angle > 45:
                if not isturning:
                    isturning = True
                    start = cltobj[lgtobj.index[i - 1]]
                    # 因为gps数据有间隔，录入速度要把两个间隔之内的速度都录入
                    for j in range(lgtobj.index[i - 1], lgtobj.index[i]):
                        if j == lgtobj.index[i - 1]:
                            self.spdsave.append(spdobj[j])
                        self.spdsave.append(spdobj[j])
                        a = cal_accel(j, spdobj, cltobj)
                        t = get_timeduration(i, cltobj)

                        self.timeduration.append(t)
                        self.accelsave.append(a)

                # isturning = True
                for k in range(lgtobj.index[i - 1], lgtobj.index[i]):
                    if k == lgtobj.index[i - 1]:
                        self.spdsave.append(spdobj[k])
                    self.spdsave.append(spdobj[k])
                    a = cal_accel(k, spdobj, cltobj)
                    t = get_timeduration(i, cltobj)

                    self.timeduration.append(t)
                    self.accelsave.append(a)

            if angle < 45:
                if isturning:
                    end = cltobj[lgtobj.index[i]]
                    isturning = False
                    for l in range(lgtobj.index[i - 1], lgtobj.index[i]):
                        if l == lgtobj.index[i - 1]:
                            self.spdsave.append(spdobj[l])
                        self.spdsave.append(spdobj[l])
                        a = cal_accel(l, spdobj, cltobj)
                        t = get_timeduration(i, cltobj)

                        self.timeduration.append(t)
                        self.accelsave.append(a)

                    self.turn_eventresult(start, end)
                    # 事件类型
                    self.type.append('turn')

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
                dataisExists = os.path.exists(
                    self.datasetpath + folder + '/' + str(self.day) + self.filename_extenstion)
                if dataisExists:
                    df = pd.read_csv(
                        self.datasetpath + folder + '/' + str(self.day) + self.filename_extenstion,
                        encoding='gbk')
                    df.rename(columns={u'油门踏板开度(%)': 'trg', u'脉冲车速(km/h)': 'spd', u'刹车': 'brk', u'采集时间': 'clt',
                                       u'经度': 'lgt', u'纬度': 'lat'},
                              inplace=True)
                    trgobj = df['trg']
                    spdobj = df['spd']
                    cltobj = df['clt']
                    brkobj = df['brk']
                    lgtobj = df['lgt']
                    latobj = df['lat']
                    # # 列名占了一行 数据列从2开始
                    # # df['spd'][1] = spdobj[0]
                    # print(spdobj[0])
                    self.accel_event(trgobj, spdobj, cltobj)
                    self.brake_event(spdobj, cltobj, brkobj, trgobj)
                    # self.turn_event(spdobj, cltobj, lefobj, rgtobj)
                    self.turn_event_gps(spdobj, cltobj, lgtobj, latobj)

                    # 有些事件只有一个加速度，求标准差则是空值。
                    dic = {
                        '车辆编号': folder,
                        '开始时间': self.start,
                        '结束时间': self.endt,
                        '持续时间': self.durat,
                        '最大速度': self.spdmax,
                        '最小速度': self.spdmin,
                        '速度极差': self.spddif,
                        '速度标准差': self.spdstd,
                        '速度均值': self.spdmea,
                        '临界速度': self.spdcrt,
                        '最大加速度': self.amax,
                        '最小加速度': self.amin,
                        '加速度极差': self.adif,
                        '加速度标准差': self.astd,
                        '加速度均值': self.amea,
                        '首尾加速度和': self.ahead,
                        '加加速度最大值': self.jmax,
                        '加加速度最小值': self.jmin,
                        '加加速度极差': self.jdif,
                        '加加速度标准差': self.jstd,
                        '加加速度均值': self.jmea,
                        '加加速度首尾和': self.jhead,
                        '事件类型': self.type

                    }
                    data = DataFrame(dic)
                    data.sort_values(by='开始时间', inplace=True)
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
    # eventdetection.get_a()
    eventdetection.eventprocess()
    print('提取完毕')
    print('提取完毕!!!')
