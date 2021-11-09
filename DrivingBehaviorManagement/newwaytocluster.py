import os
import pandas as pd
import numpy as np
import datetime
from pandas.core.frame import DataFrame


def cal_timeduration(start, end):
    starttime = datetime.datetime.strptime(str(start), "%Y%m%d%H%M%S")
    endtime = datetime.datetime.strptime(str(end), "%Y%m%d%H%M%S")
    return (endtime - starttime).seconds


class EVENT:

    def __init__(self):
        self.folder_name = ["031267", "077102", "078351", "078837", "080913", "082529",
                            "090798", "098840", "108140", "112839"]
        self.filename_extenstion = '.csv'
        self.datasetpath = "E:/wakeup/dataset/"
        self.datapath = 'E:/wakeup/data/'
        self.eventpath = "E:/wakeup/cluster/"
        self.day = 20200901

        # self.start = []
        # self.endt = []
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
        self.daytime = []

        self.time = []
        self.all_a = []

        self.accelsave = []
        self.spdsave = []

    def get_a_and_v(self, spdobj, cltobj):
        # df = pd.read_csv(
        #     self.datasetpath + "031267" + '/' + str(self.day) + self.filename_extenstion,
        #     encoding='gbk')
        # df.rename(columns={u'脉冲车速(km/h)': 'spd', u'刹车': 'brk', u'采集时间': 'clt', u'存储时间': 'svt',
        #                    u'左转向灯': 'lef', u'右转向灯': 'rgt'},
        #           inplace=True)
        # spdobj = df['spd']
        # cltobj = df['clt']
        self.spdsave = spdobj.tolist()
        j = 0
        for i in range(len(self.spdsave)):
            if self.spdsave[j] == 0:
                self.spdsave.pop(j)
            else:
                j += 1
        #计算加速度 和 输出时间
        for i in range(len(spdobj)):
            if i == 0:
                continue
            delta_v = (spdobj[i] - spdobj[i - 1]) / 3.6  # m/s
            delta_t = cal_timeduration(cltobj[i - 1], cltobj[i])
            a = delta_v / delta_t
            self.accelsave.append(a)
            self.time.append(cltobj[i - 1])

    def get_result(self):
        # 刹车结束 结算各项指标
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

        self.spdsave.clear()
        self.accelsave.clear()

    def clearevent(self):
        self.durat.clear()
        self.spddif.clear()
        self.spdstd.clear()
        self.spdmea.clear()
        self.amax.clear()
        self.amin.clear()
        self.adif.clear()
        self.astd.clear()
        self.amea.clear()
        self.daytime.clear()

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
                    df.rename(columns={u'脉冲车速(km/h)': 'spd', u'采集时间': 'clt', u'存储时间': 'svt'},
                              inplace=True)
                    spdobj = df['spd']
                    cltobj = df['clt']
                    self.get_a_and_v(spdobj, cltobj)
                    self.get_result()

                    self.daytime.append(self.day)
                    self.day += 1
                else:
                    self.day += 1
            dic = {
                'day': self.daytime,
                'spddif': self.spddif,
                'spdstd': self.spdstd,
                'spdmea': self.spdmea,
                'amax': self.amax,
                'amin': self.amin,
                'adif': self.adif,
                'astd': self.astd,
                'amea': self.amea,
                'ahead': self.ahead,
            }
            data = DataFrame(dic)
            data.sort_values(by='day', inplace=True)
            data_new = data.reset_index(drop=True)
            data_new.to_csv(self.eventpath + folder + '/' + str(self.day) + 'event' + self.filename_extenstion,
                            encoding='gbk')
            print("complete")
            self.clearevent()
            self.day = 20200901


if __name__ == '__main__':
    eventdetection = EVENT()
    eventdetection.eventprocess()
    print('提取完毕')
    print('提取完毕!!!')