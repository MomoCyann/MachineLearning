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
        self.rootpath = "E:/wakeup/"
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
        # 计算加速度 和 输出时间
        for i in range(len(spdobj)):
            if i == 0:
                continue
            delta_v = (spdobj[i] - spdobj[i - 1]) / 3.6  # m/s
            delta_t = cal_timeduration(cltobj[i - 1], cltobj[i])
            a = delta_v / delta_t
            self.accelsave.append(a)
            self.time.append(cltobj[i - 1])

        j = 0
        for i in range(len(self.accelsave)):
            if self.accelsave[j] == 0:
                self.accelsave.pop(j)
            else:
                j += 1

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
        absolute_a = np.maximum(np.array(self.accelsave),-np.array(self.accelsave))
        self.amea.append(np.mean(absolute_a))

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

                    #计算速度和加速度
                    self.get_a_and_v(spdobj, cltobj)
                    #检查是否是 一天没有出车导致spdsave为空
                    if self.spdsave:
                        self.get_result()

                    self.daytime.append(self.day)
                    self.day += 1
                else:
                    self.day += 1
            dic = {
                '数据日期': self.daytime,
                '速度极差': self.spddif,
                '速度标准差': self.spdstd,
                '速度均值': self.spdmea,
                '最大加速度': self.amax,
                '最大刹车加速度': self.amin,
                '加速度极差': self.adif,
                '加速度标准差': self.astd,
                '加速度均值': self.amea,
            }
            data = DataFrame(dic)

            #设备号添加列
            data['设备号'] = folder
            item = data['设备号']
            data.drop(labels=['设备号'], axis=1, inplace=True)
            data.insert(0, '设备号', item)

            data.sort_values(by='数据日期', inplace=True)
            data_new = data.reset_index(drop=True)
            data_new.to_csv(self.eventpath + folder + '/' + 'event' + self.filename_extenstion,
                            encoding='gbk')
            print("complete")
            self.clearevent()
            self.day = 20200901

    def speed_level_and_merge(self):
        general_data = pd.read_excel(self.rootpath + 'general_data.xlsx')
        speed_data = general_data[['设备号', '数据日期', '怠速时长(s)', '行驶时长(s)', '40公里超速时长(s)',
                                   '50公里超速时长(s)', '60公里超速时长(s)', '70公里超速时长(s)', '80公里超速时长(s)',
                                   '90公里超速时长(s)', '100公里超速时长(s)']]
        x = pd.DataFrame(speed_data)
        for i in range(x.shape[0]):
            if x.loc[i, '行驶时长(s)'] == 0:
                continue
            x.loc[i, 'over40'] = x.loc[i, '40公里超速时长(s)'] / x.loc[i, '行驶时长(s)']
            x.loc[i, 'over50'] = x.loc[i, '50公里超速时长(s)'] / x.loc[i, '行驶时长(s)']
            x.loc[i, 'over60'] = x.loc[i, '60公里超速时长(s)'] / x.loc[i, '行驶时长(s)']
            x.loc[i, 'over70'] = x.loc[i, '70公里超速时长(s)'] / x.loc[i, '行驶时长(s)']
            x.loc[i, 'over80'] = x.loc[i, '80公里超速时长(s)'] / x.loc[i, '行驶时长(s)']
            x.loc[i, 'over90'] = x.loc[i, '90公里超速时长(s)'] / x.loc[i, '行驶时长(s)']
            x.loc[i, 'over100'] = x.loc[i, '100公里超速时长(s)'] / x.loc[i, '行驶时长(s)']

        # 合并
        useful_data = general_data[['设备号', '数据日期', '过长怠速时长(s)', '急加速时长(s)',
                                    '急减速时长(s)', '空档滑行时长(s)', '疲劳驾驶时长(s)', '长刹车时长(s)', '长离合时长(s)',
                                    '大踩油门时长(s)', '停车踩油门时长(s)', '立即起步时长(s)', '立即停车时长(s)']]
        merge_data = pd.merge(x, useful_data, on=['设备号', '数据日期'], how='left')
        #双索引
        merge_data.set_index(['设备号', '数据日期'], inplace=True)
        count = 0
        for folder in self.folder_name:
            waiting_data = pd.read_csv(
                self.eventpath + folder + '/' + 'event' + self.filename_extenstion, encoding='gbk')
            waiting_data['设备号'] = waiting_data['设备号'].astype(int)
            waiting_data.set_index(['设备号', '数据日期'], inplace=True)
            if count == 0:
                merge_data = pd.merge(merge_data, waiting_data, on=['设备号', '数据日期'], how='outer')
            else:
                merge_data.combine_first(waiting_data)
            count+=1
        merge_data.to_csv(self.rootpath + 'general_data_merged' + self.filename_extenstion, encoding='gbk')


if __name__ == '__main__':
    eventdetection = EVENT()
    #eventdetection.eventprocess()
    # print('提取完毕')
    # print('提取完毕!!!')
    eventdetection.speed_level_and_merge()

