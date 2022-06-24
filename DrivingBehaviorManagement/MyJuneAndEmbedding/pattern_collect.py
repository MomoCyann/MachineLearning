# -*- coding : utf-8 -*-
# coding: utf-8
import os
import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
import matplotlib
from math import sqrt,pow,acos
from pandas.core.frame import DataFrame

class PATTERNCOLLECT:

    def __init__(self):
        self.cars_num = ["077102"]
        self.filename_extenstion = '.csv'
        self.root = "D:/RX-105/wakeup/MyJuneAndEmbedding/"
        self.datasetpath = "D:/RX-105/wakeup/MyJuneAndEmbedding/dataset/"
        self.datapath = 'D:/RX-105/wakeup/MyJuneAndEmbedding/data/'
        self.eventpath = "D:/RX-105/wakeup/MyJuneAndEmbedding/event/"

        self.data = pd.read_csv(self.root + 'event_labeled.csv', encoding='gbk')
        self.accel_data = self.data[(self.data['事件类型'] == 'accel')]
        self.brake_data = self.data[(self.data['事件类型'] == 'brake')]
        self.turn_data = self.data[(self.data['事件类型'] == 'turn')]

        self.timegap = 180

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
        pattern_data.loc[(pattern_data['事件类型'] == 'accel') & (pattern_data['异常标签'] == 1), 'pattern_label'] = 'a'
        pattern_data.loc[(pattern_data['事件类型'] == 'accel') & (pattern_data['异常标签'] == -1), 'pattern_label'] = 'b'
        pattern_data.loc[(pattern_data['事件类型'] == 'accel') & (pattern_data['异常标签'] == -2), 'pattern_label'] = 'c'
        pattern_data.loc[(pattern_data['事件类型'] == 'brake') & (pattern_data['异常标签'] == 1), 'pattern_label'] = 'h'
        pattern_data.loc[(pattern_data['事件类型'] == 'brake') & (pattern_data['异常标签'] == -1), 'pattern_label'] = 'i'
        pattern_data.loc[(pattern_data['事件类型'] == 'brake') & (pattern_data['异常标签'] == -2), 'pattern_label'] = 'j'
        pattern_data.loc[(pattern_data['事件类型'] == 'turn') & (pattern_data['异常标签'] == 1), 'pattern_label'] = 'o'
        pattern_data.loc[(pattern_data['事件类型'] == 'turn') & (pattern_data['异常标签'] == -1), 'pattern_label'] = 'q'
        pattern_data.loc[(pattern_data['事件类型'] == 'turn') & (pattern_data['异常标签'] == -2), 'pattern_label'] = 'p'
        print(pattern_data.pattern_label.value_counts())
        return pattern_data

    def pattern_generate(self):
        pattern_data = self.label_transform()
        all_patterns = pd.DataFrame(columns=['车辆编号', '开始时间', '结束时间', '持续时间', 'pattern'])
        for cars in self.cars_num:
            m = 0
            the_car_data = pattern_data.loc[pattern_data['车辆编号'] == int(cars), :]
            start_time = the_car_data.iloc[m].loc['开始时间']
            end_time_event = the_car_data.iloc[m].loc['结束时间']
            end_time = the_car_data.iloc[-1].loc['开始时间']
            event_type = the_car_data.iloc[m].loc['pattern_label']
            while m < the_car_data.shape[0]:

                start_time_c = start_time  # 复制一个starttime用作记录时间起点
                event_gap = 0
                end_time_patn = int(cal_nexttime(start_time, self.timegap))
                time_highrisk = 0
                time_midrisk = 0
                time_safe = 0
                if the_car_data.iloc[m].loc['开始时间'] < end_time_patn:
                    #表示这段时间是有事件的
                    while start_time < end_time_patn:
                        # 高风险事件的时间。
                        if event_type == 'c' or event_type == 'j' or event_type == 'q':
                            time_highrisk += the_car_data.iloc[m].loc['持续时间']
                        # 中风险事件的时间
                        if event_type == 'b' or event_type == 'i' or event_type == 'p':
                            time_midrisk += the_car_data.iloc[m].loc['持续时间']
                        # 安全的时间
                        if event_type == 'a' or event_type == 'h' or event_type == 'o':
                            time_safe += the_car_data.iloc[m].loc['持续时间']

                        if end_time_event <= end_time_patn:
                            m += 1
                            event_gap += 1
                            if m >= the_car_data.shape[0]:
                                break
                            else:
                                start_time = the_car_data.iloc[m].loc['开始时间']
                                end_time_event = the_car_data.iloc[m].loc['结束时间']
                            event_type = the_car_data.iloc[m].loc['pattern_label']
                        else:
                            # 以最后一个事件的结束时间为pattern结束时间，并作为下一个P的开始时间
                            end_time_patn = the_car_data.iloc[m].loc['结束时间']
                            m += 1
                            event_gap += 1
                            if m >= the_car_data.shape[0]:
                                break
                            event_type = the_car_data.iloc[m].loc['pattern_label']
                            break
                    start_time = end_time_patn
                    pattern = the_car_data.iloc[m-event_gap:m, -1]
                    duration = cal_timeduration(start_time_c, end_time_patn)

                    # modify the pattern score
                    drive_time = time_safe + time_midrisk + time_highrisk
                    score_weight = np.array([100, -50, -100])
                    risk_time_ratio = np.array([time_safe/drive_time, time_midrisk/drive_time, time_highrisk/drive_time])
                    pattern_score = np.multiply(risk_time_ratio, score_weight).sum()
                    if pattern_score <= 0:
                        pattern_score = 1
                    all_patterns = all_patterns.append({'车辆编号': cars,
                                                        '开始时间': start_time_c,
                                                        '结束时间': end_time_patn,
                                                        '持续时间': duration,
                                                        '安全时间': time_safe,
                                                        '中风险时间': time_midrisk,
                                                        '高风险时间': time_highrisk,
                                                        '安全分数': pattern_score,
                                                        'pattern': ''.join(pattern.tolist())}, ignore_index=True)
                else:
                    start_time = end_time_patn
                    # duration = cal_timeduration(start_time_c, end_time_patn)
                    # all_patterns = all_patterns.append({'车辆编号': cars,
                    #                                     '开始时间': start_time_c,
                    #                                     '结束时间': end_time_patn,
                    #                                     '持续时间': duration,
                    #                                     '高风险时间': time_highrisk,
                    #                                     'pattern': ''}, ignore_index=True)
                print(m)



            print('one car complete?')
        print('all cars complete')
        all_patterns.to_csv(self.root + 'all_patterns_180s_50and100'+ self.filename_extenstion,
                        encoding='gbk')

def cal_timeduration(start, end):
    starttime = datetime.datetime.strptime(str(start), "%Y%m%d%H%M%S")
    endtime = datetime.datetime.strptime(str(end), "%Y%m%d%H%M%S")
    return (endtime - starttime).seconds

def cal_nexttime(start, timegap):
    starttime = datetime.datetime.strptime(str(start), "%Y%m%d%H%M%S")
    gap = timedelta(seconds=timegap)
    end = starttime + gap
    endtime = datetime.datetime.strftime(end, '%Y%m%d%H%M%S')
    return endtime

if __name__ == '__main__':
    pattern_generator = PATTERNCOLLECT()
    pattern_generator.pattern_generate()
