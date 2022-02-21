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

class FinalScore:

    def __init__(self):
        self.cars_num = ["031267", "077102", "078351", "078837", "080913", "082529",
                         "090798", "098840", "108140", "112839"]
        self.filename_extenstion = '.csv'
        self.root = "E:/wakeup/"
        self.datasetpath = "E:/wakeup/dataset/"
        self.datapath = 'E:/wakeup/data/'
        self.eventpath = "E:/wakeup/event/"

        self.event_data = pd.read_csv(self.root + 'allevents_label.csv', encoding='gbk')
        self.pattern_data = pd.read_csv(self.root + 'all_patterns.csv', encoding='gbk')
        self.all_features = pd.DataFrame(columns=['车辆编号', '开始时间', '安全事件比例','中风险事件','高风险事件',
                                                  '高分Pattern','低分Pattern',
                                                  '高风险时长Pattern','低风险时长Pattern','无风险时长Pattern'])

        self.time_start = 20200901000000
        self.time_limit = 20200931000000

    def filter(self):
        for car in self.cars_num:
            one_car_pattern = self.pattern_data.loc[self.pattern_data['车辆编号'] == int(car), :]
            one_car_event = self.event_data.loc[self.event_data['车辆编号'] == int(car), :]
            for time in range(self.time_start, self.time_limit, 1000000):
                one_day_event = one_car_event.loc[(one_car_event['开始时间'] >= time) &
                                                  (one_car_event['开始时间'] <= time+1000000)]
                high_risk_event = len(one_day_event.loc[one_day_event['风险等级'] == '高', :])
                mid_risk_event = len(one_day_event.loc[one_day_event['风险等级'] == '中', :])
                low_risk_event = len(one_day_event.loc[one_day_event['风险等级'] == '低', :])

                sum_event = high_risk_event + mid_risk_event + low_risk_event

                high_percent = high_risk_event/sum_event
                mid_percent = mid_risk_event/sum_event
                low_percent = low_risk_event/sum_event

                one_day_pattern = one_car_pattern.loc[(one_car_pattern['开始时间'] >= time) &
                                                  (one_car_pattern['开始时间'] <= time + 1000000)]
                high_score_pattern = len(one_day_pattern.loc[one_day_pattern['score'] >= 75, :])
                low_score_pattern = len(one_day_pattern.loc[one_day_pattern['score'] < 75, :])
                sum_pattern = high_score_pattern + low_score_pattern

                high_score_percent = high_score_pattern/sum_pattern
                low_score_percent = low_score_pattern/sum_pattern

                low_score_patt = one_day_pattern.loc[one_day_pattern['score'] < 75, :]
                high_risk_time = len(low_score_patt.loc[low_score_patt['高风险时间'] >= 8, :])
                low_risk_time = len(low_score_patt.loc[low_score_patt['高风险时间'] < 8, :])
                safe_time = len(low_score_patt.loc[low_score_patt['高风险时间'] == 0, :])

                high_risk_percent = high_risk_time/low_score_pattern
                low_risk_percent = low_risk_time/low_score_pattern
                safe_percent = safe_time/low_score_pattern

                self.all_features = self.all_features.append({'车辆编号': car,
                                                              '开始时间': time,
                                                              '安全事件比例': low_percent,
                                                              '中风险事件': mid_percent,
                                                              '高风险事件': high_percent,
                                                              '高分Pattern': high_score_percent,
                                                              '低分Pattern': low_score_percent,
                                                              '高风险时长Pattern': high_risk_percent,
                                                              '低风险时长Pattern': low_risk_percent,
                                                              '无风险时长Pattern': safe_percent}, ignore_index=True)
                print('one day complete')
            print('one car complete')

    def export_file(self):
        self.all_features.to_csv(self.root + 'all_features' + self.filename_extenstion,encoding='gbk')

if __name__ == '__main__':
    finalscore = FinalScore()
    finalscore.filter()
    finalscore.export_file()