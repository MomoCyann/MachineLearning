import numpy as np
import pandas as pd

datasetpath = 'E:/DrivingRisk/100-car_data/'

# 合并TimeSeriesData和VideoReduceData
# 40S 的VideoReduceData都是一份 分心行为有Start和End的Sync时间

# 读取文件
Time_Series_Data_NearCrash = pd.read_csv(datasetpath + 'HundredCar_NearCrash_Public_Compiled.csv', low_memory=False)
Time_Series_Data_Crash = pd.read_csv(datasetpath + 'HundredCar_Crash_Public_Compiled.csv', low_memory=False)
Video_Reduce_Data = pd.read_csv(datasetpath + '100CarEventVideoReducedData_v1_5.csv', low_memory=False)

# 把crash加进去，去掉空行，按trip_id排序，重排索引。
Time_Series_Data = pd.concat([Time_Series_Data_NearCrash,Time_Series_Data_Crash])\
    .dropna(axis=0, how='all', thresh=None, subset=None, inplace=False)\
    .sort_values(by=['trip_id', 'sync'])\
    .reset_index(drop=True)

# 根据trip_id合并两个数据
Time_Series_Data_Merged = Time_Series_Data.merge(Video_Reduce_Data,how='left',on='trip_id')
# 保存数据
Time_Series_Data_Merged.to_csv(datasetpath + 'Time_Series_Data_Merged.csv',encoding='UTF-8')

print('hello')