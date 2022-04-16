import numpy as np
import pandas as pd

datasetpath = 'E:/DrivingRisk/8-truck_data/'

# 合并TimeSeriesData和VideoReduceData
# 40S 的VideoReduceData都是一份 分心行为有Start和End的Sync时间

# 读取文件
Time_Series_Data = pd.read_csv(datasetpath + '8Truck_Public_Compiled.csv', low_memory=False)
Video_Reduce_Data = pd.read_csv(datasetpath + 'EightTruckEventVideoReducedData.csv', low_memory=False)


# 根据trip_id合并两个数据
Time_Series_Data_Merged = Time_Series_Data.merge(Video_Reduce_Data,how='left',on='trip_id')
# 保存数据
Time_Series_Data_Merged.to_csv(datasetpath + '8truck_Time_Series_Data_Merged.csv',encoding='UTF-8')

print('hello')