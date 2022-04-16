import numpy as np
import pandas as pd

datasetpath = 'E:/DrivingRisk/100-car_data/'
# 读取文件
Time_Series_Data = pd.read_csv(datasetpath + 'Time_Series_Data_Merged.csv', encoding='UTF-8', low_memory=False)

# # 对列名带空格的列重命名
# Time_Series_Data.columns = [c.replace(' ', '_') for c in Time_Series_Data.columns]
# # 保存数据
# Time_Series_Data.to_csv(datasetpath + 'Time_Series_Data_Merged.csv',encoding='UTF-8')

# 先随便打打
Time_Series_Data['label'] = 0
# Time_Series_Data.loc[Time_Series_Data['Event_Severity'] == 'Near-Crash', 'Label'] = 0
# Time_Series_Data.loc[Time_Series_Data['Event_Severity'] == 'Crash', 'Label'] = 1

# 根据event start 和 event end来确定near crash 和 crash 事件的开始
# 提取所有的trip_id 8296 9123
for trip in range(8296,9124):
    start = ((Time_Series_Data.loc[Time_Series_Data['trip_id'] == trip, 'Event_Start']).tolist())[0]  # 危险事件再提前5s
    end = ((Time_Series_Data.loc[Time_Series_Data['trip_id'] == trip, 'Event_End']).tolist())[0]
    Time_Series_Data.loc[
        (Time_Series_Data['trip_id'] == trip)&(start<=Time_Series_Data['sync'])&(Time_Series_Data['sync']<=end), 'label'] = 1
    print(str(trip)+'已完成标记')
# 扫除异常值
Time_Series_Data.loc[Time_Series_Data['lateral_accel'] == '.', 'lateral_accel'] = 0
Time_Series_Data.loc[Time_Series_Data['longitudinal_accel'] == '.', 'longitudinal_accel'] = 0
Time_Series_Data.loc[Time_Series_Data['brake_on_off'] == '.', 'brake_on_off'] = 0

# 样本标签计数
print(Time_Series_Data['label'].value_counts())

# 保存数据
input('Press any key to save the label')
Time_Series_Data.to_csv(datasetpath + 'Time_Series_Data_Merged_Labeled_no_pre.csv',encoding='UTF-8', index=False)
print('Save Complete')