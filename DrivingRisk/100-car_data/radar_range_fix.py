import numpy as np
import pandas as pd

datasetpath = 'E:/DrivingRisk/100-car_data/'
# 读取文件
Time_Series_Data = pd.read_csv(datasetpath + 'Time_Series_Data_Merged_Labeled_no_pre.csv', encoding='UTF-8', low_memory=False)


# range等于0的变为9999 不然就表示距离为0了
# 提取所有的trip_id 8296 9123
for trip in range(8296,9124):
    for num in range(1,8):
        Time_Series_Data.loc[Time_Series_Data['radar_forward_range_'+str(num)] == 0, 'radar_forward_range_'+str(num)] = 9999
        Time_Series_Data.loc[Time_Series_Data['radar_rearward_range_' + str(num)] == 0, 'radar_rearward_range_' + str(num)] = 9999
    print(str(trip)+'已完成标记')


# 保存数据
input('Press any key to save the label')
Time_Series_Data.to_csv(datasetpath + 'Time_Series_Data_Merged_Labeled_no_pre_fix.csv',encoding='UTF-8', index=False)
print('Save Complete')