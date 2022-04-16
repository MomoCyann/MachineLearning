import numpy as np
import pandas as pd

datasetpath = 'E:/DrivingRisk/100-car_data/'
# 读取文件
Time_Series_Data = pd.read_csv(datasetpath + 'Time_Series_Data_Merged_Labeled_no_pre.csv', encoding='UTF-8',
                               low_memory=False)
index_set = []

for trip in range(8296,9124):
    current_data = Time_Series_Data.loc[Time_Series_Data['trip_id'] == trip,]
    Sync_start = ((Time_Series_Data.loc[Time_Series_Data['trip_id'] == trip, 'sync']).tolist())[0]
    Sync_end = ((Time_Series_Data.loc[Time_Series_Data['trip_id'] == trip, 'sync']).tolist())[-1]
    start = Sync_start
    end = Sync_end
    while True:
        if start < Sync_end:
            # Time_Series_Data = Time_Series_Data.drop(
            #     current_data[(current_data.sync < (start+10)) & (current_data.sync > start)].index)
            index = current_data[(current_data.sync < (start+10)) & (current_data.sync > start)].index.values
            index_set += index.tolist()
            start+=10
            continue
        else:
            # Time_Series_Data = Time_Series_Data.drop(
            #     current_data[(current_data.sync < Sync_end) & (current_data.sync > start)].index)
            index = current_data[(current_data.sync < Sync_end) & (current_data.sync > start)].index.values
            index_set += index.tolist()
            # 保留每个trip的最后一行
            break
    print(trip)

Time_Series_Data = Time_Series_Data.drop(index=index_set)
# 样本标签计数
print(Time_Series_Data['label'].value_counts())

# 保存数据
input('Press any key to save the label')
Time_Series_Data.to_csv(datasetpath + 'Time_Series_Data_Merged_Labeled_no_pre_1s.csv',encoding='UTF-8', index=False)
print('Save Complete')