import pandas as pd
import numpy as np
import os

root = 'E:/DrivingRisk/shrp2_open_access/'
datasetPath = 'E:/DrivingRisk/shrp2_open_access/all_csv/'
file_ID = 'File_ID_'
format_Name = '.csv'
# 先读取一份作为基底

final_Data = pd.read_csv(datasetPath + 'File_ID_978.csv', low_memory=False)
print(final_Data.shape)

# 添加一列作为文件标识
final_Data.insert(0, 'File_ID', 978)
print(final_Data.shape)

    # \
    # .dropna(axis=0, how='all', thresh=None, subset=None, inplace=False)\
    # .sort_values(by=['trip_id', 'sync'])\
    # .reset_index(drop=True)

# 根据File_ID循环concat每一个csv文件
for IDnum in range(990,4617):
    # 判断是否存在某fileID的csv文件
    isExists = os.path.exists(datasetPath + file_ID + str(IDnum) + format_Name)
    if isExists:
        print("file detected: " + str(IDnum))
        new_Data = pd.read_csv(datasetPath + file_ID + str(IDnum) + format_Name, low_memory=False)
        # 先给这个文件加上File_ID列
        new_Data.insert(0, 'File_ID', IDnum)
        # 去掉这个文件的空行，引擎转速为空的行会被去掉
        new_Data = new_Data.dropna(subset=['vtti.engine_rpm_instant'])
        # 合并到主表里
        final_Data = pd.concat([final_Data,new_Data])
        print(final_Data.shape)

print(final_Data.shape)

# 保存数据
final_Data.to_csv(root + 'shrp2_data.csv',encoding='UTF-8')
print('Complete！')