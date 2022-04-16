import pandas as pd
import numpy as np
import os

root = 'E:/DrivingRisk/shrp2_open_access/'
datasetPath = 'E:/DrivingRisk/shrp2_open_access/all_csv/'
file_ID = 'File_ID_'
format_Name = '.csv'
#需要替换空值的列
space_column = ['vtti.heading_gps','vtti.head_rotation_z','vtti.head_rotation_y','vtti.head_rotation_x','vtti.head_position_z','vtti.head_position_y','vtti.head_position_x']

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
for IDnum in range(978,4617):
    # 判断是否存在某fileID的csv文件
    isExists = os.path.exists(datasetPath + file_ID + str(IDnum) + format_Name)
    if isExists:
        print("file detected: " + str(IDnum))
        new_Data = pd.read_csv(datasetPath + file_ID + str(IDnum) + format_Name, low_memory=False)
        # 先给这个文件加上File_ID列
        new_Data.insert(0, 'File_ID', IDnum)
        # 特定的几行的空格值转换为空值
        new_Data['vtti.latitude'] = new_Data['vtti.latitude'].apply(lambda x: np.NaN if str(x).isspace() else x)
        new_Data['vtti.longitude'] = new_Data['vtti.longitude'].apply(lambda x: np.NaN if str(x).isspace() else x)
        for column in space_column:
            new_Data[column] = new_Data[column].apply(lambda x: np.NaN if str(x).isspace() else x)
            idx_null = new_Data[column].isnull().sum(axis=0)
            # 统计空的个数
            print('剩余空值个数为：'+str(idx_null))
            # 向后传播非空值
            new_Data[column] = new_Data[column].fillna(method='ffill')
        # 选择GPS经纬度不为空的行，刚好间隔为1S
        new_Data = new_Data[new_Data['vtti.latitude'].notnull()]
        # 去掉这个文件的空行，引擎转速为空的行会被去掉
        # new_Data = new_Data.dropna(subset=['vtti.engine_rpm_instant'])
        # 合并到主表里
        if IDnum == 978:
            final_Data = new_Data
        else:
            final_Data = pd.concat([final_Data,new_Data]).reset_index(drop=True)
        print(final_Data.shape)

print(final_Data.shape)

# 保存数据
final_Data.to_csv(root + 'shrp2_data_1s.csv',encoding='UTF-8')
print('Complete！')