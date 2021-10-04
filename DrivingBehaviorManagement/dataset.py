import pandas as pd
import os
'''
can ControllerAreaNetwork
ext 扩展
fault 故障
gps_sq 经纬度高度gps速度脉冲车速
spd 车速与转速_2秒一次
swt 刹车手刹离合转弯远光空调近光
'''
folder_name = ["031267", "077102", "078351", "078837", "080913", "082529", "090798", "098840", "108140", "112839"]
filename_extenstion = '.csv'
day = 20200901
datasetpath = "D:/wakeup/dataset/"
datapath = 'D:/wakeup/'

#创建目录
for folder in folder_name:
    isExists = os.path.exists(datasetpath + folder)
    if not isExists:
        os.makedirs(datasetpath + folder)
    else:
        continue

#数据合并
for folder in folder_name:
    while day < 20200931:
        dataisExists = os.path.exists(datapath + folder + '/spd/' + str(day) + filename_extenstion)
        if dataisExists:
            spd = pd.read_csv(datapath + folder + '/spd/' + str(day) + filename_extenstion)
            swt = pd.read_csv(datapath + folder + '/swt/' + str(day) + filename_extenstion)
            gps = pd.read_csv(datapath + folder + '/gps_sq/' + str(day) + filename_extenstion)
            swt.drop('数据号', axis=1, inplace=True)
            gps.drop('数据号', axis=1, inplace=True)
            print(spd.head())
            print("__")
            print(swt.head())
            print("__")
            print(gps.head())
            data = pd.merge(spd, swt, on=['采集时间', '存储时间'], how='left')
            data = pd.merge(data, gps, on=['采集时间', '存储时间'], how='left')
            data.to_csv('D:/wakeup/dataset/' + folder + '/' + str(day) + filename_extenstion, encoding='gbk')
            day += 1
        else:
            day += 1
    day = 20200901

print("数据合并完成")