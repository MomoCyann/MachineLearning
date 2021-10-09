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
class SET:

    def __init__(self):
        self.folder_name = ["031267", "077102", "078351", "078837", "080913", "082529",
                            "090798", "098840", "108140", "112839"]
        self.filename_extenstion = '.csv'
        self.datasetpath = "D:/wakeup/dataset/"
        self.datapath = 'D:/wakeup/data/'
        self.day = 20200901

    def generate_folder(self):
        #创建目录
        for folder in self.folder_name:
            isExists = os.path.exists(self.datasetpath + folder)
            if not isExists:
                os.makedirs(self.datasetpath + folder)
            else:
                continue

    def merge(self):
        #数据合并
        for folder in self.folder_name:
            while self.day < 20200931:
                dataisExists = os.path.exists(self.datapath + folder + '/spd/' + str(self.day) + self.filename_extenstion)
                if dataisExists:
                    spd = pd.read_csv(self.datapath + folder + '/spd/' + str(self.day) + self.filename_extenstion)
                    swt = pd.read_csv(self.datapath + folder + '/swt/' + str(self.day) + self.filename_extenstion)
                    gps = pd.read_csv(self.datapath + folder + '/gps_sq/' + str(self.day) + self.filename_extenstion)
                    swt.drop('数据号', axis=1, inplace=True)
                    gps.drop(labels=['数据号','脉冲车速(km/h)','总里程'], axis=1, inplace=True)
                    print(spd.head())
                    print("__")
                    print(swt.head())
                    print("__")
                    print(gps.head())
                    data = pd.merge(spd, swt, on=['采集时间', '存储时间'], how='left')
                    data = pd.merge(data, gps, on=['采集时间', '存储时间'], how='left')
                    data.to_csv(self.datasetpath + folder + '/' + str(self.day) + self.filename_extenstion,
                                encoding='gbk',)
                    self.day += 1
                else:
                    self.day += 1
            self.day = 20200901
        print("数据合并完成")

if __name__ == "__main__":
    dataset = SET()
    dataset.generate_folder()
    dataset.merge()