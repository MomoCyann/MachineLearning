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
        self.folder_name = [#"031267",
                            "077102", "078351", "078837", "080913", "082529",
                            "090798", "098840", "108140"]
            #"112839"]
        self.filename_extenstion = '.csv'
        self.datasetpath = "E:/wakeup/dataset/"
        self.datapath = 'E:/wakeup/data/'
        self.day = 20200901

    def generate_folder(self):
        # 创建目录
        for folder in self.folder_name:
            isExists = os.path.exists(self.datasetpath + folder)
            if not isExists:
                os.makedirs(self.datasetpath + folder)
            else:
                continue

    def merge(self):
        # 数据合并
        for folder in self.folder_name:
            while self.day < 20200931:
                dataisExists = os.path.exists(
                    self.datapath + folder + '/spd/' + str(self.day) + self.filename_extenstion)
                if dataisExists:
                    spd = pd.read_csv(self.datapath + folder + '/spd/' + str(self.day) + self.filename_extenstion)
                    swt = pd.read_csv(self.datapath + folder + '/swt/' + str(self.day) + self.filename_extenstion)
                    gps = pd.read_csv(self.datapath + folder + '/gps_sq/' + str(self.day) + self.filename_extenstion)
                    gas = pd.read_csv(self.datapath + folder + '/can/' + str(self.day) + '/can_油门踏板开度(208).csv')
                    swt.drop('数据号', axis=1, inplace=True)
                    gps.drop(labels=['数据号', '脉冲车速(km/h)', '总里程'], axis=1, inplace=True)
                    gas.drop(labels=['数据号', '类型'], axis=1, inplace=True)
                    print(spd.head())
                    print("__")
                    print(swt.head())
                    print("__")
                    print(gps.head())
                    print("__")
                    print(gas.head())
                    data = pd.merge(spd, swt, on=['采集时间'], how='outer')
                    data = pd.merge(data, gps, on=['采集时间'], how='outer')
                    data = pd.merge(data, gas, on=['采集时间'], how='outer')
                    data.sort_values("采集时间", inplace=True)
                    data[u'脉冲车速(km/h)'] = data[u'脉冲车速(km/h)'].fillna(data[u'脉冲车速(km/h)'].interpolate())
                    data[u'油门踏板开度(%)'] = data[u'油门踏板开度(%)'].fillna(data[u'油门踏板开度(%)'].interpolate())
                    data[u'脉冲车速(km/h)'] = data[u'脉冲车速(km/h)'].fillna(method='bfill')
                    data[u'油门踏板开度(%)'] = data[u'油门踏板开度(%)'].fillna(method='bfill')
                    data.drop(labels=['存储时间_x', '存储时间_y', '数据号'], axis=1, inplace=True)
                    data['No.'] = data.apply(lambda x: folder, axis=1)
                    #调换一下编号列位置
                    numb = data['No.']
                    data.drop(labels=['No.'], axis=1, inplace=True)
                    data.insert(0, 'No.', numb)
                    data_new = data.reset_index(drop=True)
                    order = ['No.', '采集时间', '脉冲车速(km/h)', '油门踏板开度(%)', '刹车', '纬度', '经度', '高度', '转速(转/分)',
                             '总里程(KM)']
                    data_new = data_new[order]
                    data_new.to_csv(self.datasetpath + folder + '/' + str(self.day) + self.filename_extenstion,
                                    encoding='gbk', )
                    self.day += 1
                else:
                    self.day += 1
            self.day = 20200901
        print("数据合并完成")


if __name__ == "__main__":
    dataset = SET()
    dataset.generate_folder()
    dataset.merge()
