import pandas as pd


root = "E:/wakeup/volatility_data/"
datasetpath = "E:/wakeup/volatility_data/dataset/"
datapath = 'E:/wakeup/volatility_data/data/'
eventpath = "E:/wakeup/volatility_data/event/"

data_row = pd.read_csv(root + 'allevents_washed_outlier.csv', encoding='gbk')

# data_outlier_row = pd.read_csv(root + 'allevents_washed_outlier_cluster.csv', encoding='gbk')
data_outlier_row = pd.read_csv(root + 'cluster_label.csv', encoding='gbk')
data = pd.merge(data_row,data_outlier_row,how='left')
data.loc[data['cluster标签']==0, '异常标签'] = 2
data.loc[data['cluster标签']==-1, '异常标签'] = 3
data.drop('Unnamed: 0', axis=1, inplace=True)
data.to_csv(root + 'allevents_washed_final_label.csv',
                   encoding='gbk')