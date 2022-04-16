import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler

datasetpath = 'E:/DrivingRisk/100-car_data/'
data_train_set = pd.read_csv(datasetpath + 'Time_Series_Data_Merged_Labeled_no_pre - 副本.csv', low_memory=False)
y = data_train_set['label']
y = y.astype('int')
X = data_train_set[['gas_pedal_position',
                    'speed_vehicle_composite',
                    'speed_gps_horizontal',
                    'yaw_rate',
                    'lateral_accel',
                    'longitudinal_accel',
                    'lane_markings_distance_left',
                    'lane_markings_distance_right',
                    'lane_markings_probability_left',
                    'lane_markings_probability_right',
                    'radar_forward_range_1',
                    'radar_forward_range_2',
                    'radar_forward_range_3',
                    'radar_forward_range_4',
                    'radar_forward_range_5',
                    'radar_forward_range_6',
                    'radar_forward_range_7',
                    'radar_rearward_range_1',
                    'radar_rearward_range_2',
                    'radar_rearward_range_3',
                    'radar_rearward_range_4',
                    'radar_rearward_range_5',
                    'radar_rearward_range_6',
                    'radar_rearward_range_7',
                    'radar_forward_range_rate_1',
                    'radar_forward_range_rate_2',
                    'radar_forward_range_rate_3',
                    'radar_forward_range_rate_4',
                    'radar_forward_range_rate_5',
                    'radar_forward_range_rate_6',
                    'radar_forward_range_rate_7',
                    'radar_rearward_range_rate_1',
                    'radar_rearward_range_rate_2',
                    'radar_rearward_range_rate_3',
                    'radar_rearward_range_rate_4',
                    'radar_rearward_range_rate_5',
                    'radar_rearward_range_rate_6',
                    'radar_rearward_range_rate_7',
                    'radar_forward_azimuth_1',
                    'radar_forward_azimuth_2',
                    'radar_forward_azimuth_3',
                    'radar_forward_azimuth_4',
                    'radar_forward_azimuth_5',
                    'radar_forward_azimuth_6',
                    'radar_forward_azimuth_7',
                    'radar_rearward_azimuth_1',
                    'radar_rearward_azimuth_2',
                    'radar_rearward_azimuth_3',
                    'radar_rearward_azimuth_4',
                    'radar_rearward_azimuth_5',
                    'radar_rearward_azimuth_6',
                    'radar_rearward_azimuth_7',
                    'light_intensity',
                    ]]
#X = StandardScaler().fit_transform(X)
X = X.corr()
plt.figure(1)
sns.heatmap(X,annot=False, vmax=1, square=True)#绘制new_df的矩阵热力图
plt.show()