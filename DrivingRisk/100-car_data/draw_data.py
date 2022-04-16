import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal
import numpy as np

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
plt.figure(figsize=(8, 6))
x_sync = data_train_set.loc[data_train_set['trip_id'] == 8296, 'sync']
y_label = data_train_set.loc[data_train_set['trip_id'] == 8296,'label']
y_radar = data_train_set.loc[data_train_set['trip_id'] == 8296,'radar_forward_range_1']
x_risk = data_train_set.loc[(data_train_set['trip_id'] == 8296) & (data_train_set['label'] == 1), 'sync']
y_risk = data_train_set.loc[(data_train_set['trip_id'] == 8296) & (data_train_set['label'] == 1), 'radar_forward_range_1']

y_spd = data_train_set.loc[data_train_set['trip_id'] == 8296, 'speed_vehicle_composite']
y_spd_risk = data_train_set.loc[(data_train_set['trip_id'] == 8296) & (data_train_set['label'] == 1), 'speed_vehicle_composite']

y_acc = data_train_set.loc[data_train_set['trip_id'] == 8296, 'longitudinal_accel']
y_acc_risk = data_train_set.loc[(data_train_set['trip_id'] == 8296) & (data_train_set['label'] == 1), 'longitudinal_accel']

#plt.plot(x, y, label, color, linewidth, linestyle)

plt.plot(x_sync, y_radar,"b--", label='radar_forward_range')
plt.xlabel("Sync(0.1s)")
plt.ylabel("forward_range(ft)")
plt.scatter(x_risk, y_risk, c='red', marker='+')
plt.title("radar_forward_range_1")
plt.show()


# plt.plot(x_sync, y_acc,"b--")
# plt.xlabel("Sync(0.1s)")
# plt.ylabel("longitudinal_accel(g)")
# plt.plot(x_risk, y_acc_risk, 'red')
# plt.title("longitudinal_accel")
# plt.show()