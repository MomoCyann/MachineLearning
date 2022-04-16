from keras.models import load_model
import shap
import tensorflow as tf
from tensorflow import keras
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.model_selection import StratifiedShuffleSplit


model_path = "E:/MachineLearning/DrivingRisk/100-car_data/checkpoints/"

def load_data():
    datasetpath = 'E:/DrivingRisk/100-car_data/'
    data_train_set = pd.read_csv(datasetpath + 'Time_Series_Data_Merged_Labeled_no_pre.csv', low_memory=False)
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
                        'radar_forward_ID_1',
                        'radar_forward_ID_2',
                        'radar_forward_ID_3',
                        'radar_forward_ID_4',
                        'radar_forward_ID_5',
                        'radar_forward_ID_6',
                        'radar_forward_ID_7',
                        'radar_rearward_ID_1',
                        'radar_rearward_ID_2',
                        'radar_rearward_ID_3',
                        'radar_rearward_ID_4',
                        'radar_rearward_ID_5',
                        'radar_rearward_ID_6',
                        'radar_rearward_ID_7',
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
    # 标准化数据
    X = StandardScaler().fit_transform(X)

    # # 归一化
    # X = MinMaxScaler().fit_transform(X)

    y = y.values
    return X, y

def create_model():
    tf.random.set_seed(120)
    model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(120, activation='relu', input_shape=(67,)),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer='Adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    # Adam默认学习率0.001

    return model



model = create_model()
model.load_weights(model_path+'weights.57-0.5327.hdf5', by_name=True)
X, y = load_data()
features = ['gas_pedal_position',
                        'speed_vehicle_composite',
                        'speed_gps_horizontal',
                        'yaw_rate',
                        'lateral_accel',
                        'longitudinal_accel',
                        'lane_markings_distance_left',
                        'lane_markings_distance_right',
                        'lane_markings_probability_left',
                        'lane_markings_probability_right',
                        'radar_forward_ID_1',
                        'radar_forward_ID_2',
                        'radar_forward_ID_3',
                        'radar_forward_ID_4',
                        'radar_forward_ID_5',
                        'radar_forward_ID_6',
                        'radar_forward_ID_7',
                        'radar_rearward_ID_1',
                        'radar_rearward_ID_2',
                        'radar_rearward_ID_3',
                        'radar_rearward_ID_4',
                        'radar_rearward_ID_5',
                        'radar_rearward_ID_6',
                        'radar_rearward_ID_7',
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
                        ]
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=120)
for train_index, test_index in split.split(X, y):
    x_train = X[train_index]
    y_train = y[train_index]
    x_val = X[test_index]
    y_val = y[test_index]
    background_data = x_train[np.random.choice(x_train.shape[0], 100, replace=False)]
    explainer = shap.DeepExplainer(model, background_data)
    explain_data = x_train[np.random.choice(x_train.shape[0], 10, replace=False)]
    shap_values = explainer.shap_values(explain_data)
    shap.summary_plot(shap_values, explain_data)


