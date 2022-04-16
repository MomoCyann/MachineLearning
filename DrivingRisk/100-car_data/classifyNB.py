
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# 读取数据集
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
std = StandardScaler().fit(X)
X = std.transform(X)

f1_scores = []
precision = []
recall = []
acc=[]


gnb = GaussianNB()
skf = StratifiedKFold(n_splits=5, shuffle=True)
for train_index, test_index in skf.split(X, y):
    clone_model = clone(gnb)

    X_train_folds = X[train_index]
    y_train_folds = y[train_index]
    X_test_fold = X[test_index]
    y_test_fold = y[test_index]

    clone_model.fit(X_train_folds, y_train_folds)
    test_labels_pred = clone_model.predict(X_test_fold)
    accuracy_score_fold = accuracy_score(y_test_fold, test_labels_pred)
    precision_fold = precision_score(y_test_fold, test_labels_pred, average="binary")
    f1_score_fold = f1_score(y_test_fold, test_labels_pred, average="binary")
    recall_fold = recall_score(y_test_fold, test_labels_pred, average='binary')
    precision.append(precision_fold)
    f1_scores.append(f1_score_fold)
    recall.append(recall_fold)
    acc.append(accuracy_score_fold)
    print(classification_report(y_test_fold, test_labels_pred))

print(np.array(acc).mean())
print("1类平均精度为：")
print(np.array(precision).mean())
print('1类平均召回率为：')
print(np.array(recall).mean())
print('1类平均f1 score为：')
print(np.array(f1_scores).mean())


# param = {"n_estimators": range(1,200)}
# gs = GridSearchCV(estimator=rfc, param_grid=param, scoring='f1')
# gs.fit(X, y)
# print(gs.best_score_)
# print(gs.best_params_)