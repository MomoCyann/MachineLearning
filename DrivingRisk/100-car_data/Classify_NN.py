import tensorflow as tf
from tensorflow import keras
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import StratifiedShuffleSplit
import keras_tuner as kt
from collections import Counter
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler

os.environ["CUDA_VISIBLE_DEVICES"]="0"

#在每个epoch的末尾计算f1
class Metrics(tf.keras.callbacks.Callback):
    def __init__(self, valid_data):
        super(Metrics, self).__init__()
        self.validation_data = valid_data

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_predict = np.argmax(self.model.predict(self.validation_data[0]), -1)
        val_targ = self.validation_data[1]
        if len(val_targ.shape) == 2 and val_targ.shape[1] != 1:
            val_targ = np.argmax(val_targ, -1)

        print("\n"+classification_report(val_targ, val_predict))
        # report = classification_report(val_targ, val_predict, output_dict=True)

        _val_f1 = f1_score(val_targ, val_predict, average='binary')
        _val_recall = recall_score(val_targ, val_predict, average='binary')
        _val_precision = precision_score(val_targ, val_predict, average='binary')

        # _risk_recall = report['1']['recall']
        # _risk_precison = report['1']['precision']
        # _risk_f1 = report['1']['f1-score']

        logs['val_precision'] = _val_precision
        logs['val_recall'] = _val_recall
        logs['val_f1'] = _val_f1

        # logs['risk_precision'] = _risk_precison
        # logs['risk_recall'] = _risk_recall
        # logs['risk_f1'] = _risk_f1


        return

# 读取数据集
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

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0


# 搭建网络
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

# 搜索模型最佳超参
def create_model_search(hp):
    tf.random.set_seed(120)
    model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(units=hp.Int('units',
                                        min_value=8,
                                        max_value=256,
                                        step=8), activation='relu', input_shape=(53,)),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate',
                      values=[1e-3])),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    # Adam默认学习率0.001

    return model

def f1_callback():
    if not os.path.exists('../checkpoints'):
        os.makedirs('../checkpoints')
    # 按照 val_f1 保存模型
    ck_callback = tf.keras.callbacks.ModelCheckpoint('E:/MachineLearning/DrivingRisk/100-car_data/checkpoints/weights.{epoch:02d}-{val_f1:.4f}.hdf5',
                                                     monitor='val_f1',
                                                     mode='max', verbose=2,
                                                     save_best_only=True,
                                                     save_weights_only=True)
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir='E:/MachineLearning/DrivingRisk/100-car_data/logs',
                                                 profile_batch=0,
                                                 histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                                                 write_graph=True,  # 是否存储网络结构图
                                                 write_grads=True,  # 是否可视化梯度直方图
                                                 write_images=True,  # 是否可视化参数
                                                 embeddings_freq=0,
                                                 embeddings_layer_names=None,
                                                 embeddings_metadata=None)
    return ck_callback, tb_callback

def main():
    # load data
    X, y = load_data()

    # F1 callback
    ck_callback, tb_callback = f1_callback()

    # 10折交叉
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=120)
    avg_accuracy = 0
    avg_loss = 0
    for train_index, test_index in skf.split(X, y):
        model = create_model()

        X_train_folds = X[train_index]
        y_train_folds = y[train_index]
        X_test_fold = X[test_index]
        y_test_fold = y[test_index]

        model.fit(X_train_folds, y_train_folds,
                  epochs=100,
                  batch_size=32,
                  validation_data=(X_test_fold, y_test_fold),
                  verbose=2,
                  class_weight={0: 1, 1: 3.2},
                  callbacks=[
                      Metrics(valid_data=(X_test_fold, y_test_fold)),
                      ck_callback,
                      tb_callback]
                  )

        print('Model evaluation: ', model.evaluate(X_test_fold, y_test_fold))
        avg_accuracy += model.evaluate(X_test_fold, y_test_fold)[1]
        avg_loss += model.evaluate(X_test_fold, y_test_fold)[0]

        # model.evaluate(x_test,  y_test, verbose=2)

    print("K fold average accuracy: {}".format(avg_accuracy / 10))
    print("K fold average accuracy: {}".format(avg_loss / 10))


def search():
    # load data
    X, y = load_data()

    # F1 callback
    ck_callback, tb_callback = f1_callback()

    # 调参
    # 划分数据集
    # x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=120)
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=120)
    for train_index, test_index in split.split(X, y):
        x_train = X[train_index]
        y_train = y[train_index]
        x_val = X[test_index]
        y_val = y[test_index]

        tuner = kt.Hyperband(
                            create_model_search,
                            objective='val_loss',  # 优化目标为精度'val_accuracy'（最小化目标）
                            max_epochs=10,
                            factor=3,
                            directory='tuner',
                            project_name='100-car_data')

        # 搜索空间综述
        print(tuner.search_space_summary())

        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

        tuner.search(x_train,
                     y_train,
                     verbose=1,  # just slapping this here bc jupyter notebook. The console out was getting messy.
                     epochs=3,
                     validation_data=(x_val, y_val),
                     class_weight={0: 1, 1: 7},
                     callbacks=[stop_early]
        )

        # Get the optimal hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        print(f"""
        The hyperparameter search is complete. The optimal number of units in the first densely-connected
        layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
        is {best_hps.get('learning_rate')}.
        """)

def batch_adjust():
    # load data
    X, y = load_data()
    print(Counter(y))
    # x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=120)

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=120)
    for train_index, test_index in split.split(X, y):
        x_train = X[train_index]
        y_train = y[train_index]
        x_val = X[test_index]
        y_val = y[test_index]
        print(Counter(y_val))
        # F1 callback
        ck_callback, tb_callback = f1_callback()

        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

        model = create_model()
        model.fit(x_train, y_train,
                  epochs=200,
                  batch_size=64,
                  validation_data=(x_val, y_val),
                  verbose=2,
                  class_weight={0: 1, 1: 6.88},
                  callbacks=[
                      Metrics(valid_data=(x_val, y_val)),
                      ck_callback,
                      tb_callback,
                      stop_early]
                  )


#main()

#search()

batch_adjust()