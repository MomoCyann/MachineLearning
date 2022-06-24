from keras.datasets import mnist
from keras.layers import Dense, LSTM, Dropout
from keras.utils.np_utils import *
from keras.models import Sequential
import os
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
#parameters for LSTM
nb_lstm_outputs = 100  #神经元个数
nb_input_vector = 9 #特征维度
root = "D:/RX-105/wakeup/MyJuneAndEmbedding/"
#data preprocessing: tofloat32, normalization, one_hot encoding
#load data
data = pd.read_csv(root + 'all_patterns_180s_50and100.csv', encoding='gbk')
#onehot encode for each pattern
pattern = data['pattern']
score = np.array(data['安全分数']).reshape(-1,1)

"""归一化"""
scaler = MinMaxScaler()
score = scaler.fit_transform(score)

all_labels = ['a', 'b', 'c', 'h', 'i', 'j', 'o', 'p', 'q']

# 删除大于30的长度pattern 只有4个
for i in range(len(pattern)):
    if len(pattern[i]) >= 30:
        pattern.drop(labels=i, inplace=True)
        score = np.delete(score, i, 0)
pattern = pattern.reset_index(drop=True)
# 找到最大长度的某个pattern
maxlen=0
for i in range(len(pattern)):
    if len(pattern[i]) >= maxlen:
        maxlen=len(pattern[i])
#encoding
pattern_onehot = np.empty((len(pattern), maxlen, 9))
for i in range(len(pattern)):
    pattern_encode = np.zeros((maxlen,9))
    for j in range(len(pattern[i])):
        encode = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        encode[all_labels.index(pattern[i][j])] = 1
        pattern_encode[j] = encode
    pattern_onehot[i] = pattern_encode
print(pattern_onehot.shape)
print(pattern_onehot[0].shape)
# 训练集80% 测试集20%
x_train = pattern_onehot[0:6683]
y_train = np.array(score[0:6683])
x_test = pattern_onehot[6683:-1]
y_test = np.array(score[6683:-1])
x_short_test = pattern_onehot[8250:-1]
y_short_test = np.array(score[8250:-1])



#build model
model = Sequential()
model.add(LSTM(units=nb_lstm_outputs, input_shape=(None, nb_input_vector)))
model.add(Dropout(0.2))

model.add(Dense(1))

#compile:loss, optimizer, metrics
model.compile(loss='mean_squared_error', optimizer='adam')

#checkpoint
# checkpoint_save_path = "./checkpoint/mnist.ckpt"
# if os.path.exists(checkpoint_save_path + '.index'):
#     print('-------------load the model-----------------')
#     model.load_weights(checkpoint_save_path)
#
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
#                                                  save_weights_only=True,
#                                                  save_best_only=True)

#train: epcoch, batch_size
history = model.fit(x_train, y_train, epochs=20, batch_size=64, validation_data=(x_test, y_test), validation_freq=1, verbose=1)
model.summary()
# score = model.evaluate(x_test, y_test,batch_size=128, verbose=1)
# print(score)

###############################################    show   ###############################################

# 显示训练集和验证集的acc和loss曲线
loss = history.history['loss']
val_loss = history.history['val_loss']

# plt.subplot(1, 2, 1)
# plt.plot(acc, label='Training Accuracy')
# plt.plot(val_acc, label='Validation Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.legend()

# plt.subplot(1, 2, 2)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

plt.plot(loss, label='Training Loss', c="deepskyblue")
plt.plot(val_loss, label='Validation Loss', c="salmon")
plt.title('Training and Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# 预测
#yhat = model.predict(x_test)
yhat = model.predict(x_short_test)

# 原始y逆标准化
#inv_y = scaler.inverse_transform(y_test)
inv_y = scaler.inverse_transform(y_short_test)

# # 预测y 逆标准化
inv_yhat = scaler.inverse_transform(yhat)

# 计算 R2
r_2 = r2_score(inv_y, inv_yhat)
print('Test r_2: %.3f' % r_2)
# 计算MAE
mae = mean_absolute_error(inv_y, inv_yhat)
print('Test MAE: %.3f' % mae)
# 计算RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
plt.plot(inv_y, c="deepskyblue", label='真实数据')
inv_yhat_onehot = np.load('yhat_onehot.npy')
plt.plot(inv_yhat_onehot, c="salmon", label='预测数据')
#plt.plot(inv_yhat, c="salmon", label='预测数据')
plt.legend()
plt.title('Intensity Prediction')
# 坐标轴
plt.gca().set(xlabel='驾驶段落', ylabel='Score')
plt.show()
'''
Test r_2: 0.759
Test MAE: 5.085
Test RMSE: 10.295
'''
'''
Test r_2: 0.764
Test MAE: 5.151
Test RMSE: 10.187
'''
'''
Test r_2: 0.768
Test MAE: 6.600
Test RMSE: 10.088
'''
'''
Test r_2: 0.749
Test MAE: 6.263
Test RMSE: 10.510
'''
input('press any key to save the predict yhat array')
np.save('yhat_onehot.npy',inv_yhat)
