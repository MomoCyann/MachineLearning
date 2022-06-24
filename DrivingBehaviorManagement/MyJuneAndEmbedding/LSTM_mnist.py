from keras.datasets import mnist
from keras.layers import Dense, LSTM
from keras.utils.np_utils import *
from keras.models import Sequential
import os
import tensorflow as tf
from matplotlib import pyplot as plt
#parameters for LSTM
nb_lstm_outputs = 30  #神经元个数
nb_time_steps = 28  #时间序列长度
nb_input_vector = 28 #输入序列

#data preprocessing: tofloat32, normalization, one_hot encoding
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255 #灰度值除以255放缩到01之间
x_test /= 255
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
# 特别注意label要使用one_hot encoding，x_train的shape(60000, 28,28）

#build model
model = Sequential()
model.add(LSTM(units=nb_lstm_outputs, input_shape=(nb_time_steps, nb_input_vector)))
model.add(Dense(10, activation='softmax'))

#compile:loss, optimizer, metrics
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

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
history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test), validation_freq=1, verbose=1)
model.summary()
# score = model.evaluate(x_test, y_test,batch_size=128, verbose=1)
# print(score)

###############################################    show   ###############################################

# 显示训练集和验证集的acc和loss曲线
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
