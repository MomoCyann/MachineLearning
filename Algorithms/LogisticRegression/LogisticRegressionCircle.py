from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#TODO 添加训练效果的可视化

alpha = 0.9
g = 100
hx = []
y = []
loss_min = 99999
loss_save = []


# 圆数据集
def load_circle_data():
    dataframe = pd.read_csv("./data2.txt")
    dataframe = dataframe.values
    x = dataframe[:,:2]
    y = dataframe[:,2:]
    # # 给x第一列加一列1，常数项
    x_one = np.ones([len(x)])
    x = np.insert(x,0,values=x_one,axis=1)
    plt.scatter(x[:57, 1], x[:57, 2],color='red')
    plt.scatter(x[57:, 1], x[57:, 2], color='blue')
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)
    return x_train,x_test,y_train,y_test


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 映射多项式
def create_fx():
    fx = np.zeros([len(x_train),len(x_train[0])])

    fx = x_train * x_train
    # fx = np.insert(fx, 1, values=x_train[:,2], axis=1)
    # fx = np.insert(fx, 1, values=x_train[:,1], axis=1)
    return fx


def cal_hx_y_more():
    hx.clear()
    y.clear()
    for i in range(len(y_train)):
        hx.append(sigmoid(np.dot(theta,fx[i])))
        if hx[i] >= 0.5:
            y.append(1)
        else:
            y.append(0)
    return hx,y


def cal_loss():
    sum = 0
    for i in range(len(hx)):
        sum += y_train[i] * np.log(hx[i]) + (1-y_train[i]) * np.log(1-hx[i])
    return sum * (-1) / len(hx)


def gradiant_descent(theta):
    desent_theta = np.zeros([len(theta)])
    for i in range(len(hx)):
        for j in range(len(theta)):
            desent_theta[j] += fx[i,j] * (hx[i]-y_train[i]) / len(hx)
    theta = theta - desent_theta * alpha
    return theta


def predict():
    y_pred = np.empty([len(y_test)])
    for i in range(len(y_test)):
        if sigmoid(np.dot(theta, fx[i])) >= 0.5:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    return y_pred


def cal_accuracy():
    count = 0
    for i in range(len(y_test)):
        if y_test[i] == y_pred[i]:
            count+=1
        else:
            continue
    return count/len(y_test)


if __name__ == "__main__":
    # 初始化数据集和参数和假设
    x_train,x_test,y_train,y_test = load_circle_data()
    fx = create_fx()
    theta = np.zeros([len(fx[0])])
    hx,y = cal_hx_y_more()
    # 迭代
    for g in range(g):
        loss = cal_loss()
        loss_save.append(loss)
        if loss_min > loss:
            loss_min = loss
            theta = gradiant_descent(theta)
            hx,y = cal_hx_y_more()
    print(loss_min)
    print(theta)
    y_pred = predict()
    print(y_test)
    print(y_pred)
    accuracy = cal_accuracy()
    print(accuracy)
    plt.show()