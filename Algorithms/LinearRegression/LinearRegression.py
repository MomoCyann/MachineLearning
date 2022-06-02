import random
import numpy as np
import matplotlib.pyplot as plt

# TODO 添加正则化

# config
# origin: y = 3x0 + 2x1

m = 200 # 样本数
n = 2 # 特征向量个数
g = 50 # 迭代次数
x = np.zeros([m,n])
y = np.zeros([m])
hx = np.zeros([m])
theta = np.zeros([n])
alpha = 0.003 # 学习速率/梯度下降幅度
loss_min = 99999
loss_save = [] # 方便画图

def generate(x,y,hx):
    for i in range(m):
        x[i,0] = 1 # x0 = 1
        tmp = random.uniform(0,20)
        x[i,1] = tmp
        y[i] = 2*tmp + 3 + 5*np.random.normal() # 正态分布
        for j in range(n):
            hx[i] += theta[j] * x[i,j]
    return x,y,hx


def cal_loss():
    loss = (1/m) * np.sum(np.square(hx - y))
    return loss


def gradient_desent(theta):
    desent_theta = np.zeros([n])
    for i in range(m):
        for j in range(n):
            desent_theta[j] += x[i,j] * (hx[i]-y[i]) / m
    theta = theta - desent_theta * alpha
    return theta


def normal_equation(theta):
    theta = np.linalg.inv(x.T@x) @x.T @y
    return theta


def plot_cost():
    gg = []
    for i in range(g+1):
        gg.append(i)
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.plot(gg, loss_save, 'm', linewidth = "5")
    plt.show()


def plot_point():
    xx = []
    for i in range(m):
        xx.append(x[i, 1])
    plt.scatter(xx, y)  # 散点
    x2 = np.arange(0, 20, 0.1)
    y2 = theta[1] * x2 + theta[0]
    plt.plot(x2, y2, color='r')
    plt.show()

if __name__ == "__main__":
    generate(x,y,hx)
    ## 梯度下降
    # for g in range(g):
    #     loss = cal_loss()
    #     loss_save.append(loss)
    #     if loss_min > loss:
    #         loss_min = loss
    #         theta = gradient_desent(theta)
    #         for i in range(m): # 重新计算预测值
    #             hx[i] = 0
    #             for j in range(n):
    #                 hx[i] += theta[0, j] * x[i, j]
    #print("final loss: " + str(loss))
    #print("result is y = " + str(theta[0, 1]) + "x + " + str(theta[0, 0]))
    #plot_cost()

    # 正规方程
    theta = normal_equation(theta)
    print(theta)

    # 可视化
    plot_point()

