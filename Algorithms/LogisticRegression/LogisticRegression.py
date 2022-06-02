from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from KFOLD.KFOLD import KFOLD
import load_data

#!!!!!!!!!!!!!!!!!!!!!!!!!!!
#很重要的一点：！！！！
#fit的一些参数不设置清零的话，比如最小的loss，在交叉验证重复训练，后面就炸掉了


class LogisticRegression:

    def __init__(self):
        self.alpha = 0.01
        self.gen = 100
        self.hx = []
        self.y = []
        self.loss_save = []
        self.theta = None
        self.X_train = None
        self.y_train = None
        self.hx = []
        self.y = []


    def sigmoid(self, x):
        # return 1 / (1 + np.exp(-x)) 溢出
        return .5 * (1 + np.tanh(.5 * x))

    def fit(self, X_train, y_train):
        # 初始化数据集和参数和假设
        self.theta = np.zeros([len(X_train[0])])
        self.X_train = X_train
        self.y_train = y_train
        #关键步骤 ，初始化最小loss
        self.loss_min = 99999
        self.loss_save.clear()
        self.cal_hx_y()
        # 迭代
        for g in range(self.gen):
            loss = self.cal_loss()
            self.loss_save.append(loss)
            if self.loss_min > loss:
                self.loss_min = loss
                self.gradiant_descent()
                self.cal_hx_y()


    def cal_hx_y(self):
        self.hx.clear()
        self.y.clear()
        for i in range(len(self.y_train)):
            self.hx.append(self.sigmoid(np.dot(self.theta,self.X_train[i])))
            if self.hx[i] >= 0.5:
                self.y.append(1)
            else:
                self.y.append(0)


    def cal_loss(self):
        sum = 0
        for i in range(len(self.hx)):
            sum += self.y_train[i] * np.log(self.hx[i] + 1e-5) + (1-self.y_train[i]) * np.log(1-self.hx[i] + 1e-5)
        return sum * (-1) / len(self.hx)


    def gradiant_descent(self):
        desent_theta = np.zeros([len(self.theta)])
        for i in range(len(self.hx)):
            for j in range(len(self.theta)):
                desent_theta[j] += self.X_train[i,j] * (self.hx[i]-self.y_train[i]) / len(self.hx)
        self.theta = self.theta - desent_theta * self.alpha



    def predict(self, X_predict):
        self.y_pred = []
        for i in range(len(X_predict)):
            if self.sigmoid(np.dot(self.theta,X_predict[i])) >= 0.5:
                self.y_pred.append(1)
            else:
                self.y_pred.append(0)
        return self.y_pred


    def accuracy(self, X_test, y_test):
        self.y_pred = self.predict(X_test)
        count = 0
        for i in range(len(y_test)):
            if y_test[i] == self.y_pred[i]:
                count+=1
            else:
                continue
        return count/len(y_test)

# 鸢尾花
def load_ddata():
    iris = datasets.load_iris()
    x = iris['data']
    y = iris['target']
    x = x[y!=2]
    y = y[y!=2]
    # 选择哪些数据
    x = x[:,]
    # 给x第一列加一列1，常数项
    x_one = np.ones([len(x)])
    x = np.insert(x,0,values=x_one,axis=1)
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
    return x_train,x_test,y_train,y_test


if __name__ == "__main__":
    # 初始化数据集和参数和假设
    X, y = load_data.make_moons()
    # X = X[y != 2]
    # y = y[y != 2]
    # # 选择哪些数据
    X = X[:,]
    # # 给x第一列加一列1，常数项
    X_one = np.ones([len(X)])
    X = np.insert(X, 0, values=X_one, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    classfier = LogisticRegression()
    classfier.fit(X_train, y_train)
    accuracy = classfier.accuracy(X_test,y_test)
    print('test accuracy is ',accuracy)

    # cross validation
    kf = KFOLD(X_train, y_train, 10)
    score = []
    clf = LogisticRegression()
    score.append(kf.cross_validation(clf))
    print(score)
    # get_plot
    kf.get_plot()