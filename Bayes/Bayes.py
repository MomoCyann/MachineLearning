from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, load_iris
from collections import Counter
import numpy as np
from numpy import exp, pi, sqrt

import load_data
from KFOLD.KFOLD import KFOLD

class BAYES:
    def __init__(self):
        # 存储先验概率、训练集的均值、方差及label的类别数量。
        self.prob = None
        self.avgs = None
        self.vars = None
        self.n_class = None
        self.X_train = None
        self.y_train = None

    # 通过Python自带的Counter计算每个类别的占比，再将结果存储到numpy数组中。
    def cal_prob(self):
        prob = []
        count = Counter(self.y_train)
        for i in range(len(count)):
            prob.append(count[i] / len(self.y_train))
        return np.array(prob)

    # 每个label类别分别计算均值。
    def cal_avgs(self):
        mean = []
        for i in range(self.n_class):
            mean.append(self.X_train[self.y_train == i].mean(axis=0))
        return np.array(mean)

    # 每个label类别分别计算方差。
    def cal_vars(self):
        vars = []
        for i in range(self.n_class):
            vars.append(self.X_train[self.y_train == i].var(axis=0))
        return np.array(vars)

    # 通过高斯分布的概率密度函数计算出似然再连乘。
    def cal_likelihood(self, X_test):
        #np.prod()连乘
        return (1 / sqrt(2 * pi * self.vars) * exp(-(X_test - self.avgs) ** 2 / (2 * self.vars))).prod(axis=1)

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.prob = self.cal_prob()
        self.n_class = len(self.prob)
        self.avgs = self.cal_avgs()
        self.vars = self.cal_vars()

    # 用先验概率乘以似然度再归一化得到每个label的prob。
    def predict_prob(self, X_test):
        # np.apply_along_axis 对 一个数组释放一个函数
        likelihood = np.apply_along_axis(self.cal_likelihood, axis=1, arr=X_test)
        probs = self.prob * likelihood
        probs_sum = probs.sum(axis=1)
        return probs / probs_sum[:, None]

    def predict(self, X_test):
        return self.predict_prob(X_test).argmax(axis=1)

    def accuracy(self, X_test, y_test):
        y_pred = self.predict(X_test)
        count = 0
        for i in range(len(y_test)):
            if y_test[i] == y_pred[i]:
                count += 1
            else:
                continue
        return count / len(y_test)

def load_data_breastcancer():
    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    return X_train, X_test, y_train, y_test

def load_data_iris():
# load datasets
    iris = load_breast_cancer()
    data = iris.data[:,]
    target = iris.target
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=1)
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    X, y = load_data.breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    classfier = BAYES()
    classfier.fit(X_train, y_train)
    acc = classfier.accuracy(X_test, y_test)
    print("TEST:Accuracy is %.3f" % acc)

    # cross validation
    kf = KFOLD(X, y, 10)
    scores = []
    clf = BAYES()
    scores.append(kf.cross_validation(clf))