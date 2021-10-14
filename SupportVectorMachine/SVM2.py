import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from KFOLD.KFOLD import KFOLD
from sklearn.model_selection import train_test_split
import load_data

class SVM:

    def __init__(self):
        self.X = None
        self.Y = None
        self.m = None
        self.a = None
        self.b = 0
        self.w = None

        self.g = 100 # 迭代次数
        self.C = 10 # 惩罚系数
        self.ep = 1e-3 # 精准度
        self.E = None # 预测值与真实值之差

    def choose_a(self, i, m):
        '''
        随机一个数作为
        :param i:
        :param m:
        :return:
        '''
        j = np.random.randint(0,m)
        while j == i:
            j = np.random.randint(0,m)
        return j

    def fx(self, xi):
        '''
        预测值
        :param xi:
        :return:
        '''
        res = 0
        for i in range(self.m):
           res += self.a[i] * self.Y[i] * self.kernel(self.X[i], xi)
        res += self.b
        return res

    def predict(self, xi):
        '''
        预测分类
        :param xi:
        :return:
        '''
        result = self.fx(xi)
        return np.sign(result)

    def kernel(self, x1, x2):
        '''
        计算内积
        :param x1:
        :param x2:
        :return:
        '''
        return np.dot(x1, x2.T)

    def accuracy(self, X_test, y_test):
        self.y_pred = self.predict(X_test)
        count = 0
        for i in range(len(y_test)):
            if y_test[i] == self.y_pred[i]:
                count+=1
            else:
                continue
        return count/len(y_test)

    def fit(self, X_train, y_train):
        #SMO算法
        self.X = X_train
        self.Y = y_train
        self.m = self.X.shape[0]
        self.a = np.zeros(self.m)
        self.w = np.zeros(len(self.X[0]))
        self.E = np.zeros(self.m)
        g_now = 0
        while g_now < self.g:
            g_now += 1
            for i in range(self.m):
                a1 = self.a[i]
                self.E[i] = self.fx(self.X[i]) - self.Y[i]
                y1 = self.Y[i]

                if (self.E[i] * y1 > self.ep and a1 > self.ep) or (self.E[i] * y1 < 0 and a1 < self.C):
                    # 违反KKT
                    # a2即是E1-E2差距最大的那一个
                    step = self.E[i] - self.E
                    j = np.argmax(step)
                    a2 = self.a[j]
                    self.E[j] = self.fx(self.X[j]) - self.Y[j]

                    #计算上下界
                    y1 = self.Y[i]
                    y2 = self.Y[j]
                    if y1 != y2:
                        l = max(0, a2-a1)
                        h = min(self.C, self.C + a2 - a1)
                    else:
                        l = max(0,a2 + a1 - self.C)
                        h = min(self.C, a2 + a1)
                    k11 = self.kernel(self.X[i], self.X[i])
                    k12 = self.kernel(self.X[i], self.X[j])
                    k22 = self.kernel(self.X[j], self.X[j])
                    k21 = self.kernel(self.X[j], self.X[i])
                    eta = k11 + k22 - 2*k12

                    #更新a
                    if eta == 0:
                        eta += 1e-6

                    a2_noclip = a2 + y2 * (self.E[i] - self.E[j]) / eta
                    a2_new = np.clip(a2_noclip, l, h)

                    a1_new = a1 + y1 * y2 * (a2 - a2_new)

                    self.a[i] = a1_new
                    self.a[j] = a2_new

                    #更新b
                    b1 = -self.E[i] - y1 * k11 * (a1_new - a1) - y2 * k21 * (a2_new - a2) + self.b
                    b2 = -self.E[j] - y2 * k12 * (a1_new - a1) - y2 * k22 * (a2_new - a2) + self.b
                    self.b = (b1 + b2) / 2

    def draw_dec_bud(self):
        for i in range(self.m):
            self.w += self.a[i] * self.Y[i] * self.X[i]
        x0 = np.linspace(1, 3, 200)
        decision_boundary = - self.w[0] / self.w[1] * x0 - self.b / self.w[1]
        plt.plot(x0, decision_boundary, "k", linewidth=2)


def draw_data(X, Y):
    plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y), cmap=plt.cm.Spectral)
    plt.show()

if __name__ == "__main__":
    # X, y = load_data.iris()
    # X = X[y != 2]
    # y = y[y != 2]
    # y[y == 0] = -1

    # X, y = load_data.make_moons()
    # y[y == 0] = -1

    X, y = load_data.make_moons()
    y[y == 0] = -1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    classfier = SVM()
    classfier.fit(X_train, y_train)
    accuracy = classfier.accuracy(X_test, y_test)
    print('test accuracy is ',accuracy)

    # draw
    classfier.draw_dec_bud()
    draw_data(X_train, y_train)

    # cross validation
    kf = KFOLD(X_train, y_train, 10)
    clf = SVM()
    score =  kf.cross_validation(clf)
    print(score)



