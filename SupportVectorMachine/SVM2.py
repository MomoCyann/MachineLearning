import numpy as np
from sklearn import datasets


class SVM:

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.m = self.X.shape[0]
        self.a = np.zeros(self.m)
        self.b = 0
        self.w = 0

        self.g = 100 # 迭代次数
        self.C = 10 # 惩罚系数
        self.ep = 1e-3 # 精准度
        self.E = np.zeros(self.m) # 预测值与真实值之差

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
        res = self.fx(xi)
        return np.sign(res)

    def kernel(self, x1, x2):
        '''
        计算内积
        :param x1:
        :param x2:
        :return:
        '''
        return np.dot(x1, x2.T)

    def smo(self):
        g_now = 0
        while g_now < self.g:
            g_now += 1
            for i in range(0, self.m):
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

    # def import_w(self,X):
    #     self.w =

def load_data():
    # 鸢尾花
    iris = datasets.load_iris()
    x = iris['data']
    y = iris['target']
    x = x[y != 2]
    x = x[:, 2:]
    y = y[y != 2]
    y[y == 0] = -1
    return x,y

def main():
    X, Y = load_data()
    model = SVM(X, Y)
    model.smo()

    y_pred = model.predict(X)
    correct = (y_pred == Y).astype('float')
    correct = correct.sum() / correct.shape[0]
    print(correct)


if __name__ == "__main__":
    main()
    print("complete")


