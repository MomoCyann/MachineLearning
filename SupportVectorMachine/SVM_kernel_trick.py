import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt


class SVM:

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.m = self.X.shape[0]
        self.a = np.zeros(self.m)
        self.b = 0
        self.w = np.zeros(len(X[0]))

        self.g = 100 # 迭代次数
        self.C = 10 # 惩罚系数
        self.ep = 1e-3 # 精准度
        self.sigma = 10 # sigma越小，图像越陡峭，超平面越细致， sigma越大 超平面越平滑
        # sigma = 1 准确率78
        # sigma = 2 准确率80
        # sigma = 5 准确率82
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
           res += self.a[i] * self.Y[i] * self.kernel_rbf(self.X[i], xi)
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

    def kernel_rbf(self, x1, x2):
        '''
        计算内积
        :param x1:
        :param x2:
        :return:
        '''
        return np.exp(-1 * np.dot((x1 - x2).T, x1 - x2) / (2 * np.square(self.sigma)))

    def smo(self):
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
                    k11 = self.kernel_rbf(self.X[i], self.X[i])
                    k12 = self.kernel_rbf(self.X[i], self.X[j])
                    k22 = self.kernel_rbf(self.X[j], self.X[j])
                    k21 = self.kernel_rbf(self.X[j], self.X[i])
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
        x0 = np.linspace(-2, 2, 200)
        decision_boundary = - self.w[0] / self.w[1] * x0 - self.b / self.w[1]
        plt.plot(x0, decision_boundary, "k", linewidth=2)

def load_data():
    # 月亮数据集
    X, Y = make_moons(n_samples = 100, noise = 0.15, random_state=0)
    Y[Y == 0] = -1
    return X, Y

def draw_data(X, Y):
    plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y), cmap=plt.cm.Spectral)
    plt.show()

def main():
    X, Y = load_data()
    model = SVM(X, Y)
    model.smo()

    y_pred = model.predict(X)

    correct = np.array([y_pred == Y])
    correct = correct.sum() / correct.shape[0]
    print("the percent of correct: "+str(correct))

    model.draw_dec_bud()
    draw_data(X, Y)


if __name__ == "__main__":
    main()


