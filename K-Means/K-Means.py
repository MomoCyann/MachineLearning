import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

class KMeans:

    def __init__(self, X):
        self.X = X
        self.k = 3

        self.g = 300 # 迭代次数
        self.tol =  0.0001 # 误差范围
        self.centers = {}
        self.types ={}

    def train(self):
        # generate the centers
        for i in range(self.k):
            self.centers[i] = self.X[i]

        for i in range(self.g):
            self.types = {}
            for j in range(self.k):
                self.types[j] = []
            for item in self.X:
                dis = []
                for center in self.centers:
                    dis.append(np.linalg.norm(item - self.centers[center]))
                type_now = dis.index(min(dis))
                self.types[type_now].append(item)

            # move center
            '''
            这个dict如果不加结果会出错
            '''
            centers_old = dict(self.centers)
            for type in self.types:
                self.centers[type] = np.average(self.types[type], axis=0)

            # check end
            thebest = True
            for center in self.centers:
                c1 = centers_old[center]
                c2 = self.centers[center]
                if np.sum((c2 - c1) / c1 * 100.0) > self.tol:
                    thebest = False
            if thebest:
                break

    def predict(self, x):
        for center in self.centers:
            dis = [np.linalg.norm(x - self.centers[center])]
        type = dis.index(min(dis))
        return type

    def draw(self):
        for center in self.centers:
            plt.scatter(self.centers[center][0], self.centers[center][1], marker='*', s=150)

        for type in self.types:
            for point in self.types[type]:
                if type == 0:
                    plt.scatter(point[0], point[1], c='r')
                elif type == 1:
                    plt.scatter(point[0], point[1], c='g')
                else:
                    plt.scatter(point[0], point[1], c='b')
        plt.show()


def load_data():
    # 鸢尾花
    iris = datasets.load_iris()
    x = iris['data']
    y = iris['target']
    x = x[:, :2]
    return x, y


if __name__ == '__main__':
    X, Y = load_data()
    model = KMeans(X)
    model.train()
    model.draw()



