import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from KFOLD.KFOLD import KFOLD
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

class KMeans:

    def __init__(self, k):
        self.X = None
        self.k = k

        self.g = 300 # 迭代次数
        self.tol =  0.0001 # 误差范围
        self.centers = {}
        self.types ={}

    def fit(self, X):
        self.X = X
        # generate the centers
        for i in range(self.k):
            self.centers[i] = self.X[i]

        for i in range(self.g):
            self.types = {}
            #每个中心  对应 储存多个点
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

    def predict(self, X_test):
        y_pred = []
        for item in X_test:
            dis = []
            for center in self.centers:
                dis.append([np.linalg.norm(item - self.centers[center])])
            type = dis.index(min(dis))
            y_pred.append(type)
        return y_pred

#acc有错，聚类还没想好怎么把标签对应
    def accuracy(self, X_test, y_test):
        self.y_pred = self.predict(X_test)
        count = 0
        for i in range(len(y_test)):
            if y_test[i] == self.y_pred[i]:
                count += 1
            else:
                continue
        return count / len(y_test)

    def get_plot(self):
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
# load datasets
    iris = load_iris()
    data = iris.data
    target = iris.target
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=1)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()
    clf = KMeans(3)
    clf.fit(X_train)
    clf.get_plot()
    # print(clf.accuracy(X_test, y_test))



