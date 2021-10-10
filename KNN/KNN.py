import numpy as np
from math import sqrt
from collections import Counter

class KNN:

    def __init__(self, k):
        self.k = k
        self._X_train = None
        self._y_train = None


    def fit(self, X_train, y_train):
        self._X_train = X_train
        self._y_train = y_train
        return self  # 模仿sklearn，调用fit函数会返回自身


    def predict(self, X_predict):
        # 预测X_predict矩阵每一行所属的类别
        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)  # 返回的结果也遵循sklearn


    def _predict(self, x):
        distances = [sqrt(np.sum((x_train - x) ** 2)) for x_train in self._X_train]
        nearest = np.argsort(distances)  # 对距离排序并返回对应的索引
        topK_y = [self._y_train[i] for i in nearest[:self.k]]  # 返回最近的k个距离对应的分类
        votes = Counter(topK_y)
        return votes.most_common(1)[0][0]

    def __repr__(self):
        return "KNN(k=%d)" % self.k


if __name__ == '__main__':
    X_train = np.array([[0, 0],
                        [1, 1],
                        [2, 2],
                        [10, 10],
                        [11, 11],
                        [12, 12]])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    x = np.array([[13,13],[-1,-1]])


    knn_clf = KNN(2)
    knn_clf.fit(X_train, y_train)
    print(knn_clf.predict(x))
