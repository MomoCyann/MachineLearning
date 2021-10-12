import numpy as np
from math import sqrt
from collections import Counter
from KFOLD.KFOLD import KFOLD
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

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
    # load datasets
    iris = load_iris()
    data = iris.data[:, :2]
    target = iris.target
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=1)

    #cross validation
    k_choices = range(1,31)
    kf = KFOLD(X_train, y_train, 10)
    scores = []
    for k in k_choices:
        clf = KNN(k)
        clf.fit(X_train, y_train)
        print('when k is ',k)
        scores.append(kf.cross_validation(clf))
    print(scores)

    plt.plot(k_choices, scores)
    plt.xlabel('K')
    plt.ylabel('Accuracy')		#通过图像选择最好的参数
    plt.show()
