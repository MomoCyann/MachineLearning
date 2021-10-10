import numpy as np
from math import sqrt
from collections import Counter

class KNN(k, X_train, y_train):

    def __int__(self):
        self.k = k
        self.X_train = X_train
        self.y_train = y_train


    def predict(self, x):
        distance = [np.sqrt(np.sum((self.X_train-x)**2)) for x_train in self.X_train]
        nearest = np.argsort(distance)

        #最近的点的类别为
        types = [y_train[i] for i in nearest]
        types = types[0:self.k]
        votes = Counter(types)

        return votes.most_common(1)[0][0]


if __name__ == '__main__':
    x = [[0, 0],
         [1, 1],
         [2, 2],
         [10, 10],
         [11, 11],
         [12, 12]]
    y = [0, 0, 0, 1, 1, 1]
    model = KNN(3, x, y)
    model.predict()
