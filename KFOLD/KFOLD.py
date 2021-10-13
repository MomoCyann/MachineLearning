import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

class KFOLD:

    def __init__(self, X, y, folds):
        self.X = X
        self.y = y
        self.folds = folds


    def cross_validation(self,clf):
        X_folds = np.array_split(self.X, self.folds, axis=0)
        y_folds = np.array_split(self.y, self.folds, axis=0)
        self.acc = []
        # split the train sets and validation sets
        for i in range(self.folds):
            X_train = np.vstack(X_folds[:i] + X_folds[i + 1:])
            X_val = X_folds[i]
            y_train = np.hstack(y_folds[:i] + y_folds[i + 1:])
            y_val = y_folds[i]
            clf.fit(X_train, y_train)
            y_val_pred = clf.predict(X_val)
            accuracy = np.mean(y_val_pred == y_val)
            self.acc.append(accuracy)
        result=np.mean(self.acc)
        print('KFOLD: the mean of accuracy is ',result)
        return result

    def get_scores(self):
        return self.acc

    def get_plot(self):
        # kfold graph
        scores = self.get_scores()
        plt.plot(range(self.folds), scores)
        plt.xlabel('folds')
        plt.ylabel('Accuracy')  # 通过图像选择最好的参数
        plt.show()

