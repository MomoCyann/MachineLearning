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
        X_folds = []
        y_folds = []
        X_folds = np.vsplit(self.X, self.folds)
        y_folds = np.hsplit(self.y, self.folds)
        acc = []
        # split the train sets and validation sets
        for i in range(self.folds):
            X_train = np.vstack(X_folds[:i] + X_folds[i + 1:])
            X_val = X_folds[i]
            y_train = np.hstack(y_folds[:i] + y_folds[i + 1:])
            y_val = y_folds[i]

            y_val_pred = clf.predict(X_val)
            accuracy = np.mean(y_val_pred == y_val)
            #accuracy_of_k[k].append(accuracy)
            acc.append(accuracy)
        # for k in sorted(k_choices):
        #     for accuracy in accuracy_of_k[k]:
        #         print('k = %d,accuracy = %f' % (k, accuracy))
        result=np.mean(acc)
        print('the mean of accuracy is ',result)
        return result


#we chose the best one
# best_k = 25
# classify = KNN(best_k)
# classify.fit(X_train,y_train)
# y_test_pred = classify.predict(X_test)
# num_correct = np.sum(y_test==y_test_pred)
# accuracy_test = np.mean(y_test==y_test_pred)
# print('test accuracy is %d/%d = %f' %(num_correct,X_test.shape[0],accuracy_test))


