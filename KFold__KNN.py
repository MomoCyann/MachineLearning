from KNN.KNN import KNN
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


#load datasets
iris = load_iris()
data = iris.data[:,:2]
target = iris.target

#cross validation
X,X_test,y,y_test = train_test_split(data,target,test_size=0.2,random_state=1)
folds = 10
k_choices = [1,3,5,7,9,13,15,20,25]

X_folds = []
y_folds = []

X_folds = np.vsplit(X, folds)
y_folds = np.hsplit(y, folds)

accuracy_of_k = {}
for k in k_choices:
    accuracy_of_k[k] = []
# split the train sets and validation sets
for i in range(folds):
    X_train = np.vstack(X_folds[:i] + X_folds[i + 1:])
    X_val = X_folds[i]
    y_train = np.hstack(y_folds[:i] + y_folds[i + 1:])
    y_val = y_folds[i]

    for k in k_choices:
        clf = KNN(k)
        clf.fit(X_train, y_train)
        y_val_pred = clf.predict(X_val)
        accuracy = np.mean(y_val_pred == y_val)
        accuracy_of_k[k].append(accuracy)

for k in sorted(k_choices):
    for accuracy in accuracy_of_k[k]:
        print('k = %d,accuracy = %f' % (k, accuracy))
    print(np.mean(accuracy))


#we chose the best one
best_k = 25
classify = KNN(best_k)
classify.fit(X_train,y_train)
y_test_pred = classify.predict(X_test)
num_correct = np.sum(y_test==y_test_pred)
accuracy_test = np.mean(y_test==y_test_pred)
print('test accuracy is %d/%d = %f' %(num_correct,X_test.shape[0],accuracy_test))


