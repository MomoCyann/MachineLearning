from KNN.KNN import KNN
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()
data = iris.data[:,:2]
target = iris.target
X_train,X_test,y_train,y_test = train_test_split(data,target,test_size=0.2,random_state=1)

k_choices = range(1,31)
result = []
for k in k_choices:
    print('when k is', k)
    KNN_classfier = KNN(k)
    score = []
    scores = cross_val_score(KNN_classfier, X_train, y_train, cv=10,scoring='accuracy')
    # cv：选择每次测试折数  accuracy：评价指标是准确度,可以省略使用默认值。
    result.append(np.mean(scores))

plt.plot(k_choices,result)
plt.xlabel('K')
plt.ylabel('Accuracy')		#通过图像选择最好的参数
plt.show()
# best_knn = KNeighborsClassifier(n_neighbors=3)	# 选择最优的K=3传入模型
# best_knn.fit(train_X,train_y)			#训练模型
# print(best_knn.score(test_X,test_y))




