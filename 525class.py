import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import clone
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

root = "E:/Personal Files/2022/data_mining/"
datasets = ['bal', 'gla', 'hay', 'iri', 'new', 'win', 'zoo']
file_type = '.xls'
# 每一折的指标存储
f1_save = []
accuracy_save = []


def load_data(dataname):
    raw_data = pd.read_excel(root + dataname + file_type, header=None)
    label = raw_data.iloc[:, -1:]
    data = raw_data.iloc[:, :-1]
    return data, label


def minmax(trainset):
    """归一化"""
    scaler = MinMaxScaler()
    trainset_new = scaler.fit_transform(trainset)
    return trainset_new


def onehot(data):
    data = np.array(data)
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(data)
    return pd.DataFrame(onehot_encoded)


def cross_validation(dataset, label, model):
    """交叉验证"""
    print(type(dataset))
    if str(type(dataset)) == "<class 'numpy.ndarray'>":
        X = dataset
    if str(type(dataset)) == "<class 'pandas.core.frame.DataFrame'>":
        X = dataset.values
    y = label.values
    # KNN时做归一化
    if type(model).__name__ == "KNeighborsClassifier":
        dataset = minmax(dataset)
    # 每一折的指标存储
    f1_save.clear()
    accuracy_save.clear()
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    # 10次10折交叉验证
    for i in range(10):
        for train_index, test_index in skf.split(X, y):
            # 复制一个纯净模型
            clf = clone(model)
            # K折
            X_train_folds = X[train_index]
            y_train_folds = y[train_index]
            X_test_fold = X[test_index]
            y_test_fold = y[test_index]
            # 训练模型
            clf.fit(X_train_folds, y_train_folds)
            # 预测测试集
            test_labels_pred = clf.predict(X_test_fold)
            # 计算指标
            f1_score_fold = f1_score(y_test_fold, test_labels_pred, average="weighted")
            acc_fold = accuracy_score(y_test_fold, test_labels_pred)
            f1_save.append(f1_score_fold)
            accuracy_save.append(acc_fold)
            print(classification_report(y_test_fold, test_labels_pred))
    print('平均准确率为：')
    print(np.array(f1_save).mean())
    print('平均f1 score为：')
    print(np.array(accuracy_save).mean())


def get_accuracy():
    return np.array(f1_save).mean()


def get_f1():
    return np.array(accuracy_save).mean()


def main():
    KNN = KNeighborsClassifier()
    DT = DecisionTreeClassifier()
    NB = GaussianNB()
    RF = RandomForestClassifier()
    models = [KNN, DT, NB, RF]
    models_name = ['KNN', 'DT', 'NB', 'RF']
    # result为最终结果，保存不同数据集不同模型最好结果
    result = pd.DataFrame(columns=['KNN', 'DT', 'NB', 'RF'])
    # result_model保存不同数据集不同参数的结果
    for model in models:
        result_model = pd.DataFrame(columns=['bal', 'gla', 'hay', 'iri', 'new', 'win', 'zoo'])
        # 为每个模型分配不同调参策略
        print(type(model).__name__)
        if type(model).__name__ == "KNeighborsClassifier":
            param_grid = [{'n_neighbors': [3, 5, 7, 9, 11, 13, 15]}]
        if type(model).__name__ == "DecisionTreeClassifier":
            param_grid = [{'max_depth': [3, 4, 5, 6, 7, 8, 9, 10]}]
        if type(model).__name__ == "GaussianNB":
            param_grid = [{'priors': [None]}]
        if type(model).__name__ == "RandomForestClassifier":
            param_grid = [{'n_estimators': [20, 40, 60, 80, 100, 150, 200]}]
        for name in datasets:
            # 加载数据集
            dataset, label = load_data(name)
            dataset, label = shuffle(dataset, label)
            if name == 'zoo':
                dataset = onehot(dataset)
            # KNN时做归一化
            if type(model).__name__ == "KNeighborsClassifier":
                dataset = minmax(dataset)
            grid_search = GridSearchCV(model, param_grid, cv=10, scoring='accuracy')
            grid_search.fit(dataset, label)
            # 获取每个参数对应结果
            means = grid_search.cv_results_['mean_test_score']
            params = grid_search.cv_results_['params']
            # 填入result_model.csv
            for mean, param in zip(means, params):
                result_model.loc[str(param), datasets[datasets.index(name)]] = mean
            best_model = grid_search.best_estimator_
            cross_validation(dataset, label, best_model)
            # 获取所有参数组合中最优结果做最终结果
            best_accuracy = get_accuracy()
            # 填入准确率
            result.loc[name, models_name[models.index(model)]] = best_accuracy
        result_model.to_csv(root + type(model).__name__ + "_result.csv")
    # 保存结果
    result.to_csv(root + "result525.csv")


if __name__ == "__main__":
    main()

