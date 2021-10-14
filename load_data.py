from sklearn.datasets import load_iris,load_breast_cancer
import pandas as pd
import numpy as np
from sklearn import datasets
PATH = 'D:/wakeup/ml_datasets/'

def iris():
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X,y

def breast_cancer():
    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target
    return X,y

def make_moons():
    X, y = datasets.make_moons(n_samples=100, noise=0.15, random_state=0)
    #Y[Y == 0] = -1
    return X,y

def abalone():
    data = pd.read_csv(PATH + 'abalone.csv')
    data.columns = ["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight", "Rings"]
    y = data['Rings']
    X = data.drop(columns=['Rings'])
    X.loc[X.Sex == 'M', 'Sex'] = 0
    X.loc[X.Sex == 'F', 'Sex'] = 1
    X.loc[X.Sex == 'I', 'Sex'] = 2
    X = np.array(X)
    y = np.array(y)
    return X,y

def yeast():
    data = pd.read_csv(PATH + 'yeast.csv')
    print(data.head())
