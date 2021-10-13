from sklearn.datasets import load_iris,load_breast_cancer


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