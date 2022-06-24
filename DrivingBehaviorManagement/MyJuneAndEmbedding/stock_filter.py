from keras.datasets import mnist
from keras.layers import Dense, LSTM, Dropout
from keras.utils.np_utils import *
from keras.models import Sequential
import os
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
from sklearn.feature_extraction.text import CountVectorizer
import gensim


root = "D:/RX-105/wakeup/MyJuneAndEmbedding/"
#data preprocessing: tofloat32, normalization, one_hot encoding
#load data
data = pd.read_csv(root + 'all_patterns.csv', encoding='gbk')

all_labels = ['a', 'b', 'c', 'h', 'i', 'j', 'o', 'p', 'q']

pattern = data['pattern']


maxlen=0
count_30 = 0
for i in range(len(pattern)):
    if len(pattern[i]) >= maxlen:
        maxlen=len(pattern[i])
    if len(pattern[i]) >= 30:
        count_30+=1
print(maxlen)
print(count_30)