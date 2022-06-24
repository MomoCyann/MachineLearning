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
#onehot encode for each pattern
pattern = data['pattern']
score = np.array(data['安全分数']).reshape(-1,1)

"""归一化"""
scaler = MinMaxScaler()
score = scaler.fit_transform(score)

all_labels = ['a', 'b', 'c', 'h', 'i', 'j', 'o', 'p', 'q']

text = data.loc[:,'pattern']
file = pd.Series.to_list(text)
text_washed = []
for i in range(len(file)):
    ls = [0]
    ls[0] = file[i][0]
    for j in range(1, len(file[i])):
        if file[i][j] != ls[-1]:
            ls.append(file[i][j])
    text_washed.append(ls)

model = gensim.models.Word2Vec(text_washed,
                               epochs=1000,
                               vector_size=100,
                               window=5,
                               sg=1,min_count=2)
print(model.wv.key_to_index)
print(model.wv['c'])
print(model.wv['j'])
print(model.wv.similarity('c', 'j'))
print(model.wv.similarity('a', 'b'))
print(model.wv.similarity('c', 'q'))
print(model.wv.similarity('b', 'q'))
print(model.wv.similarity('i', 'q'))
print(model.wv.similarity('j', 'q'))
'''10向量500次
0.72381383
0.41123614
0.8826378
0.686424
0.42074645
0.56611943'''
'''10向量 1000次
0.6362572
0.43329018
0.8510749
0.6948418
0.34426078
0.50986236'''
'''30
0.6711906
0.45342964
0.7966394
0.6759541
0.37249503
0.5146724'''
'''50向量
0.62252814
0.38317814
0.83826023
0.65269935
0.3424983
0.52623963'''
'''100向量
0.66274726
0.29745844
0.8219148
0.6826999
0.31320328
0.5208364'''
'''
0.6565429
0.4428146
0.8152961
0.67889446
0.39084345
0.5366955'''
print('complete')

c_similar=model.wv.most_similar("c",topn=8)#8个最相关的
print("与【c】最相关的词有：\n")
for word in c_similar:
    print(word[0],word[1])
print("*********\n")

j_similar=model.wv.most_similar("j",topn=8)#8个最相关的
print("与【j】最相关的词有：\n")
for word in j_similar:
    print(word[0],word[1])
print("*********\n")

q_similar=model.wv.most_similar("q",topn=8)#8个最相关的
print("与【q】最相关的词有：\n")
for word in q_similar:
    print(word[0],word[1])
print("*********\n")


input('input anything to save the embedding vector')
skip_gram = np.zeros((9,10))
all_labels = ['a', 'b', 'c', 'h', 'i', 'j', 'o', 'p', 'q']
for i in range(9):
    skip_gram[i] = model.wv[all_labels[i]]
#保存为npy类型
np.save('embedding.npy',skip_gram)
# #npy的读取
# your_np_arr = np.load('文件的名字.npy')