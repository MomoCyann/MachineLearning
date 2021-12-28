# -*- coding : utf-8 -*-
# coding: utf-8
import os
import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
import matplotlib
from math import sqrt,pow,acos
from pandas.core.frame import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
import joblib
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
import pyLDAvis.sklearn

root = "E:/wakeup/"
patterns_data = pd.read_csv(root + 'all_patterns.csv', encoding='gbk')

#构建词汇统计向量
text = patterns_data.loc[:,'pattern']
text_vectorizer = CountVectorizer(analyzer= 'char') # 设置单个字母
text_vec = text_vectorizer.fit_transform(text)

print(text_vectorizer.get_feature_names_out())
print(text_vec.toarray().sum(axis=0))

lda = LatentDirichletAllocation(n_components=4,
                                max_iter=30,
                                learning_method='batch')
# 学习方法batch 和online max_iter为em迭代次数
lda.fit(text_vec)


def print_top_words(model, feature_names, n_top_words):
    #打印每个主题下权重较高的term
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    #打印主题-词语分布矩阵
    print(model.components_)

n_top_words=20
text_feature_names = text_vectorizer.get_feature_names_out()
print_top_words(lda, text_feature_names, n_top_words)

probability = lda.transform(text_vec)
print('complete')

#可视化
data = pyLDAvis.sklearn.prepare(lda, text_vec, text_vectorizer)
pyLDAvis.display(data)
pyLDAvis.save_html(data, 'lda_4topics.html')

# 可能性
probability_data = pd.DataFrame(probability)
all_patterns_proba = patterns_data.join(probability_data, rsuffix='_right')
print('score complete')
all_patterns_proba.to_csv(root + 'all_patterns_proba.csv',encoding='gbk')

