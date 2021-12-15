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

root = "E:/wakeup/"

all_patterns_proba = pd.read_csv(root + 'all_patterns_proba.csv', encoding='gbk')
probability = all_patterns_proba.loc[:, ['0','1','2','3']]

s1 = 100
s2 = 75
s3 = 50
s4 = 25
score_weight = np.array([s3, s4, s1, s2])
score = pd.DataFrame({'score': np.multiply(probability, score_weight).sum(1)})

all_patterns_scores = all_patterns_proba.join(score, rsuffix='_right')

print('score complete')

all_patterns_scores.to_csv(root + 'all_patterns_scores.csv',encoding='gbk')