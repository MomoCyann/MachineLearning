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

class ScoreEngine:

    def __init__(self):
        self.root = "E:/wakeup/volatility_data/"
        self.s1 = 75
        self.s2 = 100
        self.s3 = 50
        self.s4 = 25
        self.score_weight = np.array([self.s2, self.s3, self.s4, self.s1])

    def cal_pat_score(self):
        all_patterns_proba = pd.read_csv(self.root + 'all_patterns_proba.csv', encoding='gbk')

        probability = all_patterns_proba.loc[:, ['0','1','2','3']]
        score = pd.DataFrame({'score': np.multiply(probability, self.score_weight).sum(1)})
        all_patterns_scores = all_patterns_proba.join(score, rsuffix='_right')
        print('score complete')

        all_patterns_scores.to_csv(self.root + 'all_patterns_scores.csv',encoding='gbk')

if __name__ == '__main__':

    se = ScoreEngine()
    se.cal_pat_score()
