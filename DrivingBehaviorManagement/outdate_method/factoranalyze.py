import os
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager as fm, rcParams
from sklearn.preprocessing import StandardScaler


class EVENT:

    def __init__(self):
        self.folder_name = ["031267", "077102", "078351", "078837", "080913", "082529",
                            "090798", "098840", "108140", "112839"]
        self.filename_extenstion = '.csv'
        self.datasetpath = "E:/wakeup/dataset/"
        self.datapath = 'E:/wakeup/data/'
        self.eventpath = "E:/wakeup/cluster/"
        self.rootpath = "E:/wakeup/"
        self.day = 20200901


    def factor_analyze(self):
        general_data_merged = pd.read_csv(self.rootpath + 'general_data_merged_final.csv', encoding='gbk')
        factors = general_data_merged[['速度极差', '速度标准差', '速度均值', '最大加速度',
                                       '最大刹车加速度', '加速度标准差', '加速度均值','加速度极差',
                                       'over40','over50','over60','over70','over80','over90','over100',
                                       '急加速时长(s)', '急减速时长(s)', '疲劳驾驶时长(s)',
                                       '空档滑行时长(s)', '长刹车时长(s)', '长离合时长(s)', '大踩油门时长(s)', '停车踩油门时长(s)',
                                       '立即起步时长(s)', '立即停车时长(s)', '过长怠速时长(s)']]
        factors_fixed = factors.dropna(axis=0, how='all')

        # # 标准化
        # std = StandardScaler()
        # cache = std.fit_transform(factors_fixed)
        # factors_fixed = pd.DataFrame(cache)

        # Bartlett's test of sphericity
        chi_square_value, p_value = calculate_bartlett_sphericity(factors_fixed)
        print('p-value = '+str(p_value))
        print(chi_square_value)
        # Kaiser-Meyer-Olkin (KMO) Test
        kmo_all, kmo_model = calculate_kmo(factors_fixed)
        print('kmo test value = ' +str(kmo_model))

        # Create factor analysis object and perform factor analysis
        fa = FactorAnalyzer(26, rotation=None)
        fa.fit(factors_fixed)

        # 方差贡献率
        fa_sd = fa.get_factor_variance()
        fa_sd_df = pd.DataFrame(
            {'特征值': fa_sd[0], '方差贡献率': fa_sd[1], '方差累计贡献率': fa_sd[2]})
        print(fa_sd_df)


        # Check Eigenvalues
        ev, v = fa.get_eigenvalues()

        # Create scree plot using matplotlib
        plt.scatter(range(1, factors_fixed.shape[1] + 1), ev)
        plt.plot(range(1, factors_fixed.shape[1] + 1), ev)
        plt.title('Scree Plot')
        plt.xlabel('Factors')
        plt.ylabel('Eigenvalue')
        plt.grid()
        plt.show()

        fa = FactorAnalyzer(5, rotation="varimax")
        fa.fit(factors_fixed)
        """
        FactorAnalyzer(bounds=(0.005, 1), impute='median', is_corr_matrix=False,
                method='minres', n_factors=5, rotation='varimax',
                rotation_kwargs={}, use_smc=True)
        """
        df_cm = pd.DataFrame(np.abs(fa.loadings_), index=factors_fixed.columns)
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置
        plt.figure(figsize=(14, 14))
        ax = sns.heatmap(df_cm, annot=True, cmap="BuPu")
        # 设置y轴的字体的大小
        ax.yaxis.set_tick_params(labelsize=15)
        plt.title('Factor Analysis', fontsize='xx-large')
        # Set y-axis label
        plt.ylabel('Sepal Width', fontsize='xx-large')
        plt.savefig('factorAnalysis.png', dpi=500)
        plt.show()

        transformer_df = pd.DataFrame(fa.transform(factors_fixed))
        transformer_df.to_csv(self.rootpath + 'data_factors' + self.filename_extenstion, index=False, encoding='gbk')

if __name__ == '__main__':
    eventdetection = EVENT()
    # eventdetection.eventprocess()
    # print('提取完毕')
    # print('提取完毕!!!')
    eventdetection.factor_analyze()