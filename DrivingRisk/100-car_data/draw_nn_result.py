import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal
import numpy as np

datasetpath = 'E:/MachineLearning/DrivingRisk/100-car_data/training_score_data/'
csvname = 'run-validation_batch64unit120withID-tag-epoch_recall.csv'

import pandas as pd
import numpy as np
import os

def smooth(csv_path, weight=0.8):
    data = pd.read_csv(filepath_or_buffer=csv_path, header=0, names=['Step', 'Value'],
                       dtype={'Step': np.int64, 'Value': np.float64})
    scalar = data['Value'].values
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    save = pd.DataFrame({'Step': data['Step'].values, 'Value': smoothed})
    save.to_csv(datasetpath + 'smooth_' + csvname)


if __name__ == '__main__':
    smooth(datasetpath+ csvname)