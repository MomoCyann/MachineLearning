import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal
import numpy as np

datasetpath = 'E:/MachineLearning/DrivingRisk/100-car_data/training_score_data/'
data_train_set = pd.read_csv(datasetpath + 'smooth_run-validation_batch64unit120withID-tag-epoch_loss.csv')
y = data_train_set['Value']

x = data_train_set['Step']
plt.figure(figsize=(8, 6))


plt.plot(x, y)
plt.xlabel("Epoch")
plt.ylabel("loss")

plt.title("loss")
plt.show()
