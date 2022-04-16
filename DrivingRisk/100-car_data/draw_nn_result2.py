import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib import ticker
from matplotlib.font_manager import FontProperties
import csv

datasetpath = 'E:/MachineLearning/DrivingRisk/100-car_data/training_score_data/'
csvname = 'smooth_run-validation_batch64unit120withID-tag-epoch_accuracy.csv'

def readcsv(files):
    csvfile = open(files, 'r')
    plots = csv.reader(csvfile, delimiter=',')
    x = []
    y = []
    for row in plots:
        y.append((row[2]))
        x.append((row[1]))
    return x, y


plt.figure(figsize=(8, 6))

x, y = readcsv(datasetpath+ csvname)
ticker_spacing = 15
fig,ax = plt.subplots(1,1)
ax.plot(x,y)
ax.xaxis.set_major_locator(ticker.MultipleLocator(ticker_spacing))
plt.title('loss')
plt.xlabel('Steps')
plt.ylabel('Score')
plt.show()
