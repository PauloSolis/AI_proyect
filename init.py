import math, pandas as pd, sklearn, numpy as np
from datetime import datetime as dt
from dateutil.parser import parse
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split



class SVR:

    def __init__(self, kernel='rbf', gamma='scale', tol=1e-3, C=1.0, n_iters=100):
        self.kernel = kernel
        self.gamma = gamma
        self.tol = tol
        self.C = C
        self.n_iters = n_iters

    def score(self, X, y):

        y_pred = self.predict(X)
        return r2_score(y, y_pred, )


def plot_results(predicted_data, true_data, title='', xlab='', ylab=''):
    plt.title(title)
    plt.plot(range(len(predicted_data)), predicted_data, label='Prediction')
    plt.plot(range(len(true_data)), true_data, label='True Data')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend()
    plt.show()
    return


def newPlot():
    plt.figure(plt.gcf().number + 1)

