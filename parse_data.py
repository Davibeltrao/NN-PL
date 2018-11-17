import csv
import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import train_test_split, StratifiedKFold


def load_data_kfold(k):
    data = genfromtxt('sdss.csv', delimiter=',')
    data = data[1:]
    print(data)

    X_data = data[:, :-1]
    y_data = data[:, -1]

    print(X_data.shape)
    print(y_data.reshape(-1, 1).shape)

    folds = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=1)
                 .split(X_data, y_data))

    folds = np.array(folds)

    return folds, X_data, y_data
