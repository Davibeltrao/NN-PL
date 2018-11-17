import csv
import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd
from sklearn import preprocessing
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
import time


def load_data_kfold(k):
    df = pd.read_csv('sdss.csv', sep=',')
    
    y_data = df['class']
    print(y_data.head(10))
    del df['class']
    X_data = np.array(df)
    # X_data = df
    print(X_data.shape)

    scaler = StandardScaler()
    X_data = scaler.fit_transform(X_data)
    # print(X_data.head(5))
    # print(scaled_data)

    folds = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=1)
                 .split(X_data, y_data))

    folds = np.array(folds)

    label_encoder = preprocessing.LabelEncoder()
    y_data = label_encoder.fit_transform(y_data)
    y_data = to_categorical(y_data, 3)
    
    print(folds)
    # print(y_data.reshape(-1, 1))
    print("Y data")
    print(y_data.shape)

    return folds, X_data, y_data
