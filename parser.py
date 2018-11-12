import csv
import numpy as np

from numpy import genfromtxt
data = genfromtxt('sdss.csv', delimiter=',')
data = data[1:]
print(data)

X_data = data[:, :-1]
y_data = data[:, -1]

print(X_data.shape)
print(y_data.reshape(-1, 1).shape)