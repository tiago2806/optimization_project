import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


parkinson = pd.read_csv('parkinson.csv')

print(parkinson.head())

X = parkinson.drop('status', axis = 1)
y = parkinson['status']

X_train, X_test, y_train, y_test = train_test_split(X,y , test_size=0.2, stratify=y)