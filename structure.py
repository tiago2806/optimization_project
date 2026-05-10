import pandas as pd
import numpy as np
import random
from utils import (
    evaluate_performance,
    fit_model
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.neural_network import MLPClassifier



dataset = pd.read_csv('parkinson.csv')


X = dataset.drop('status', axis = 1)
y = dataset['status']

print(y.value_counts())

#as we can observe, the dataset is unbalanced!!

X_train, X_test, y_train, Y_test = train_test_split(X, y, test_size= 0.2, stratify=y)

mlp = MLPClassifier(hidden_layer_sizes=(10,))

fitted_mlp = fit_model(mlp, X_train, y_train)

print(fitted_mlp.coefs_)
print()
print(fitted_mlp.intercepts_)
print([c.shape for c in mlp.coefs_])
print([b.shape for b in mlp.intercepts_])
def generate_solution(size, lower_bound, upper_bound):
    """
    In this context, a solution is a vector of weights.
    Parameters
    ----------
    size->int: Size of the vector, it is equal to the number of weights in the network

    lower_bound->float

    
    Output
    ----------
    
    """
    
    return [random.uniform(lower_bound,upper_bound) for weight in range(size)]



def fitness_function(solution, model, X_train, y_train):
    """
    Solution is just a vector of weights. MLP expects wieghts in matrices and vectors
    """
    idx = 0

    for i in range(len(model.coefs_)):
        shape = mlp.coefs_[i].shape
        size = shape[0] * shape[1]

        solution[idx:idx+size]

    
    y_pred = model.predict(X)

    return evaluate_performance(y, y_pred)