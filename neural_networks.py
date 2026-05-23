import pandas as pd
import numpy as np
import random
from utils import (
    evaluate_performance,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.neural_network import MLPClassifier

def create_model():
    return MLPClassifier(
        hidden_layer_sizes=(10,),
        max_iter = 1,
        random_state=42
    )

def initialize_model(model, X_train, y_train): #cria coefs_, intercepts_, etc
    model.fit(X_train, y_train)

    return model

def get_solution_size(model):
    total_size = 0

    for coef_matrix in model.coefs_:
        total_size += coef_matrix.size
    
    for bias_vector in model.intercepts_:
        total_size += bias_vector.size

    return total_size
    
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


def set_weights(model, solution):
    idx = 0

    for i in range(len(model.coefs_)):
        shape = model.coefs_[i].shape
        size = model.coefs_[i].size

        values = solution[idx:idx + size]
        model.coefs_[i] = np.array(values).reshape(shape)

        idx += size

    for i in range(len(model.intercepts_)):
        shape = model.intercepts_[i].shape
        size = model.intercepts_[i].size

        values = solution[idx:idx + size]
        model.intercepts[i] = np.array(values).reshape(shape)

        idx += size

def fitness_function(solution, model, X_train, y_train ):
    """
    Solution is just a vector of weights. MLP expects wieghts in matrices and vectors
    """

    set_weights(model, solution)

    
    y_pred = model.predict(X_train)

    return evaluate_performance(y_train, y_pred)