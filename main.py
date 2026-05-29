import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from neural_networks import generate_solution_uniform, generate_solution_gaussian, fitness_function
from genetic_algorithm import genetic_algorithm
from PSO import pso
from crossover import arithmetic_crossover, blend_crossover, single_point_crossover
from mutators import gaussian_mutation, uniform_reset_mutation
from selectors import rank_selection, tournament_selection
from sklearn.model_selection import train_test_split


#Prepare data
parkinson = pd.read_csv('parkinson.csv')

#Drop the target variable
X = parkinson.drop('status', axis = 1)
y = parkinson['status']

X_train, X_test, y_train, y_test = train_test_split(X,y , test_size=0.2, stratify=y)


results_gridsearch = pd.DataFrame(columns = ['initialization','selection', 'crossover', 'mutation'])

