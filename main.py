import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from neural_networks import (generate_solution_uniform, generate_solution_gaussian, fitness_function, create_model, initialize_model, get_solution_size)
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = create_model()
model = initialize_model(model, X_train, y_train)
size = get_solution_size(model) 

results_gridsearch = pd.DataFrame(columns = ['initalization', 'selection', 'crossover', 'mutation', 'mutation_rate', 'fitness score'])

print(results_gridsearch.head())

initialization = [generate_solution_uniform, generate_solution_gaussian]
selection = [tournament_selection, rank_selection]
crossover = [arithmetic_crossover, blend_crossover, single_point_crossover]
mutation = [gaussian_mutation, uniform_reset_mutation]
mutation_rate = [0.01, 0.05, 0.1]

for first_parameter in initialization:
    for second_parameter in selection:
        for third_parameter in crossover:
            for fourth_parameter in mutation:
                for fifth_parameter in mutation_rate:
                    sum_result = 0
                    for i in range(30):
                        random.seed(i)
                        best_solution, history = genetic_algorithm(
                            first_parameter, 
                            fitness_function,
                            second_parameter,
                            third_parameter,
                            fourth_parameter,
                            100,
                            1000,
                            fifth_parameter,
                            verbose=True
                        )

        #avg 30 runs ga <- score destes parametros
        #append dataframe com os parametros e o score
                    score = fitness_function(best_solution, model, X_train, y_train)
                    sum_result += score

            options = {
                'initialization': first_parameter,
                'selection': second_parameter,
                'crossover': third_parameter,
                'mutation': fourth_parameter,
                'mutation_rate': fifth_parameter,
                'results': sum_result/30
            }

results_gridsearch = results_gridsearch.append(options, ignore_index = True)
