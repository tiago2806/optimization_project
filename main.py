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

def generate_uniform_for_project():
    return generate_solution_uniform(size, -1, 1)


def generate_gaussian_for_project():
    return generate_solution_gaussian(size, 0, 1)

def fitness_for_project(solution):
    return fitness_function(solution, model, X_train, y_train)

results_gridsearch = pd.DataFrame(columns = ['initalization', 'selection', 'crossover', 'mutation', 'mutation_rate', 'fitness score'])

print(results_gridsearch.head())

initialization = {
    "uniform": generate_uniform_for_project,
    "gaussian": generate_gaussian_for_project
}

selection = {
    'tournament': tournament_selection,
    'rank-based': rank_selection
}

crossover = {
    "arithmetic": arithmetic_crossover,
    "single": single_point_crossover,
    "blend": blend_crossover
}

mutation = {
    'gaussian': gaussian_mutation,
    'uniform-reset': uniform_reset_mutation
}

mutation_rate = [0.01, 0.05, 0.1]



for initialization_name, intialization_function in initialization.items():
    for selection_name, selection_operator in selection.items():
        for crossover_name, crossover_operator in crossover.items():
            for mutation_name, mutation_operator in mutation.items():
                for alpha in mutation_rate:
                    all_histories = 0
                    for i in range(30):
                        random.seed(i)
                        best_solution, history = genetic_algorithm(
                            intialization_function, 
                            fitness_for_project,
                            selection_operator,
                            crossover_operator,
                            mutation_operator,
                            100,
                            1000,
                            alpha,
                            verbose=True
                        )
                        
                        score = history[-1] #PARA CADA RUN, history é uma lista em que cada elemetno é o melhor fitness da geração [G0,G1,...,G100] e o best fitness é considerado o ultimo valor
                        all_histories += score #adicionamos essa fitness, respetiva ao numero da run, à lista all_histories
                        #a fitness final de uma combinação de parametros, vai ser a media final da lista all_histories, que contem a melhor fitness de cada run

                    options = {
                        'initialization': initialization_name,
                        'selection': selection_name,
                        'crossover': crossover_name,
                        'mutation': mutation_name,
                        'mutation_rate': alpha,
                        'results': all_histories/30
                    }

                    results_gridsearch = results_gridsearch.append(options, ignore_index = True)
                    #cada row da resutls_gridsearch DataFrame corresponde a uma combinação de parâmetros e o valor na coluna results, corresponde à fitness final associada a essa combinação


best_GA_combination = results_gridsearch.sort_values("results", ascending=False).iloc[0,:-1]

best_initialization_function = initialization[best_GA_combination["initialization"]]
best_selection_function = selection[best_GA_combination["selection"]]
best_crossover_function = crossover[best_GA_combination["crossover"]]
best_mutation_function = mutation[best_GA_combination["mutation"]]
best_mutation_rate = best_GA_combination["mutation_rate"]

test_scores = []

for i in range(30): #run 30 times on the test set, save the score per run, then average the score, to get a final, robust score, and not get a good score, because of a lucky run.
    best_solution, history = genetic_algorithm(
        best_initialization_function,
        fitness_for_project,
        best_selection_function,
        best_crossover_function,
        best_mutation_function,
        100,
        100,
        best_mutation_rate,
        verbose=True
    )

    test_score_per_run = fitness_function(best_solution, model, X_test, y_test)
    test_scores.append(test_score_per_run)

final_avg_test_score = sum(test_scores) / len(test_scores)

print(final_avg_test_score)
