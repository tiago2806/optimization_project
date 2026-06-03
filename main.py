import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from neural_networks import (generate_solution_uniform, 
                            generate_solution_gaussian, 
                            fitness_function, create_model, 
                            initialize_model, 
                            get_solution_size
                        )
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

results_list_GA = []

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
                    for i in range(2):
                        random.seed(i)
                        best_solution, history = genetic_algorithm(
                            intialization_function, 
                            fitness_for_project,
                            selection_operator,
                            crossover_operator,
                            mutation_operator,
                            100,
                            100,
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
                        'results': all_histories/2
                    }

                    results_list_GA.append(options)
                    #cada row da resutls_gridsearch DataFrame corresponde a uma combinação de parâmetros e o valor na coluna results, corresponde à fitness final associada a essa combinação

results_gridsearch_GA = pd.DataFrame(results_list_GA)

best_GA_combination = results_gridsearch_GA.sort_values("results", ascending=False).iloc[0,:-1]

print(results_gridsearch_GA.sort_values("results", ascending=False))

best_initialization_function = initialization[best_GA_combination["initialization"]]
best_selection_function = selection[best_GA_combination["selection"]]
best_crossover_function = crossover[best_GA_combination["crossover"]]
best_mutation_function = mutation[best_GA_combination["mutation"]]
best_mutation_rate = best_GA_combination["mutation_rate"]

print()
print("Run 5 iterations with the best combination of parameterso of the genetic algorithm on the test set:")
print()
test_scores_GA = []

for i in range(2): #run 5 times on the test set, save the score per run, then average the score, to get a final, robust score, and not get a good score, because of a lucky run.
    best_solution, history=genetic_algorithm(
        generate_solution=best_initialization_function,
        fitness_function=fitness_for_project,
        selection=best_selection_function,
        crossover=best_crossover_function,
        mutation=best_mutation_function,
        pop_size=100,
        n_generations=100,
        mutation_rate=best_mutation_rate,
        verbose=True
    )

    test_score_per_run = fitness_function(best_solution, model, X_test, y_test)
    test_scores_GA.append(test_score_per_run)

print(plt.plot(test_scores_GA))

final_avg_test_score_GA = sum(test_scores_GA) / len(test_scores_GA)

print()
print(f"PSO final avg test score: {final_avg_test_score_GA:.4f}")

##Agora a parte do PSO e para perceberes melhor:

# PSO has no crossover/mutation operators, so we grid search over:
#   - initialization (same two as GA)
#   - w  (inertia weight)
#   - c1 (cognitive component)
#   - c2 (social component)

results_list_PSO = []

##Isto aqui são exemplos de valores (que vamos ter de ir experimentando) 
w_values  = [0.4, 0.7, 0.9]
c1_values = [1.0, 1.5, 2.0]
c2_values = [1.0, 1.5, 2.0]
 
for initialization_name, initialization_function in initialization.items():
    for w in w_values:
        for c1 in c1_values:
            for c2 in c2_values:
                all_histories = 0
                for i in range(2):
                    random.seed(i)
                    best_position, best_fitness, history = pso(
                        initialization_function,
                        fitness_for_project,
                        n_particles=30,
                        n_iterations=100,
                        w=w,
                        c1=c1,
                        c2=c2,
                        verbose=False
                    )
                    # Same logic as GA: take the last value of history (best fitness of each iteration)
                    score = history[-1]
                    all_histories += score
 
                options = {
                    'initialization': initialization_name,
                    'w':  w,
                    'c1': c1,
                    'c2': c2,
                    'results': all_histories / 2
                }

                results_list_PSO.append(options)
 

results_gridsearch_PSO = pd.DataFrame(results_list_PSO)

# Find best PSO combination and evaluate on test set
best_PSO_combination = results_gridsearch_PSO.sort_values("results", ascending=False).iloc[0,:-1]
 
best_PSO_initialization = initialization[best_PSO_combination["initialization"]]
best_w  = best_PSO_combination["w"]
best_c1 = best_PSO_combination["c1"]
best_c2 = best_PSO_combination["c2"]
 
test_scores_PSO = []
 
for i in range(2):
    random.seed(i)
    best_position, best_fitness, history = pso(
        best_PSO_initialization,
        fitness_for_project,
        n_particles=30,
        n_iterations=100,
        w=best_w,
        c1=best_c1,
        c2=best_c2,
        verbose=False
    )
    test_score_per_run = fitness_function(best_position, model, X_test, y_test)
    test_scores_PSO.append(test_score_per_run)


print(plt.plot(test_scores_PSO))
 
final_avg_test_score_PSO = sum(test_scores_PSO) / len(test_scores_PSO)
print(f"PSO final avg test score: {final_avg_test_score_PSO:.4f}")


print("\n=== Final Results ===")

print("\n=== Best GA parameters ===")
print(best_GA_combination)
 
print("\n=== Best PSO parameters ===")
print(best_PSO_combination)

print("Comparison between algorithms on the test set:\n")

print(f"GA  best avg test score: {final_avg_test_score_GA:.4f}\n")
print(f"PSO best avg test score: {final_avg_test_score_PSO:.4f}")

