import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from neural_networks import generate_solution_uniform, generate_solution_gaussian, fitness_function
from genetic_algorithm import genetic_algorithm
from PSO import pso
from crossover import arithmetic_crossover, blend_crossover, single_point_crossover
from mutators import gaussian_mutation, uniform_reset_mutation
from selectors import rank_selection
from sklearn.model_selection import train_test_split


#Prepare data
parkinson = pd.read_csv('parkinson.csv')

#Drop the target variable
X = parkinson.drop('status', axis = 1)
y = parkinson['status']

X_train, X_test, y_train, y_test = train_test_split(X,y , test_size=0.2, stratify=y)



def run_multiple_times(algorithm, params, n_runs=10):
    '''
    Run the algorithm n_runs times and average the convergence histories.
    This is necessary because both GA and PSO are stochastic (random) which means that
    a single run could be lucky or unlucky - averaging gives a fairer picture!

    Parameters: 
    algorithm - the algorithm used
    params - the parameters the algorithm is going to use
    n_runs - the number of iterations we're going to run the algorithm

    Output:
    avg_history -> list : returns an averaged convergence curve. 

    '''
    all_histories = []

    for run in range(n_runs):
        print(f"\n--- Run {run + 1}/{n_runs} ---")
        result = algorithm(**params)
        #GA returns (best_solution, history)
        #PSO returns (best_solution, best_fitness, history), therefore the history is the last element of result
        history = result[-1]
        all_histories.append(history)

    #Average the fitness at each generation/iteration across all runs
    n_steps = len(all_histories[0])
    avg_history = []
    for step in range(n_steps):
        total = 0
        for run in range(n_runs):
            total += all_histories[run][step]
        avg_history.append(total / n_runs)

    return avg_history


def plot_comparison(history_ga, history_pso):
    plt.plot(history_ga, label="Genetic Algorithm")
    plt.plot(history_pso, label="PSO")
    plt.xlabel("Generation / Iteration")
    plt.ylabel("Best Fitness (Recall)")
    plt.title("GA vs PSO - Average over multiple runs")
    plt.legend()
    plt.show()


if __name__ == "__main__":

    N_RUNS = 30 

    '''Run GA'''
    print("=" * 50)
    print("Running Genetic Algorithm...")
    print("=" * 50)

    ga_params_uniform = {
        "generate_solution": generate_solution_uniform, 
        "fitness_function": fitness_function,
        "selection": rank_selection,
        "crossover": arithmetic_crossover,
        "mutation": gaussian_mutation,
        "pop_size": 50,
        "n_generations": 50,
        "mutation_rate": 0.1,
        "verbose": True
    }

    avg_history_ga = run_multiple_times(genetic_algorithm, ga_params, n_runs=N_RUNS)

    '''Run PSO'''
    print("\n" + "=" * 50)
    print("Running PSO...")
    print("=" * 50)

    pso_params = {
        "generate_solution": [generate_solution_uniform, generate_solution_gaussian],
        "fitness_function": fitness_function,
        "n_particles": 30,
        "n_iterations": 50,
        "w": 0.7,
        "c1": 1.5,
        "c2": 1.5,
        "verbose": True
    }

    avg_history_pso = run_multiple_times(pso, pso_params, n_runs=N_RUNS)

    '''Compare'''
    print("\n" + "=" * 50)
    print(f"GA   final avg fitness: {avg_history_ga[-1]:.4f}")
    print(f"PSO  final avg fitness: {avg_history_pso[-1]:.4f}")
    print("=" * 50)

    plot_comparison(avg_history_ga, avg_history_pso)