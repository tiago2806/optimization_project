import matplotlib.pyplot as plt
import numpy as np

from structure import generate_solution, generate_solution_v2, fitness_function
from algorithms import genetic_algorithm, pso
from xover_operators import arithmetic_crossover, single_point_crossover
from mutators import gaussian_mutation, uniform_reset_mutation
from selectors import tournament_selection, roulette_selection


def run_multiple_times(algorithm_fn, params, n_runs=10):
    '''Run the algorithm n_runs times and average the convergence histories'''
    '''This is necessary because both GA and PSO are stochastic (random)'''
    '''A single run could be lucky or unlucky - averaging gives a fairer picture'''
    all_histories = []

    for run in range(n_runs):
        print(f"\n--- Run {run + 1}/{n_runs} ---")
        result = algorithm_fn(**params)
        '''GA returns (best_solution, history)'''
        '''PSO returns (best_solution, best_fitness, history)'''
        history = result[-1]
        all_histories.append(history)

    '''Average the fitness at each generation/iteration across all runs'''
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
    plt.ylabel("Best Fitness (Macro Recall)")
    plt.title("GA vs PSO - Average over multiple runs")
    plt.legend()
    plt.show()


if __name__ == "__main__":

    N_RUNS = 10  
    '''increase for the final report (20-30 is better)'''

    '''Run GA'''
    print("=" * 50)
    print("Running Genetic Algorithm...")
    print("=" * 50)

    ga_params = {
        "generate_solution": generate_solution,
        "fitness_function": fitness_function,
        "selection": tournament_selection,
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
        "generate_solution": generate_solution,
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