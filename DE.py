import random
import numpy as np
from population import initialize_pop, evaluate_pop


def differential_evolution(generate_solution, fitness_function,
                            pop_size=30, n_generations=100,
                            F=0.8, CR=0.7,
                            verbose=True):
    """
    Differential Evolution (DE) - DE/rand/1/bin variant.

    How it works:
    For each individual x in the population (the target vector):
        1. Mutation: pick 3 other random individuals (a, b, c)
                     create a mutant vector v = a + F * (b - c)
        2. Crossover: mix x and v gene by gene with probability CR
                     to create a trial vector u
        3. Selection: if u is better than x, replace x with u

    Parameters
    ----------
    F  -> float: scaling factor, controls how much b-c difference is applied (typically 0.5-1.0)
    CR -> float: crossover rate, controls how many genes come from the mutant (typically 0.5-0.9)
    """

    # Step 1: initialise population and evaluate it
    population = initialize_pop(generate_solution, pop_size)
    pop_fits = evaluate_pop(fitness_function, population)

    # Keep track of the best solution found so far
    best_idx = np.argmax(pop_fits)
    best_solution = population[best_idx][:]
    best_fitness = pop_fits[best_idx]

    history = [best_fitness]

    # Step 2: evolve for n_generations
    for gen in range(n_generations):

        new_population = []
        new_fits = []

        for i in range(pop_size):

            # --- Mutation: pick 3 individuals different from i and from each other ---
            candidates = list(range(pop_size))
            candidates.remove(i)
            a_idx, b_idx, c_idx = random.sample(candidates, 3)

            a = population[a_idx]
            b = population[b_idx]
            c = population[c_idx]

            # Create mutant vector: v = a + F * (b - c)
            mutant = []
            for j in range(len(a)):
                mutant.append(a[j] + F * (b[j] - c[j]))

            # --- Crossover: mix the target x[i] with the mutant gene by gene ---
            # We guarantee at least one gene comes from the mutant (j_rand)
            target = population[i]
            j_rand = random.randint(0, len(target) - 1)

            trial = []
            for j in range(len(target)):
                if random.random() < CR or j == j_rand:
                    trial.append(mutant[j])
                else:
                    trial.append(target[j])

            # --- Selection: keep whichever is better between target and trial ---
            trial_fit = fitness_function(trial)

            if trial_fit >= pop_fits[i]:
                new_population.append(trial)
                new_fits.append(trial_fit)
            else:
                new_population.append(population[i][:])
                new_fits.append(pop_fits[i])

        # Replace the old population
        population = new_population
        pop_fits = new_fits

        # Update best solution
        current_best_idx = np.argmax(pop_fits)
        current_best_fitness = pop_fits[current_best_idx]

        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_solution = population[current_best_idx][:]

        history.append(best_fitness)

        if verbose:
            print(f"Generation {gen + 1}/{n_generations}  best fitness: {best_fitness:.4f}")

    return best_solution, best_fitness, history