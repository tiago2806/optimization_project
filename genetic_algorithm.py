import random
import numpy as np
 
 
def genetic_algorithm(generate_solution, fitness_function, selection,
                      crossover, mutation, pop_size, n_generations, mutation_rate,
                      verbose=True):
 
    '''Step 1: initialise the population and evaluate it'''
    population = [generate_solution() for _ in range(pop_size)]
    pop_fits = [fitness_function(ind) for ind in population]
 
    '''Keep track of the best solution found so far'''
    best_idx = np.argmax(pop_fits)
    best_solution = population[best_idx]
    best_fitness = pop_fits[best_idx]
 
    history = [best_fitness]
 
    '''Step 2: evolve for n_generations'''
    for gen in range(n_generations):
 
        new_population = []
 
        '''Fill the new population with children'''
        while len(new_population) < pop_size:
            '''Selection: pick two parents'''
            parent1 = selection(population, pop_fits)
            parent2 = selection(population, pop_fits)
 
            '''Crossover: combine the two parents to create two children'''
            child1, child2 = crossover(parent1, parent2)
 
            '''Mutation: make small random changes to each child'''
            child1 = mutation(child1, mutation_rate)
            child2 = mutation(child2, mutation_rate)
 
            new_population.append(child1)
            new_population.append(child2)
 
        '''Trim to pop_size in case we added one extra'''
        population = new_population[:pop_size]
 
        '''Evaluate the new population'''
        pop_fits = [fitness_function(ind) for ind in population]
 
        '''Update best solution if we found something better this generation'''
        current_best_idx = np.argmax(pop_fits)
        current_best_fitness = pop_fits[current_best_idx]
 
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_solution = population[current_best_idx]
 
        history.append(best_fitness)
 
        if verbose:
            print(f"Generation {gen + 1}/{n_generations}  best fitness: {best_fitness:.4f}")
 
    return best_solution, history