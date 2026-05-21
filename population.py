def initialize_pop(individual_generator, pop_size):
    '''Returning a list with pop_size number of individuals'''
    return [individual_generator() for _ in range(pop_size)]


def evaluate_pop(fitness_function, pop):
    '''Returning a list with the fitness of each individual in the population'''
    return [fitness_function(ind) for ind in pop]