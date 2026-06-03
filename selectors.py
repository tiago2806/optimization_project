import random


def tournament_selection(population, fitnesses, tournament_size = 3, maximize=True):
    """
    Selects an individual from the population using tournament selection.
    A subset of individuals is randomly chosen from the population and the best individual among them (based on fitness) is selected.

    Parameters:
    - population -> list: Each generation has a population and each population is composed by various individuals (solutions).
    - fitnesses -> list: List of fitness values corresponding to each individual in the population.
    - tournament_size ->  int: Number of individuals randomly selected to compete in the tournament.
    - maximize -> bool: If True, selects the individual with the highest fitness. If False, selects the individual with the lowest fitness.

    ---------------
    Output:
    - best individual -> list: The selected best individual from the population.
    """
    
    indices = random.sample(range(len(population)), tournament_size)

    best_index = indices[0]

    for i in indices[1:]:
        if maximize:
            if fitnesses[i]>fitnesses[best_index]:
                best_index = i
        else:
            if fitnesses[i]<fitnesses[best_index]:
                best_index = i
            
    return population[best_index]


def rank_selection(population, fitnesses):
    """
    Selects an individual from the population using rank-based selection.
    Individuals are sorted based on their fitness, and selection probability is assigned according to their rank rather than their raw fitness value.
    Higher-ranked individuals have a higher chance of being selected

    Parameters:
    - population -> list: List of individuals (solutions)
    - fitnesses -> list: List of fitness values corresponding to each individual in the population.

    ---------------
    Output:
    - best individual -> list: The selected best individual from the population
    """

    paired = list(zip(population, fitnesses)) #zip combines each individual to its fitness using tuples
    paired.sort(key = lambda x: x[1])

    population_size = len(paired)

    ranks = list(range(1, population_size + 1))

    total_rank = sum(ranks)

    pick = random.uniform(0, total_rank)

    cumulative = 0

    for i in range(population_size):
        cumulative += ranks[i]
        if cumulative > pick:
            selected_individual = paired[i][0] #paired is a list with tuples... so tuple i, first element, the individual

            return selected_individual

    return paired[-1][0]