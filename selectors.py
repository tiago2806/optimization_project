import random


def tournament_selection(population, fitnesses, tournament_size = 3, maximize=True):
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