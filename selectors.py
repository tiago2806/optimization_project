import random


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