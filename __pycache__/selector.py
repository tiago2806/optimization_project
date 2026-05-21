import random

def tournament_selection(pop, pop_fits, tournament_size=3, maximize=True):
    '''Pick tournament_size random individuals and return the best one among them'''
    '''Using random.sample so the same individual can't appear twice in the tournament'''
    indices = random.sample(range(len(pop)), tournament_size)
    best_index = indices[0]

    for i in indices[1:]:
        if maximize:
            if pop_fits[i] > pop_fits[best_index]:
                best_index = i
        else:
            if pop_fits[i] < pop_fits[best_index]:
                best_index = i

    return pop[best_index]


def roulette_selection(pop, pop_fits, maximize=True):
    '''Each individual gets a probability of being selected proportional to its fitness'''
    '''Better individuals have a higher chance, but everyone has a chance'''
    '''This is different from tournament: here all individuals compete at the same time'''

    '''Shift fitnesses so the minimum is just above 0 (probabilities can't be negative)'''
    min_fit = min(pop_fits)
    shifted_fits = [f - min_fit + 0.0001 for f in pop_fits]

    total = sum(shifted_fits)

    '''Build a probability for each individual'''
    probabilities = [f / total for f in shifted_fits]

    '''Spin the wheel: pick a random number and walk through probabilities until we land'''
    spin = random.random()
    cumulative = 0.0

    for i in range(len(pop)):
        cumulative += probabilities[i]
        if spin <= cumulative:
            return pop[i]

    '''Fallback in case of floating point rounding'''
    return pop[-1]