import random


def gaussian_mutation(solution, mutation_rate, sigma=0.1):
    """
    For each gene, with probability mutation_rate, we add a small random value
    The random value is drawn from a normal distribution centered at 0
    sigma controls how big the change can be - small sigma = small nudges
    This is the real-valued equivalent of bit-flip mutation from the knapsack problem
    """

    mutated = [gene for gene in solution]  # copy the solution

    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            mutated[i] = mutated[i] + random.gauss(0, sigma)

    return mutated


def uniform_reset_mutation(solution, mutation_rate, lower_bound=-1.0, upper_bound=1.0):
    # For each gene, with probability mutation_rate, we replace it with a completely new random value
    # Unlike gaussian mutation (which nudges), this can jump anywhere in the search space
    # Useful for escaping local optima when gaussian mutation is not enough

    mutated = [gene for gene in solution]  # copy the solution

    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            mutated[i] = random.uniform(lower_bound, upper_bound)

    return mutated