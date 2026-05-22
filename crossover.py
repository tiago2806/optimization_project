import random


def arithmetic_crossover(parent1, parent2):
    '''We pick a random blend factor alpha between 0 and 1'''
    '''Each gene of the child is a weighted average of the two parents'''
    '''Example: if alpha=0.3, child1 gets 30% of parent1 and 70% of parent2'''
    '''This makes sense for real-valued vectors because weights are continuous numbers'''

    alpha = random.random()

    child1 = [alpha * parent1[i] + (1 - alpha) * parent2[i] for i in range(len(parent1))]
    child2 = [(1 - alpha) * parent1[i] + alpha * parent2[i] for i in range(len(parent1))]

    return child1, child2


def single_point_crossover(parent1, parent2):
    '''We pick a random cut point and swap everything after it'''
    '''This is the same idea as the binary crossover from class, adapted for real values'''
    '''Since the genes are now floats instead of bits, no other change is needed'''

    cut_point = random.randint(1, len(parent1) - 1)

    child1 = parent1[:cut_point] + parent2[cut_point:]
    child2 = parent2[:cut_point] + parent1[cut_point:]

    return child1, child2


#o offspring vai ser sempre melhor do que o pior dos pais

#