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


def blend_crossover(parent1, parent2, alpha = 0.3):
    """
    BLX-alpha crossover for real-valued vectors.

    This operator is suitable for continuous optimization problems because
    each child gene is sampled from an interval around the values of the
    corresponding parent genes.

    alpha controls how much the interval is expanded:
    - alpha = 0.0: children are sampled only between the parents
    - alpha > 0.0: children can also be sampled slightly outside the parents
    """

    child1 = []
    child2 = []

    for i in range(len(parent1)):
        x1 = parent1[i]
        x2 = parent2[i]

        lower_bound = min(x1,x2)
        higher_bound = max(x1,x2)

        distance = higher_bound - lower_bound

        new_lower_bound = lower_bound - alpha * distance
        new_higher_bound = higher_bound + alpha * distance

        child1.append(random.uniform(new_lower_bound, new_higher_bound))        
        child2.append(random.uniform(new_lower_bound, new_higher_bound))        

    return child1, child2 
#o offspring vai ser sempre melhor do que o pior dos pais

