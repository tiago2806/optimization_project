import pandas as pd
import numpy as np
import random
import utils



def generate_solution(size, lower_bound, upper_bound):
    """
    In this context, a solution is a vector of weights.
    Parameters
    ----------
    size->int: Size of the vector, it is equal to the number of weights in the network

    lower_bound->float

    
    Output
    ----------
    
    """
    
    return [random.uniform(lower_bound,upper_bound) for weight in range(size)]



def fitness_function(solution):

    #aqui acho que introduzimos a solution na neural network (que vai ser criada noutro ficheiro e dps chamamos a função)
    #esta função depois vai retornar a predictive performance (para isso temos de escolher uma metric)
    pass