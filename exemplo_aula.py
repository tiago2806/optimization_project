#exemplo aula gridsearch
import pandas as pd


results_gridsearch = pd.DataFrame(columns = ['p1', 'p2', 'result'])

print(results_gridsearch.head())

p1 = [1,2,3,4,5]
p2 = [1,2,3,4,5]


for first_parameter in p1:
    for second_parameter in p2:
        #caluclaute the fitness using these parameters

        fitness = first_parameter**2 + second_parameter**""

        new_row = {
            'p1': first_parameter,
            'p2': second_parameter,
            'result': fitness       
        }

        results_gridsearch = results_gridsearch.append(new_row, ignore_index = True)

print(results_gridsearch)