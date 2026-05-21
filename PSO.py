import random
import numpy as np

def pso(generate_solution, fitness_function,
        n_particles=30, n_iterations=100,
        w=0.7, c1=1.5, c2=1.5,
        verbose=True):
 
    '''PSO: each particle is a candidate solution (a weight vector)'''
    '''Particles move through the search space guided by:
       - their own best position found so far (personal best)
       - the best position found by the entire swarm (global best)'''
 
    n_weights = len(generate_solution())
 
    '''Step 1: initialise particles and velocities'''
    particles = initialize_pop(generate_solution, n_particles)
 
    '''Velocity starts at zero for all particles'''
    velocities = [[0.0 for _ in range(n_weights)] for _ in range(n_particles)]
 
    '''Evaluate all particles'''
    fits = evaluate_pop(fitness_function, particles)
 
    '''Each particle remembers its own best position'''
    personal_best_positions = [particles[i][:] for i in range(n_particles)]
    personal_best_fits = [fits[i] for i in range(n_particles)]
 
    '''The swarm remembers the single best position found by anyone'''
    global_best_idx = np.argmax(personal_best_fits)
    global_best_position = personal_best_positions[global_best_idx][:]
    global_best_fitness = personal_best_fits[global_best_idx]
 
    history = [global_best_fitness]
 
    '''Step 2: iterate'''
    for iteration in range(n_iterations):
 
        for i in range(n_particles):
 
            '''Update velocity using the PSO update equation from the slides:'''
            '''v = w*v + c1*r1*(personal_best - position) + c2*r2*(global_best - position)'''
            r1 = random.random()
            r2 = random.random()
 
            new_velocity = []
            for j in range(n_weights):
                inertia = w * velocities[i][j]
                cognitive = c1 * r1 * (personal_best_positions[i][j] - particles[i][j])
                social = c2 * r2 * (global_best_position[j] - particles[i][j])
                new_velocity.append(inertia + cognitive + social)
 
            velocities[i] = new_velocity
 
            '''Update position: x = x + v'''
            new_position = []
            for j in range(n_weights):
                new_position.append(particles[i][j] + velocities[i][j])
 
            particles[i] = new_position
 
            '''Evaluate the new position'''
            new_fit = fitness_function(particles[i])
 
            '''Update personal best if this position is better'''
            if new_fit > personal_best_fits[i]:
                personal_best_positions[i] = particles[i][:]
                personal_best_fits[i] = new_fit
 
                '''Update global best if this is the best anyone has found'''
                if new_fit > global_best_fitness:
                    global_best_position = particles[i][:]
                    global_best_fitness = new_fit
 
        history.append(global_best_fitness)
 
        if verbose:
            print(f"Iteration {iteration + 1}/{n_iterations}  best fitness: {global_best_fitness:.4f}")
 
    return global_best_position, global_best_fitness, history