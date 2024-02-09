import random
import matplotlib.pyplot as plt
import numpy as np
from random import choices
import time
def generate_solution(x):#Function that randomly decides whether to generate string of all 0's or 1's & 0's
    if random.choice([True, False]):  # Randomly choose between all 0's and 1's & 0's
        return [0] * x
    else:
        return choices([0, 1], k=x)

def fitness(x):
    count = sum(x)

    return count if count > 0 else 2 * len(x)

def selection(population, fitnesses):
    tournament = random.sample(range(len(population)), k=3)
    tournament_fitnesses = [fitnesses[i] for i in tournament]
    winner_index = tournament[np.argmax(tournament_fitnesses)]
    return population[winner_index]

def crossover(p1, p2):
    pt = random.randint(0, len(p1))  # Random crossover point
    c1 = p1[:pt] + p2[pt:]
    c2 = p2[:pt] + p1[pt:]
    return c1, c2

def mutation(solution, mutation_rate):
    mutated_solution = solution.copy()
    for i in range(len(mutated_solution)):
        if random.random() < mutation_rate:
            mutated_solution[i] = 1 - mutated_solution[i]  # Flip 0 to 1 and vice versa
    return mutated_solution

def genetic_algorithm_with_plot(string_length, iterations, pop_size, mutation_rate):
    population = [generate_solution(string_length) for _ in range(pop_size)]
    best, best_eval = None, float('-inf')
    avg_fitnesses = []
    start_time=time.time()

    for gen in range(iterations):
        scores = [fitness(c) for c in population]
        avg_fitness = sum(scores) / len(scores)
        avg_fitnesses.append(avg_fitness)

        for i in range(pop_size):
            if scores[i] > best_eval:
                best, best_eval = population[i], scores[i]
                print(">%d, new best f(%s)" % (gen, population[i]))

        select = [selection(population, scores) for _ in range(pop_size)]

        newPop = []
        for i in range(0, len(population), 2):
            p1 = select[i]
            p2 = select[i + 1]

            c1, c2 = crossover(p1, p2)
            c1 = mutation(c1, mutation_rate)
            c2 = mutation(c2, mutation_rate)

            newPop.extend([c1, c2])
        population = newPop

        current_time = time.time()
        elapsed_time = current_time - start_time

        plt.plot([elapsed_time], [avg_fitness], 'b.')

    plt.xlabel('Elapsed Time (seconds)')
    plt.ylabel('Average Fitness')
    plt.title('Average Fitness of Population vs Elapsed Time')
    plt.show()

    plt.plot(range(gen + 1), avg_fitnesses)
    plt.xlabel('Generations')
    plt.ylabel('Average Fitness')
    plt.title('Average Fitness of Population vs Generations')
    plt.show()

    return best

string_length = 30
iterations = 100
pop = 100
mutation_rate = .01

best_solution = genetic_algorithm_with_plot(string_length, iterations, pop, mutation_rate)
print('Best solution:', best_solution)