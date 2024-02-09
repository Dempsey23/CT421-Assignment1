import random
import numpy as np


def generate_solution(weights_counts):

    return [(random.randint(0, len(weights_counts) - 1), weight, count) for weight, count in weights_counts]#Generate intitial solutions in tuples of (bin_index,weights,count)

def fitness(solution, bin_capacity):
    bin_weights = [0 for _ in range(len(solution))]  # Creates bin for each item in solution

    for bin_index, weight, count in solution:
        if bin_index < len(bin_weights):
            bin_weights[bin_index] += weight * count  # Add total weight for each bin

    # Count the number of bins used without exceeding capacity
    bins_used = 0
    for weight in bin_weights:
        if weight > bin_capacity:
            return float('inf')  # Return invalid solution
        if weight > 0:
            bins_used += 1

    return bins_used

def selection(population, fitnesses):
    tournament = random.sample(range(len(population)), k=3)
    tournament_fitnesses = [fitnesses[i] for i in tournament]
    winner_index = tournament[np.argmin(tournament_fitnesses)]
    return population[winner_index]


def crossover_bin_packing(p1, p2):
    pt = 2
    c1 = p1[:pt] + p2[pt:]
    c2 = p2[:pt] + p1[pt:]
    return [c1, c2]


def mutation(solution, mutation_rate):#Code altered to account for different data type used
    mutated_solution = solution.copy()
    for i in range(len(mutated_solution)):
        if random.random() < mutation_rate:
            mutated_solution[i] = (
                random.randint(0, len(mutated_solution) - 1),
                mutated_solution[i][1],
                mutated_solution[i][2]
            )
    return mutated_solution


def genetic_algorithm_bin_packing(weights_counts, bin_capacity, iterations, pop_size, mutation_rate):
    population = [generate_solution(weights_counts) for _ in range(pop_size)]
    best, best_eval = population[0], fitness(population[0], bin_capacity)

    for gen in range(iterations):
        scores = [fitness(c, bin_capacity) for c in population]

        for i in range(len(scores)):
            score = scores[i]
            if score < best_eval:
                best, best_eval = population[i], score
                print(">%d, new best f(%s) = %.3f" % (gen, best, score))

        select = [selection(population, scores) for _ in range(pop_size)]

        newPop = []
        for i in range(0, pop_size, 2):
            p1, p2 = select[i], select[i + 1]
            c1, c2 = crossover_bin_packing(p1, p2)
            c1 = mutation(c1, mutation_rate)
            c2 = mutation(c2, mutation_rate)
            newPop.extend([c1, c2])
        population = newPop

    return best, best_eval


# Parameters
bin_capacity = 1000
iterations = 100
pop_size = 100
mutation_rate = 0.01

#Weight counts from data
weights_counts = [
    (200, 3), (199, 1), (198, 2), (197, 2), (194, 2), (193, 1),
    (192, 1), (191, 3), (190, 2), (189, 1), (188, 2), (187, 2),
    (186, 1), (185, 4), (184, 3), (183, 3), (182, 3), (181, 2),
    (180, 1), (179, 4), (178, 1), (177, 4), (175, 1), (174, 1),
    (173, 2), (172, 1), (171, 3), (170, 2), (169, 3), (167, 2),
    (165, 2), (164, 1), (163, 4), (162, 1), (161, 1), (160, 2),
    (159, 1), (158, 3), (157, 1), (156, 6), (155, 3), (154, 2),
    (153, 1), (152, 3), (151, 2), (150, 4)
]
weights_counts2 = [
    (200, 2), (199, 4), (198, 1), (197, 1), (196, 2), (195, 2),
    (194, 2), (193, 1), (191, 2), (190, 1), (189, 2), (188, 1),
    (187, 2), (186, 1), (185, 2), (184, 5), (183, 1), (182, 1),
    (181, 3), (180, 2), (179, 2), (178, 1), (176, 1), (175, 2),
    (174, 5), (173, 1), (172, 3), (171, 1), (170, 4), (169, 2),
    (168, 1), (167, 5), (165, 2), (164, 2), (163, 3), (162, 2),
    (160, 2), (159, 2), (158, 2), (157, 4), (156, 3), (155, 2),
    (154, 1), (153, 3), (152, 2), (151, 2), (150, 2)
]
weights_counts3=[
    (200, 1), (199, 2), (197, 2), (196, 2), (193, 3), (192, 2),
    (191, 2), (190, 2), (189, 3), (188, 1), (187, 1), (185, 3),
    (183, 2), (182, 1), (181, 3), (180, 3), (179, 3), (178, 1),
    (177, 5), (176, 2), (175, 5), (174, 4), (173, 1), (171, 3),
    (170, 1), (169, 2), (168, 5), (167, 1), (166, 4), (165, 2),
    (163, 1), (162, 2), (161, 2), (160, 3), (159, 2), (158, 2),
    (157, 1), (156, 3), (155, 3), (154, 1), (153, 2), (152, 3),
    (151, 2), (150, 1)
]
weights_counts4=[
    (200, 3), (199, 5), (198, 4), (197, 1), (195, 1),
    (193, 4), (192, 1), (188, 1), (187, 1), (186, 3),
    (185, 3), (184, 2), (183, 2), (182, 1), (181, 1),
    (180, 3), (179, 2), (178, 6), (177, 2), (176, 4),
    (175, 1), (173, 4), (172, 4), (170, 1), (169, 3),
    (168, 4), (167, 1), (165, 3), (164, 1), (163, 2),
    (162, 4), (161, 1), (160, 3), (159, 3), (158, 1),
    (157, 3), (155, 2), (154, 3), (153, 1), (152, 3),
    (151, 1), (150, 1)
]
weights_counts5=[
    (200, 5), (199, 2), (198, 2), (197, 2), (196, 1),
    (195, 3), (194, 2), (193, 2), (192, 4), (191, 2),
    (190, 4), (188, 3), (187, 2), (186, 2), (185, 1),
    (184, 1), (183, 1), (182, 1), (181, 3), (180, 1),
    (178, 3), (177, 2), (176, 2), (174, 1), (173, 1),
    (172, 1), (171, 3), (168, 2), (167, 1), (165, 1),
    (164, 1), (163, 1), (162, 3), (161, 3), (160, 3),
    (159, 2), (158, 3), (157, 3), (156, 2), (155, 5),
    (154, 3), (153, 3), (151, 5), (150, 2)
]

best, score = genetic_algorithm_bin_packing(weights_counts2, bin_capacity, iterations, pop_size, mutation_rate)

print('Best Solution:', best)
print(len(best),len(weights_counts))
print('Number of Used Bins:', score)

