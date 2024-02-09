from random import choices
import numpy as np
import random
import matplotlib.pyplot as plt

def generate_solution(x):
    return choices([0,1],k=x)#Uses choice function to create solution of x length containing 1s & 0s only

def fitness(x):
    return sum(x)#returns sum of solution

def selection(population, fitnesses):  # tournament selection
    tournament = random.sample(range(len(population)), k=3)#randomly selects 3 random solutions for tournament selection
    tournament_fitnesses = [fitnesses[i] for i in tournament]#Finds fitness of each solution selected for tournament
    winner_index = tournament[np.argmax(tournament_fitnesses)]
    return population[winner_index]

def crossover(p1,p2):
    c1=p1.copy()
    c2=p2.copy()
    pt=2

    c1=p1[:pt]+p2[pt:]#Creates child by merging part of parent 1 past the cross point & part of parent 2 before crossover point, vice versa for each child
    c2=p1[pt:]+p2[:pt]

    return [c1,c2]

def mutation(solution, mutation_rate):
    mutated_solution = solution.copy()
    for i in range(len(mutated_solution)):
        if random.random() < mutation_rate:
            mutated_solution[i] = 1 - mutated_solution[i]  # Flip 0 to 1 and vice versa
    return mutated_solution

def genetic_algorithm(func,string_length,iterations,pop_size,mutation_rate):
    population = [generate_solution(string_length) for _ in range(pop_size)]#Creates population
    best,best_eval=0,func(population[0])#Initialize best fitness value & best solution
    avg_fit=[]
    for gen in range(iterations):#iterates through generations
        scores = [func(c) for c in population]#loop to set fitness scores for each solution in population
        avg_fitness=sum(scores)/len(scores)
        avg_fit.append(avg_fitness)
        for i in range(pop_size):#Loop to iterate through population
            if scores[i] > best_eval:#if score of solution is higher than best score saved then the best score is reset & printed, finds best solution of each generation
                best, best_eval = population[i], scores[i]
                print(">%d, new best f(%s) = %.3f" % (gen, population[i], scores[i]))

        select=[selection(population,scores)for _ in range(pop_size)]#Tournament selection of solutions in population

        newPop=[]
        for i in range(0,len(population),2):#loops through,iterating 2 at a time
            p1=select[i]
            p2=select[i+1]

            c1,c2=crossover(p1,p2)#Crossover of 2 parents selected
            c1=mutation(c1,mutation_rate)#mutation of child1
            c2=mutation(c2,mutation_rate)#mutation of child2

            newPop.extend([c1,c2])
        population=newPop#resets population to new pop

    plt.plot(range(iterations), avg_fit)
    plt.xlabel('Generations')
    plt.ylabel('Average Fitness')
    plt.title('Average Fitness of Population vs Generations')
    plt.show()
    return [best,best_eval]

string_length=30
iterations=100
pop=100
mutation_rate=.01



best,score=genetic_algorithm(fitness,string_length,iterations,100,mutation_rate)#Running GA

print('f(%s) = %f' % (best, score))
