from random import choices
import numpy as np
import random
import matplotlib.pyplot as plt

def generate_solution(x):
    return choices([0,1],k=x)

def fitness(target, individual):
    return sum(target[i] == individual[i] for i in range(len(target)))
def selection(population, fitnesses):  # tournament selection
    tournament = random.sample(range(len(population)), k=3)
    tournament_fitnesses = [fitnesses[i] for i in tournament]
    winner_index = tournament[np.argmax(tournament_fitnesses)]
    return population[winner_index]

def crossover(p1,p2):
    pt=2
    c1=p1[:pt]+p2[pt:]
    c2=p1[pt:]+p2[:pt]

    return [c1,c2]

def mutation(solution, mutation_rate):
    mutated_solution = solution.copy()
    for i in range(len(mutated_solution)):
        if random.random() < mutation_rate:
            mutated_solution[i] = 1 - mutated_solution[i]  # Flip 0 to 1 and vice versa
    return mutated_solution

def genetic_algorithm(func,string_length,iterations,pop_size,mutation_rate,target):
    population = [generate_solution(string_length) for _ in range(pop_size)]
    best,best_eval=None, float('-inf')#Initialize best & best eval, similar to part 1A
    avg_fit=[]
    for gen in range(iterations):
        scores = [func(target,c) for c in population]
        avg_fitness=sum(scores)/len(scores)
        avg_fit.append(avg_fitness)
        for i in range(pop_size):
            if scores[i] > best_eval:
                best, best_eval = population[i], scores[i]
                print(">%d, new best f(%s)" % (gen, population[i]))

        if best == target:
            print(f"Target string {target} is reached.")
            break

        select=[selection(population,scores)for _ in range(pop_size)]
        newPop=[]
        for i in range(0,len(population),2):
            p1=select[i]
            p2=select[i+1]

            c1,c2=crossover(p1,p2)
            c1=mutation(c1,mutation_rate)
            c2=mutation(c2,mutation_rate)

            newPop.extend([c1,c2])
        population=newPop
    plt.plot(range(gen+1), avg_fit)
    plt.xlabel('Generations')
    plt.ylabel('Average Fitness')
    plt.title('Average Fitness of Population vs Generations')
    plt.show()
    return [best,best_eval]

string_length=30
iterations=100
pop=100
mutation_rate=.01
target_string=generate_solution(30)#generates random string


print('target string:',target_string)
best,score=genetic_algorithm(fitness,string_length,iterations,100,mutation_rate,target_string)

print('f(%s) ' % best)

if(best==target_string):#Prints match if 2 strings match
    print(target_string,'\n',best)
    print('match')

