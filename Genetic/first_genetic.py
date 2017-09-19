# Author: Eric Alcaide - First approach to evolutionary strategies (elitist selection + parents are part of the next generation)

# Import the necessary modules
import numpy as np
import matplotlib.pyplot as plt
from random import random, randint

# Create an individual here
def individual(length, minimum, maximum):
	return [ randint(minimum, maximum) for x in range(length)]

# Create the population here
def population(count, length, minimum, maximum):
	return [ individual(length, minimum, maximum) for x in range(count) ]
	
# Measure the fitness of an individual. Lower is better.
def fitness(individual, target):
	total = np.sum(np.array(individual), axis = 0)
	return abs(target-total)

# Measure the fitness of an entire population. Lower is better.
def evaluate(population, target):
	evaluation = np.array([fitness(individual, target) for individual in population])
	total = np.sum(evaluation, axis=0)
	return total / (len(population) * 1.0)

# Evolve individuals and create the next generation. Select the 20% best. Not the best approach
def evolve(population, target, minimum, maximum, count, retain, random_aditional, mutation):
	tupled = [ (fitness(individual, target), individual) for individual in population ]
	tupled = [ x[1] for x in sorted(tupled) ]
	retain_length = int(retain*count)
	parents = tupled[:retain_length]
	# Select other individuals randomly to maintain genetic diversity. Could be avoided with a better approach.
	for individual in tupled[retain_length:]:
		if random_aditional > random():
			parents.append(individual)
	# Mutate some individuals to maintain genetic diversity
	for individual in parents:
		if mutation >= random():
			pos_to_mutate = randint(0, len(individual)-1)
			# individual[pos_to_mutate] = randint(minimum, maximum)
			individual[pos_to_mutate] = randint(min(individual), max(individual))
	# Crossover of parents to generate children
	parents_length = len(parents)
	children_maxlength = count - parents_length
	children = []
	while len(children) < children_maxlength:
		male = randint(0, parents_length-1)
		female = randint(0, parents_length-1)
		if male != female:
			male = parents[male]
			female = parents[female]
			cross_point = randint(0, len(male))
			# Combine male and female
			child = male[:cross_point]+female[cross_point:] 			
			children.append(child)
	parents.extend(children)									# Extend parents list by appending children list
	return parents 												# Return the next Generation of individuals

# Plot the Fitness of each generation. Lower is better
def plot_graph(eval_history):
	plt.figure()
	# Define plots.
	plt.plot(eval_history, '-ro')
	plt.grid(True)
	plt.title("Graph visualization")
	plt.xlabel("Number of Generations")
	plt.ylabel("Fitness score")
	# Plot it all
	plt.show()

target = 704				# Number to achieve
length = 6					# Max length of an individual
minimum = 0         		# Minimum value of each gen
maximum = int(target) 		# Maximum value of each gen
count = 100 				# Individuals of a population
retain = 0.2 				# Percentage of the population that will survive
random_aditional = 0.05		# Random survivals
mutation = 0.01 				# Mutation rate

pop = population(count, length, minimum, maximum)
evaluation = evaluate(pop, target)
eval_history = [evaluation]

while evaluation > 0:
	pop = evolve(pop, target, minimum, maximum, count, retain, random_aditional, mutation)
	evaluation = evaluate(pop, target)
	eval_history.append(evaluation)

for i,log in enumerate(eval_history):
	print("Generation", i+1, ":", log)
print()
print("Proof:", pop[0], pop[-1])

plot_graph(eval_history)