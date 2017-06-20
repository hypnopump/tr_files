# Author: Eric Alcaide - First approach to evolutionary strategies (elitist selection + parents are part of the next generation)

# Import the necessary modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import pchip
from random import random, randint
from operator import add
from functools import reduce

# Create an individual here
def individual(length, minimum, maximum):
	return [ randint(minimum, maximum) for x in range(length)]

# Create the population here
def population(count, length, minimum, maximum):
	return [ individual(length, minimum, maximum) for x in range(count) ]
	
# Measure the fitness of an individual. Lower is better.
def fitness(individual, target):
	total = reduce(add, individual, 0) # Use this line for sum
	return abs(target-total)

# Measure the fitness of an entire population. Lower is better.
def evaluate(population, target):
	print(population)
	total = reduce(add, (fitness(individual, target) for individual in population), 0)
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
	for individuaL in parents:
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
			# cross_point = int(len(male)/2)
			cross_point = randint(0, len(male))
			child = male[:cross_point]+female[cross_point:] 			# Combine male and female
			children.append(child)
	parents.extend(children)											# Extend parents list by appending children list
	return parents 												# Return the next Generation of individuals

# Plot the Fitness of each generation. Lower is better
def plot_graph(eval_history):
	# Data to be interpolated.
	generations = len(eval_history)
	y = eval_history
	x = [ z for z in range(generations) ]
	print(y)
	# Create the interpolator.
	interp = pchip(x, y)
	# Dense x for the smooth curve. 2nd param = x limit. 3rd param = density 
	xx = np.linspace(0, generations-1, generations*10)
	# Define plots.
	plt.plot(xx, interp(xx))
	plt.plot(x, y, 'r.')
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
mutation = 0.05 				# Mutation rate

pop = population(count, length, minimum, maximum)
evaluation = evaluate(pop, target)
eval_history = [evaluation]

while evaluation > 0:
	pop = evolve(pop, target, minimum, maximum, count, retain, random_aditional, mutation)
	evaluation = evaluate(pop, target)
	eval_history.append(evaluation)

counter = 1
for log in eval_history:
	print("Generation", counter, ":", log)
	counter = counter + 1
print()
print("Proof:", pop[0], pop[-1])

plot_graph(eval_history)