# Author: Eric Alcaide - First approach to evolutionary strategies (elitist selection + parents are part of the next generation)

# Import the necessary modules/libraries/dependencies
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt

# Create an individual here (random combination of vocab items with the target's length)
def individual(vocab, target):
	return [ str(vocab[randint(0, len(vocab)-1)]) for x in range(len(target)) ]

# Create the population here
def population(count, vocab, target):
	return [ individual(vocab, target) for x in range(count) ] 

# Measure the fitness of an individual. Lower is better. If fitness == 0 -> Goal Achieved!
def fitness(individual, target):
	score = np.array([0 if individual[i] == target[i] else 1 for i in range(len(target))])
	return np.sum(score, axis=0)

# Measure the fitness of an entire population. Lower is better.
def evaluate(population, target):
	evaluation = np.array([fitness(individual, target) for individual in population])
	total = np.sum(evaluation, axis=0)
	return total / len(population)

# Evolve individuals and create the next generation. Select the 20% best. Not the best approach
def evolve(population, target, count, retain, random_aditional, mutation, vocab):
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
			individual[pos_to_mutate] = vocab[randint(0, len(vocab)-1)]
	# Crossover of parents to generate children - Mimics biological reproduction
	parents_length = len(parents)
	children_maxlength = count - parents_length
	children = []
	while len(children) < children_maxlength:
		male = randint(0, parents_length-1)
		female = randint(0, parents_length-1)
		if male != female:											# Ensure that an individual is not male and female at the same time
			male = parents[male]
			female = parents[female]
			cross_point = randint(0, len(male)-1)
			child = male[:cross_point]+female[cross_point:] 		# Combine male and female at the selected cross_point
			children.append(child)
	parents.extend(children)										# Extend parents list by appending children list
	return parents 													# Return the next Generation of individuals

# Turn a list into a string
def stringify(iter):
	output = ""
	for i in iter:
		output = output+i
	return output

# Plot the Fitness of each generation. Lower is better
def plot_graph(eval_history):
	plt.figure()
	# Define plots.
	plt.plot(eval_history, '-ro')
	plt.grid()
	plt.title("Graph visualization")
	plt.xlabel("Number of Generations")
	plt.ylabel("Fitness score (number of differences between the population and the target)")
	# Plot it all
	plt.show()

# Sentence to achieve
target = "una ilusión, una sombra, una ficción, y el mayor bien es pequeño; que toda la vida es sueño, y los sueños, sueños son."	
vocab = " abcdefghijklmnopqrstuvwxyz!?.,;áóúíéñ"							# Vocabulary
count = 100 																# Individuals of a population
retain = 0.2 																# Percentage of the population that will survive
random_aditional = 0.05														# Random aditional survivals - genetic diversity
mutation = 0.15 															# Mutation rate - genetic diversity

print("Worst Score Possible:", len(target))

pop = population(count, vocab, target)
evaluation = evaluate(pop, target)
eval_history = [evaluation]
print(pop)

# While not target achieved, run the genetic algorithm
while stringify(pop[0]) != target:
	pop = evolve(pop, target, count, retain, random_aditional, mutation, vocab)
	evaluation = evaluate(pop, target)
	eval_history.append(evaluation)
	# Print the best individual
	print("Best:", stringify(pop[0]), "Fitness:", fitness(pop[0], target))

for i,log in enumerate(eval_history):
	print("Generation", i+1, ":", log)

print()
plot_graph(eval_history)
print("Proof:", stringify(pop[0]))