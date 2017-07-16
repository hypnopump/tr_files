# Scond Proof of concept of GA/NN
# Just tunning the number of neurons

# Import the necessary modules
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import pchip
import tensorflow as tf
from operator import add
from functools import reduce
from scipy.interpolate import pchip

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

class NN():
	# Create a Neural Net out of nothing - Spontanious Generation! Yuhuu!
	def __init__(self, nn_params):
		self.nn_params_choices = {}
		for key in nn_params:
			self.nn_params_choices[key] = np.random.choice(nn_params[key])

		self.acc_train, self.acc_test, self.cre_train, self.cre_test = None, None, None, None

class Genetic():
	# Declare parameters
	def __init__(self):
		self.nn_params = {
			'layer_1': [i for i in range(1, 100)], 
			'layer_2': [i for i in range(1, 100)]
		}
		self.n_layers = 2
		self.n_nets = 10
		self.n_iter = 10
		self.retain = 0.2
		self.mutation = 0.1
		self.random_add = 0.1
		self.eval_history = [0]	

	# Create the population here
	def population(self):
		return [ NN(self.nn_params) for n in range(self.n_nets) ]

	# Measure the fitness of an entire population. Lower is better.
	def evaluate(self, population):
		tests = [nn.acc_test for nn in population]
		total = reduce(add, tests, 0)
		return total / float(self.n_nets)

	# Generate a child and then randomly replace all genes for the ones coming either from male or female	
	def breed(self, male, female):
		child = NN(self.nn_params)
		for key in self.nn_params:
			child.nn_params_choices[key] = random.choice(
				[male.nn_params_choices[key], female.nn_params_choices[key]]
			)
		return child

	# Evolve individuals and create the next generation. Select the 20% best +  random %. Elitist Reproduction.
	def evolve(self, population):
		self.train(population)

		tupled = [ nn for nn in sorted(population, key=lambda x: x.acc_test, reverse=True)]
		retain_length = int(self.retain*self.n_nets)

		parents = tupled[:retain_length]
		# Select other individuals randomly to maintain genetic diversity.
		for nn in tupled[retain_length:]:
			if self.random_add > random.random():
				parents.append(nn)

		print("Parents:", [nn.nn_params_choices for nn in parents])
		# Mutate some individuals to maintain genetic diversity
		for nn in parents:
			for key in self.nn_params:
				if self.mutation > random.random():
					nn.nn_params_choices[key] = random.choice(self.nn_params[key])
		# Crossover of parents to generate children
		parents_length = len(parents)
		children_maxlength = self.n_nets - parents_length
		children = []
		while len(children) < children_maxlength:
			male = random.randint(0, parents_length-1)
			female = random.randint(0, parents_length-1)
			if male != female:
				child = self.breed(parents[male], parents[female])			# Combine male and female
				children.append(child)
		parents.extend(children)									# Extend parents list by appending children list

		return parents 												# Return the next Generation of individuals

	# Train every single NN in the population
	def train(self, population):
		for nn in population:
			self.neural(nn)
		self.eval_history.append(self.evaluate(population))

	# Get the accuracy for a given neural net
	def neural(self, net):
	    
		batch_size = 100
		K = net.nn_params_choices['layer_1']
		L = net.nn_params_choices['layer_2']

		W1 = tf.Variable(tf.truncated_normal([28*28, K], stddev = 0.1))
		B1 = tf.Variable(tf.zeros([K]))
		W2 = tf.Variable(tf.truncated_normal([K, L], stddev = 0.1))
		B2 = tf.Variable(tf.zeros([L]))
		W3 = tf.Variable(tf.truncated_normal([L, 10], stddev = 0.1))
		B3 = tf.Variable(tf.zeros([10]))

		X = tf.placeholder(tf.float32, [None, 28 * 28]) 	# One layer because 28*28 gray-scaled images, the None will become the batch size
		X = tf.reshape(X, [-1, 28*28])

	    # Defining the model - changing relu by my_function
		Y1 = tf.nn.relu(tf.matmul(X, W1) + B1)
		Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
		Y3 = tf.nn.softmax(tf.matmul(Y2, W3) + B3)

		init = tf.global_variables_initializer()

		# Defining the placeholder for correct answers
		Y_ = tf.placeholder(tf.float32, [None, 10]) 		# "One-hot" encoded vector [00001000000]

		# SUCCESS metrics
		# Loss function to determine how bad is the model
		cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y3)) 	# If Y is 1, log(Y) = 0, if Y is 0, log(Y) = -infinite
		# % of correct answers found in batch
		is_correct = tf.equal(tf.argmax(Y3, 1), tf.argmax(Y_, 1))	# "One-hot" decoding here
		accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

		# Training step
		optimizer = tf.train.GradientDescentOptimizer(0.003)
		train_step = optimizer.minimize(cross_entropy)

		# RUN the fuckin' code here
		sess = tf.Session() 								# Code in tf is not computed until RUN
		sess.run(init)
		# Training loop
		for i in range(1000):
			# Load batch of images and correct answers
			batch_X, batch_Y = mnist.train.next_batch(batch_size)	# Train on mini_batches of 100 images
			train_data = {X: batch_X, Y_: batch_Y}
			# Train
			sess.run(train_step, feed_dict = train_data)
			# {X: batch_X, Y_: batch_Y}
		
		# Succes on training data
		a_tr, c_tr = sess.run([accuracy, cross_entropy], feed_dict = train_data)
		# Success on test data?
		test_data = {X: mnist.test.images, Y_: mnist.test.labels}
		a_test, c_test = sess.run([accuracy, cross_entropy], feed_dict = test_data)
	    
		sess.close() # Closing the session
	    
		# Updating values
		net.acc_train, net.cre_train = a_tr, c_tr
		net.acc_test, net.cre_test = a_test, c_test
		print(net.nn_params_choices,": acc_train:", a_tr, "|| acc_test:", a_test)

		return

if __name__ == "__main__":
	gen = Genetic()
	# proof = [(nn.nn_params_choices) for nn in gen.population()]
	# print(proof)
	# print(gen.population())

	# for i in range(5):
	# 	print(gen.neural(NN(gen.nn_params)))

	# Plot the Fitness of each generation. Lower is better
	def plot_graph(eval_history):
		# Data to be interpolated.
		x = [g for g in range(len(gen.eval_history))]
		y = gen.eval_history
		# Create the interpolator.
		interp = pchip(x, y)
		# Dense x for the smooth curve. 2nd param = x limit. 3rd param = density 
		xx = np.linspace(0, len(gen.eval_history), gen.n_iter*10)
		# Define plots.
		plt.plot(xx, interp(xx))
		# Define plots.
		plt.plot(x, y, 'r.')
		plt.grid(True)
		plt.title("Graph visualization")
		plt.xlabel("Generations")
		plt.ylabel("Average Accuracy for each Generation")
		# Plot it all
		plt.show()

	# Run the Evolutionary Algorithm
	pop = gen.population()

	for i in range(gen.n_iter):
		print("------ GEN "+str(i+1)+" ------")
		pop = gen.evolve(pop)

	print("------------------------------")
	print("------------------------------")
	print(gen.eval_history)

	plot_graph(gen.eval_history)