# -*- coding: utf-8 -*-

"""
	Code written entirely by Eric Alcaide: https://github.com/EricAlcaide

	Mimic biological reproduction in a darwinist environment.
	Probabilistic reproduction strategy: each individual has its
	acc_test probability of passing to the next generation.
	Attempt to create self-evolving, learning-based Neural Nets.

	ConvNets would perform much better, but it's not about accuracy.
"""

from functools import reduce
import keras
from keras.callbacks import EarlyStopping
from keras.datasets import cifar10
from keras.models import Sequential
from keras.optimizers import Optimizer
from keras.utils import np_utils
from keras.layers.core import Dense, Dropout, Activation
import logging
import numpy as np
from operator import add
import os
import random

# For reproducibility
np.random.seed(0)
# Record settings
LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
logging.basicConfig(filename=str(os.getcwd())+"/logging/probabilistic_log.txt",
					format = LOG_FORMAT,
					level = logging.DEBUG,
					filemode = "a")
logger = logging.getLogger()


class NetworkParams():
	"""NN params object"""
	def __init__(self, nb_layers=None, nb_neurons=None, activation=None, optimizer=None, dropout=None, params = None):
		if not params:
			self.params = { "nb_layers": nb_layers,
							"nb_neurons": nb_neurons,
							"activation": activation,
							"optimizer": optimizer,
							"dropout": dropout }
		else: self.params = params
		self.acc_test = None


# Data preparation and verification
class DataPrep():
	"""Retrieve the CIFAR dataset and process the data."""
	def __init__(self):
		self.classes = ["airplane", "automobile ", "bird ", "cat" , "deer", "dog", "frog" , "horse", "ship", "truck"]
		self.nb_classes = 10
		self.batch_size = 64
		self.input_dim = 32*32*3
		self.nb_train = 50000
		self.nb_test = 10000
		# Get the data.
		(self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()
		# Preprocess it
		self.x_train = self.x_train.reshape(self.nb_train, self.input_dim)
		self.x_test = self.x_test.reshape(self.nb_test, self.input_dim)
		self.x_train = self.x_train.astype('float32')
		self.x_test = self.x_test.astype('float32')
		self.x_train /= 255
		self.x_test /= 255
		# convert class vectors to binary class matrices
		self.y_train = np_utils.to_categorical(self.y_train, self.nb_classes)
		self.y_test = np_utils.to_categorical(self.y_test, self.nb_classes)
		# Log step
		logger.info("Cifar10 data retrieved & Processed")

 
class Network():
	"""Create the network from params set and train it"""
	def __init__(self, data, net):
		self.data = data
		params = net.params
		model = Sequential()
		# Add each layer.
		for i in range(params['nb_layers']):
			# Need input dim for first layer.
			if i == 0:
				model.add(Dense(params['nb_neurons'], activation=params['activation'], input_dim = data.input_dim))
			else:
				model.add(Dense(params['nb_neurons'], activation=params['activation']))

			model.add(Dropout(params['dropout']))

		# Output layer.
		model.add(Dense(self.data.nb_classes, activation='softmax'))
		model.compile(loss='categorical_crossentropy', optimizer=params['optimizer'], metrics=['accuracy'])
		self.model = model

	def train(self):
		"""Train the model, return test accuracy. Early stopping."""
		early_stop = EarlyStopping(patience=5, verbose = 1)
		self.model.fit(self.data.x_train, self.data.y_train,
						batch_size=self.data.batch_size,
						epochs=10000,  # using early stopping, so no real limit
						verbose=1,
						validation_data=(self.data.x_test, self.data.y_test),
						callbacks=[early_stop])

		score = self.model.evaluate(self.data.x_test, self.data.y_test, verbose=1)

		return score[1]  # return accuracy. [0] is loss.


class Genetic():
	""" Probability-based evolution. Each Individual has its 
		test_acc probability of passing to the next generation."""
	def __init__(self):
		self.data = DataPrep()

		self.n_nets = 15
		self.n_iter = 15
		self.mutation = 0.05
		self.eval_history = [0]	

		self.nb_layers = [1,2,3,4,5]
		self.nb_neurons = [32,64,128,256,512,768,1024]
		self.activation = ['tanh', 'sigmoid', 'relu', 'elu', 'selu', ]
		self.optimizer = ['sgd', 'adamax', 'rmsprop', 'adagrad', 'adadelta']
		self.dropout = [0.15, 0.2, 0.25]
	
		self.params = { "nb_layers": self.nb_layers,
						"nb_neurons": self.nb_neurons,
						"activation": self.activation,
						"optimizer": self.optimizer,
						"dropout": self.dropout }

	def generation(self):
		"""Create the population here"""
		return [ NetworkParams(
					self.nb_layers[np.random.randint(len(self.nb_layers))],
					self.nb_neurons[np.random.randint(len(self.nb_neurons))],
					self.activation[np.random.randint(len(self.activation))],
					self.optimizer[np.random.randint(len(self.optimizer))],
					self.dropout[np.random.randint(len(self.dropout))]
				) for n in range(self.n_nets) ]

	def evaluate(self, population):
		"""Measure the fitness of an entire population. Lower is better."""
		tests = [nn.acc_test*100 for nn in population]
		total = reduce(add, tests, 0)
		return total / float(self.n_nets)

	
	def breed(self, male, female):
		"""Generate a child. Randomly replace all genes for parents ones"""	
		child = NetworkParams(None, None, None, None, None)
		for key in child.params:
			child.params[key] = random.choice(
				[male.params[key], female.params[key]])

		return child

	def evolve(self, population, parents = None):
		"""Evolve individuals and create the next generation. Probabilistic Reproduction."""
		if not parents: 
			self.train(population)
			# Select the best individuals
			parents = [nn for nn in population if nn.acc_test > np.random.random()]
		# Record the selected parents
		logger.info("Parents: {0}".format(str([nn.params for nn in parents])))
		logger.info("*************************************************")
		# Crossover of parents to generate children
		parents_length = len(parents)
		children_maxlength = self.n_nets - parents_length
		children = []
		while len(children) < children_maxlength:
			male = np.random.randint(0, parents_length)
			female = np.random.randint(0, parents_length)
			if male != female:
				# Combine male and female
				child = self.breed(parents[male], parents[female])			
				children.append(child)
		# Extend parents list by appending children list
		parents.extend(children)
		# Mutate some individuals to maintain genetic diversity
		for nn in parents:
			for key in self.params:
				if self.mutation > np.random.random():
					nn.params[key] = random.choice(self.params[key])	
		# Return the next Generation of individuals
		return parents

	def train(self, population):
		"""Train the NNs in the population"""
		for nn in population:
			logger.info("Number of layers: {0}".format(nn.params['nb_layers']))
			logger.info("Number of neurons: {0}".format(nn.params['nb_neurons']))
			logger.info("Activation: {0}".format(nn.params['activation']))
			logger.info("Optimizer: {0}".format(nn.params['optimizer']))
			logger.info("Dropout: {0}".format(nn.params['dropout']))
			# Train Network
			nn.acc_test = Network(self.data, nn).train()
			# Log the result
			logger.info("Acc @ Testing: {0}%".format(nn.acc_test*100))
			logger.info("--------------------------------------------------------")
			# Free space
			self.free_gpu_mem()
		# Get population average and log it
		avg_acc = self.evaluate(population)
		self.eval_history.append(avg_acc)
		logger.info("Generations Average: {0}%".format(self.eval_history))
		logger.info("--------------------------------------------------------")

	def free_gpu_mem(self):
		"""Free gpu space"""
		keras.backend.get_session().close()
		keras.backend.set_session(keras.backend.tf.Session())


if __name__ == "__main__":
	gen = Genetic()
	# Run the Genetic Algorithm and let Nets Evolve
	parents = [{'nb_neurons': 64, 'dropout': 0.15, 'activation': 'elu', 'nb_layers': 4, 'optimizer': 'adamax'},
						{'nb_neurons': 64, 'dropout': 0.2, 'activation': 'elu', 'nb_layers': 1, 'optimizer': 'adamax'},
						{'nb_neurons': 768, 'dropout': 0.2, 'activation': 'elu', 'nb_layers': 1, 'optimizer': 'adamax'},
						{'nb_neurons': 64, 'dropout': 0.2, 'activation': 'sigmoid', 'nb_layers': 1, 'optimizer': 'adamax'},
						{'nb_neurons': 64, 'dropout': 0.2, 'activation': 'sigmoid', 'nb_layers': 2, 'optimizer': 'adamax'},
						{'nb_neurons': 128, 'dropout': 0.2, 'activation': 'sigmoid', 'nb_layers': 2, 'optimizer': 'adamax'},
						{'nb_neurons': 128, 'dropout': 0.2, 'activation': 'elu', 'nb_layers': 2, 'optimizer': 'adadelta'},
						{'nb_neurons': 128, 'dropout': 0.2, 'activation': 'elu', 'nb_layers': 4, 'optimizer': 'rmsprop'}]
	pop = gen.evolve(None,parents = [NetworkParams(params = params) for params in parents])

	logger.info("Number of layers: {0}".format(gen.params['nb_layers']))
	logger.info("Number of neurons: {0}".format(gen.params['nb_neurons']))
	logger.info("Activation: {0}".format(gen.params['activation']))
	logger.info("Optimizer: {0}".format(gen.params['optimizer']))
	logger.info("Dropout: {0}".format(gen.params['dropout']))
	logger.info("/////////////////////////////////////////////////////")
	logger.info("Saving the accuracy result of the training")

	for i in range(10,gen.n_iter):
		logger.info("------ GEN {0} ------".format(i+1))
		pop = gen.evolve(pop)

	# Record final information
	logger.info("/////////////////////////////////////////////////////")
	logger.info(str(gen.eval_history))