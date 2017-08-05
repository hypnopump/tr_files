# First attempts on the cifar10 database

import logging
import os
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import toimage

# For reproducibility
np.random.seed(6)

class NetworkParams():
	def __init__(self):
		self.nb_layers = 5
		self.nb_neurons = 512
		self.activation = keras.layers.advanced_activations.LeakyReLU(alpha=0.2)
		self.optimizer = "adamax"
		self.dropout = 0.25

class DataPrep():
	def __init__(self):
		self.classes = ["airplane", "automobile ", "bird ", "cat" , "deer", "dog", "frog" , "horse", "ship", "truck"]
		self.nb_classes = 10
		self.batch_size = 64
		self.input_dim = 32*32*3
		self.nb_train = 50000
		self.nb_test = 10000

		(self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()

	def plotting(self):
		cols, rows = 5, 5
		#Visualizing CIFAR 10
		fig, axes1 = plt.subplots(cols,rows,figsize=(5,5))
		for j in range(cols):
			for i in range(rows):
				axes1[j][i].set_axis_off()
				axes1[j][i].imshow(toimage(self.x_train[np.random.randint(self.nb_train)]))
		plt.show()

	def get_cifar10(self):
		"""Retrieve the CIFAR dataset and process the data."""
		# Get the data.
		self.x_train = self.x_train.reshape(self.nb_train, self.input_dim)
		self.x_test = self.x_test.reshape(self.nb_test, self.input_dim)
		self.x_train = self.x_train.astype('float32')
		self.x_test = self.x_test.astype('float32')
		self.x_train /= 255
		self.x_test /= 255

		# convert class vectors to binary class matrices
		self.y_train = np_utils.to_categorical(self.y_train, self.nb_classes)
		self.y_test = np_utils.to_categorical(self.y_test, self.nb_classes)
	
class Network():
	def __init__(self):
		self.data = DataPrep()
		params = NetworkParams()
		model = Sequential()
		# Add each layer.
		for i in range(params.nb_layers):
			# Need input shape for first layer.
			if i == 0:
				model.add(Dense(params.nb_neurons, activation=params.activation, input_dim = data.input_dim))
			else:
				model.add(Dense(params.nb_neurons, activation=params.activation))

			model.add(Dropout(params.dropout))

		# Output layer.
		model.add(Dense(self.data.nb_classes, activation='softmax'))
		model.compile(loss='categorical_crossentropy', optimizer=params.optimizer, metrics=['accuracy'])
		self.model = model

	def train(self):
		"""Train the model, return test accuracy."""
		# Helper: Early stopping.
		early_stopper = EarlyStopping(patience=2, verbose = 1)
		self.model.fit(data.x_train, data.y_train,
						batch_size=data.batch_size,
						epochs=10000,  # using early stopping, so no real limit
						verbose=1,
						validation_split=0.05,
						callbacks=[early_stopper])

		score = self.model.evaluate(data.x_test, data.y_test, verbose=1)

		return score[1]  # 1 is accuracy. 0 is loss.

if __name__ == "__main__":
	data = DataPrep()
	net = Network()
	params = NetworkParams()

	LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
	logging.basicConfig(filename=str(os.getcwd())+"/logging/first_log.txt",
						format = LOG_FORMAT,
						level = logging.DEBUG,
						filemode = "w")
	logger = logging.getLogger()
	logger.info("Saving the accuracy result of the first test")
	logger.info("Number of layers: "+str(params.nb_layers))
	logger.info("Number of neurons: "+str(params.nb_neurons))
	logger.info("Activation: "+str(params.activation))
	logger.info("Optimizer: "+str(params.optimizer))
	
	#the data, shuffled and split between train and test sets
	print("X_train original shape", data.x_train.shape)
	print("y_train original shape", data.y_train.shape)

	for label in data.y_train[0:5]:
		print(data.classes[label[0]], label)

	data.plotting()
	data.get_cifar10()

	acc = net.train()
	print(acc)
	logger.info("Acc: "+str(acc)+"%")