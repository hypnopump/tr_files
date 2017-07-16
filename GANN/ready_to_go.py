class Network():
	def __init__(self, nn_params = None):
		nn_params = {
			'Number of neurons': [32, 64, 128, 256, 768, 1024],
			'Number of layers': [1, 2, 3, 4, 5],
			'Activations': ['ReLU', 'sigmoid', 'tanh', 'Leaky ReLU', 'ELU', 'softmax'],
			'Optimizers': ['sdg', 'momentum', 'adagrad', 'adadelta', 'adam']
		}
		self.accuracy_test = 0
		self.accuracy_train = 0

class Population():
	def __init__(self):
		slef.retain = 0.2
		self.mutation = 0.05
		self.random_add = 0.05

class Genetic():
	# Create an individual here // n_layers + space for accuracy
	def individual(self, neurons, layers, activation, optimizer):
		return NN(neurons, layers, activation, optimizer)

