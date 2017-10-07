# -*- coding: utf-8 -*-
"""
	Code written entirely by Eric Alcaide.

	Find the parameters of the linear decision boundary that 
	separates two classes of given data.
"""
# Import dependencies
import numpy as np 
import matplotlib.pyplot as plt

def plotGraph(data, weights = np.array(None)):
	# Plot line if weights are passed
	if weights.any():
		x = np.linspace(0, 20, 1000)
		plt.plot(x, -weights[0]*x/weights[1]-weights[-1]/weights[1])
	# Plot points and graph
	colors = {1:'bo', -1:'ro'}
	for p in data:
		plt.plot(p[0], p[1], colors[p[2]])
	plt.title("Graph visualization")
	plt.show()	

def extractData(filename):
	""" Extract the data from the csv (Excel-like document)."""
	return np.genfromtxt(filename,delimiter=',').astype(int)

def f_x(w, inputs):
	"""Get the result of the squashing function. Can be a -1, 0 or 1.""" 
	return np.sign(w[-1] + np.dot(inputs, w[:-1]))

def perceptron(data):
	w = np.zeros(3)	
	# Repeat until convergence
	while True:
		convergence = True
		for p in data:
			# Adjust weights if predictions are incorrect
			if p[2] * f_x(w, p[:-1]) <= 0:
				w[0] += p[2] * p[0]
				w[1] += p[2] * p[1]
				w[-1] = -w[0]*p[0]-w[1]*p[1]-1
				convergence = False
		if convergence:
			print("Decision Boundary Line: {0}x+{1}y+{2} = 0".format(*w))
			plotGraph(data, w)
			return

data = extractData("toy_data.csv")
perceptron(data)