# -*- coding: utf-8 -*-
"""
	Code written entirely by Eric Alcaide.

	Find K clusters given a set of points.
	Points tend to be normally distributed.
"""
# Import dependecies
import numpy as np
import matplotlib.pyplot as plt 

def extract(filename):
	""" Extract data from the txt file."""
	return np.loadtxt(filename)

def distance(a,b):
	""" Return the Euclidean distance between two points."""
	return np.linalg.norm(a-b)

def kmeans(k, epsilon):
	"""KMEANS algorithm // K = number of clusters // epsilon = max iterations"""
	# Extract data
	data = extract("toy_data.txt")
	# Get the number of rows (instances) and columns (features) from the dataset
	num_instances, num_features = data.shape
	# Define k centroids (n clusters we want to find) chosen randomly (K too high -> risk of repeating a starting point)
	centroids = data[np.random.randint(0, num_instances - 1, size=k)]
	# Array to save the cluster a point pertains to
	belongs_to = np.zeros((num_instances, 1))
	# Iteratively update the clusters assignments and the centroids
	for i in range(epsilon):
		for index_p, p in enumerate(data):
			# define a distance vector of size k (distance to each centroid)
			dist_vec = np.zeros((k,1))
			# for each centroid
			for index_c, c in enumerate(centroids):
				# compute the distance between p and centroid
				dist_vec[index_c] = distance(p,c)
			# find the closest centroid, assign the point with that cluster
			belongs_to[index_p, 0] = np.argmin(dist_vec)

		for index in range(len(centroids)):
			# get all the points assigned to a cluster
			points_close = [j for j in range(len(belongs_to)) if belongs_to[j] == index]
			# find the mean of those points, this is our new centroid
			centroids[index] = np.mean(data[points_close], axis=0)
	# Return position of centroids, data and cluster assignment for each point
	return centroids, data, belongs_to


def justDoIt():
	""" Run the algorithm."""
	post = kmeans(2, 15)
	# Plot graph
	for i, c in enumerate(post[0]):
		plt.plot([p[0] for j,p in enumerate(post[1]) if post[2][j] == i], [p[1] for j,p in enumerate(post[1]) if post[2][j] == i], '.')

	plt.plot([c[0] for c in post[0]], [c[1] for c in post[0]], 'o')
	plt.title("KMeans algorithm with K = {0}".format(len(post[0])))
	plt.show()

if __name__ == "__main__":
	justDoIt()