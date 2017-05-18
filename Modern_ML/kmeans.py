# Code written entirely by Eric Alcaide
# K-means algorithm for clustering in 2-D

from random import random,randint 			# Generate pseudo-random numbers
import numpy as np 							# Maths library
import matplotlib.pyplot as plt 			#Required for graphs plotting

class KMEANS():
	k = 2				# Number of clusters - less than 7
	num = (5,15)		# Range for number of points per cluster
	lims = (4,7)		# Range for max distance between point and center of the cluster
	board = (15,15)		# Board of points
	it = 5 	 			# Number of iterations of the algorithm
	points = list() 	# Set of points
	parents = list() 	# List of parents
	centers = list() 	# List of centers
	exchange_dict = {0: "r", 1: "b", 2: "g", 3: "y", 4: "m", 5: "c"}

	# Create the initial data
	def createData(self):
		# First create the parents of the clusters
		self.parents = self.createCenters()
		# Create the points (5-15) for each cluster (within less than 3-5 euclidean distance from the center to ensure Gaussian distribution)
		for p in self.parents:
			count = 0										# Counter of points
			stop = randint(self.num[0], self.num[1]) 		# Number of points in the cluster
			lim = randint(self.lims[0],self.lims[1]) 		# Limit distance
			while count < stop:
				x = randint(p[0]-lim,p[0]+lim)				# X coord of the new point
				y = randint(p[1]-lim,p[1]+lim) 				# Y coord of the new point
				d = np.sqrt(((x-p[0])**2)+((y-p[1])**2))	# Ensure Euclidean distance is less or equal than limit
				if d <= lim: 								# Add a new point to the set
					self.points.append([x,y,0])				# The 0 will be the label of the cluster later
					count = count+1

		self.plotGraph()

	# Initialize centers to random points
	def createCenters(self):
		centers = list()
		for i in range(self.k):
			p = [randint(0,self.board[0]), randint(0,self.board[1])]
			centers.append(p)
		return centers

	# Evolve the N iterations (set before)
	def evolve(self):
		self.centers = self.createCenters()
		self.plotGraph()

		for i in range(self.it):
			for p in self.points: 			# Assign each point to a cluster
				dist = list()
				for c in self.centers:
					d = np.sqrt(((c[0]-p[0])**2)+((c[1]-p[1])**2))
					dist.append((d,c))
				p[2] = self.centers.index(min(dist)[1])
			print(self.points)

			for c in self.centers:			# Move each center-cluster to the average of the points assigned to the cluster
				k = self.centers.index(c)
				total_x = 0
				total_y = 0
				counter = 0
				for p in self.points:
					if p[2] == k:
						total_x = total_x+p[0]
						total_y = total_y+p[1]
						counter = counter+1
				if counter != 0:
					self.centers[k] = [total_x/counter, total_y/counter]
			print(self.centers)
			
			self.plotGraph()

	def plotGraph(self):
		for p in self.points:
			plt.plot(p[0], p[1], self.exchange_dict[p[2]]+".")
		for c in self.centers:
			plt.plot(c[0], c[1], self.exchange_dict[self.centers.index(c)]+"o")
		plt.xlabel("X coord")
		plt.ylabel("Y coord")
		plt.axis()

		plt.show()

def main():
	kmeans = KMEANS()
	kmeans.createData()
	print(kmeans.parents)
	print(kmeans.points)
	print(len(kmeans.points))
	print(kmeans.evolve())

if __name__ == "__main__":   				# Run the algorithm
	main()