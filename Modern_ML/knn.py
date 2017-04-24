# Code written entirely by Eric Alcaide
# Contact: ericalcaide1@gmail.com
# KNN algorithm

# First, import the necessary libraries
from random import random, randint
import matplotlib.pyplot as plt 
import numpy as np
from operator import add
from functools import reduce

class KNN():
	k = 1 					# Number of Nearest Neighbors. Don't select a divisible by 2 number
	colors = {-1: "r", 0: "y", 1: "b", 2: "c"}	# A points will be 1, B points will be -1
	classes = (-1, 1)
	x_range = 1000			# X range will be between 0 and this number
	y_range = 1000			# Y range will be between 0 and this number
	number = 15				# Number of Initial points for both sides.
	filled = 3000			# Number of points to fill the graph
	points = []
	a_staff = []
	b_staff = []
	for label in classes:
		for i in range(int(number/2)):
			x = randint(0, x_range-1)	# +random()
			y = randint(0, y_range-1) 	# +random()
			points.append((x,y,label))
			a_staff.append((x,y,label)) if label == 1 else b_staff.append((x,y,label))

	unclassified = []
	for i in range (filled):
		x = randint(0, x_range)
		y = randint(0, y_range)
		unclassified.append((x,y))
	a_points = []
	b_points = []

	def fillGraph(self):
		# Classify points
		for point in self.unclassified:
			# Calculate Euclidean distance and make tuples of (distance, point):
			neighbors = [ (np.sqrt((point[0] - p[0])**2 + (point[1] - p[1])**2), p ) for p in self.points ]
			# for p in self.points:
			# 	d = np.sqrt((point[0] - p[0])**2 + (point[1] - p[1])**2)
			# 	neighbors.append((d, p))
			tupled = [ x[1][2] for x in sorted(neighbors)[:self.k] ]
			total = reduce(add, (x for x in tupled), 0)

			self.a_points.append(point) if total > 0 else self.b_points.append(point)
			# if total > 0:
			# 	a_points.append(point)
			# else:
			# 	b_points.append(point)
			# break

		print(tupled)
		print(total)
		print(self.a_points)
		print(self.b_points)




	def plotGraph(self):
		# Plot A points
		plt.plot([ a[0] for a in self.a_points], [ a[1] for a in self.a_points], 'y.')
		# Plot B points
		plt.plot([ b[0] for b in self.b_points], [ b[1] for b in self.b_points], 'c.')
		# Plot A Staff
		plt.plot([ a[0] for a in self.a_staff], [ a[1] for a in self.a_staff], 'ro')
		# Plot B Staff
		plt.plot([ b[0] for b in self.b_staff], [ b[1] for b in self.b_staff], 'bo')

		plt.title("Graph visualization")
		plt.xlabel("X coord")
		plt.ylabel("Y coord");

		plt.show()

def main():
	knn = KNN()

	knn.fillGraph()
	knn.plotGraph()
	lista = [9,3,5,1]
	total = reduce(add, (str(x) for x in lista), "")
	print(total)

	# print(knn.points)
	# print(knn.unclassified)
	# print(knn.fillGraph())

if __name__ == "__main__":
	main()