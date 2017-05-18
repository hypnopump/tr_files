# Code written entirely by Eric Alcaide
# Contact: ericalcaide1@gmail.com
# KNN algorithm

# First, import the necessary libraries
from random import random, randint			# Used to generate pseudo-random numbers.													
import matplotlib.pyplot as plt 			# Used for graphs plotting														
import numpy as np 							# Used for maths operations		
# Used to reduce verbosity / simplify code															
from operator import add															
from functools import reduce 															

class KNN():
	# Number of Nearest Neighbors. Don't select a divisible by 2 number since we have just 2 different categories (A, B).
	k = 1	
	# A points will be 1, B points will be -1	
	classes = (-1, 1)
	# Defining colors for the graph -> not relevant for the algorithm																		
	colors = {-1: "r", 0: "y", 1: "b", 2: "c"}					
	# X and Y range will be between 0 and this number
	x_range, y_range = 1000, 1000
	# Number of Initial points for both sides.
	number = 7
	# Number of points to fill the graph			
	filled = 3000	
	# Defining later-used lists
	points = list()
	a_staff = list()
	b_staff = list()
	# Generating the initial labeled points
	for label in classes:
		for i in range(number):
			x = randint(0, x_range-1)	# +random()
			y = randint(0, y_range-1) 	# +random()
			points.append((x,y,label))
			a_staff.append((x,y,label)) if label == 1 else b_staff.append((x,y,label))
	# Generating the unclassified points
	unclassified = list()
	for i in range (filled):
		x = randint(0, x_range)
		y = randint(0, y_range)
		unclassified.append((x,y))
	a_points = list()
	b_points = list()

	def fillGraph(self):
		# Classify points
		for point in self.unclassified:
			# Calculate Euclidean distance and make tuples of (distance, point):
			neighbors = [ (np.sqrt((point[0] - p[0])**2 + (point[1] - p[1])**2), p ) for p in self.points ]
			# for p in self.points:
			# 	d = np.sqrt((point[0] - p[0])**2 + (point[1] - p[1])**2)
			# 	neighbors.append((d, p))
			# Sort the centers by distance to the point
			tupled = [ x[1][2] for x in sorted(neighbors)[:self.k] ]
			total = reduce(add, (x for x in tupled), 0)
			# Select the K points. Classify for A or B by majority vote.
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
		# Plot A and B points
		plt.plot([ a[0] for a in self.a_points], [ a[1] for a in self.a_points], 'y.')
		plt.plot([ b[0] for b in self.b_points], [ b[1] for b in self.b_points], 'c.')
		# Plot A and B Staff (Previous points, also called centers)
		plt.plot([ a[0] for a in self.a_staff], [ a[1] for a in self.a_staff], 'ro')
		plt.plot([ b[0] for b in self.b_staff], [ b[1] for b in self.b_staff], 'bo')
		# Define Axis labels and Graph title
		plt.title("Graph visualization")
		plt.xlabel("X coord")
		plt.ylabel("Y coord");
		# Show the graph
		plt.show()

def main():
	# Initialize the KNN class
	knn = KNN()
	# Call KNN important functions
	knn.fillGraph()
	knn.plotGraph()
	# lista = [9,3,5,1]
	# total = reduce(add, (str(x) for x in lista), "")
	# print(total)
	# print(knn.points)
	# print(knn.unclassified)
	# print(knn.fillGraph())

# Run the program
if __name__ == "__main__":
	main()