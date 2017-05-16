# **
# Code written entireley by Eric Alcaide
# https://github.com/EricAlcaide
# **

# Import necessary modules/libraries/dependencies
import sys
import csv
import numpy as np 
import matplotlib.pyplot as plt

# Extract the data from the csv (Excel-like document)
def extractData(filename):
	features = []
	with open(filename) as csvfile:
			reader = csv.reader(csvfile)
			for row in reader:
				x_coord = int(row[0])
				y_coord = int(row[1])
				label = int(row[2])
				features.append([x_coord, y_coord, label])

	return features

# Separate the extracted data into 3 lists (2 coordinates + classification +1/-1) - Required to plot the graph
def separate(data_list, which):
	features_x_coord = []
	features_y_coord = []
	features_labels = []
		
	for point in data_list:
		x_coord = point[0]
		y_coord = point[1]
		label = point[2]
		if which == 1 and label == 1:
			features_x_coord.append(x_coord)
			features_y_coord.append(y_coord)
			features_labels.append(label)
		elif which == -1 and label == -1:
			features_x_coord.append(x_coord)
			features_y_coord.append(y_coord)
			features_labels.append(label)

	return features_x_coord, features_y_coord, features_labels

# Get the result of the squashing function. Could be either a 0 or a 1.
def f_x(weight1, weight2, value1, value2, weight0):
	result = np.sign(weight1*value1 + weight2*value2 + weight0) 

	return result-1 if result==0 else result 		# returns 1 if x>0, -1 if x<=0

def perceptron(data_list):
	weight1 = 0 									# A when the linear equation is like Ax + By+ C = 0
	weight2 = 0 									# B when the linear equation is like Ax + By+ C = 0
	b = 0											# C when the linear equation is like Ax + By+ C = 0
	weight1_aux = 0 								# for later calculations
	weight2_aux = 0 								# for later calculations
	b_aux = 0 										# for later calculations
	counter = 0 									# auxiliar counter variable
	convergence_consideration = len(data_list) 		# number of iterations correct before considering convergence - length of the data list
	# Repeat until convergence
	while True:
		for point in data_list:
			# Adjust weights if predictions are incorrect
			if point[2] * f_x(weight1, weight2, point[0], point[1], b) <= 0:
				weight1 = weight1 + point[0]*point[2]
				weight2 = weight2 + point[1]*point[2]
				b = -weight1*point[0]-weight2*point[1]
			# Check convergence
			if weight1_aux == weight1 and weight2_aux == weight2 and b == b_aux:
				counter = counter+1
				if counter >= convergence_consideration: # Ensure all points are correctly classified
					# Defining the length of the separation line
					x = np.linspace(0, 20, 1000) 
					# Plot the +1 labeled points with blue color
					plt.plot(separate(data_list, 1)[0], separate(data_list, 1)[1], 'bo')
					# Plot the -1 labeled points with red color
					plt.plot(separate(data_list, -1)[0], separate(data_list, -1)[1], 'ro')
					# Plot the decision boundary
					b_changed = b-50
					plt.plot(x, -weight1*x/weight2-b_changed/weight2)
					# Plot extra information of the graph
					plt.title("Graph visualization")
					plt.xlabel("X coord")
					plt.ylabel("Y coord");
					# Show the graph
					plt.show()	
					# Terminates the program
					return
			else:
				# Reset to 0 every time a point is missclassified
				counter = 0
			# Update weights
			weight1_aux = weight1
			weight2_aux = weight2
			b_aux = b

# Enter the name of the input file - input1.csv and run the algorithm
input_file = str(sys.argv[1])
perceptron(extractData(input_file))