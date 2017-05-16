# **
# Code written entireley by Eric Alcaide
# https://github.com/EricAlcaide
# **

# Import modules
import sys
import csv
import numpy as np 
import matplotlib.pyplot as plt
# import sklearn

# extract the data from the csv
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

# get the f(x) result
def f_x(weight1, weight2, value1, value2, weight0):
	result = np.sign(weight1*value1 + weight2*value2 + weight0) # returns 1 if x>0, 0 if x==0, -1 if x<0

	if result == 0:
		result = -1

	return result

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

def perceptron(data_list, output_file):
	# print(data_list)
	x = np.linspace(0, 20, 1000)
	# print("-------------")
	weight1 = 0 # weight of x axis
	weight2 = 0 # weight of y axis
	b = 0		# idk wtf is this fuckin' value
	weight1_aux = 0 # for later calculations
	weight2_aux = 0 # for later calculations
	b_aux = 0 # for later calculations
	counter = 0 # auxiliar variable
	convergence_consideration = 25 # number of iterations correct before considering convergence
	with open(output_file, 'w', newline='') as csvfile:
		while True:
			for point in data_list:
				# print("POINT:", point[0], ',', point[1])
				# print("Labeled as:", point[2])
				# print("Predicted as:", f_x(weight1, weight2, point[0], point[1], b))
				# print()
				# # Adjust weights if predictions are incorrect
				if point[2] * f_x(weight1, weight2, point[0], point[1], b) <= 0:
					weight1 = weight1 + point[0]*point[2]
					weight2 = weight2 + point[1]*point[2]
					b = -weight1*point[0]-weight2*point[1]

				# Print results
				# print("WEIGHT 1:", weight1)
				# print("WEIGHT 2:", weight2)
				# print("B:", b)
				# print("-----------------------------------")
				# Write CSV file
					docuwriter = csv.writer(csvfile, delimiter=' ')
					docuwriter.writerow(str(weight1)+','+str(weight2)+','+str(b-2))
				# Check convergence
				if weight1_aux == weight1 and weight2_aux == weight2 and b == b_aux:
					counter = counter+1
					if counter >= convergence_consideration:
						docuwriter = csv.writer(csvfile, delimiter=' ')
						docuwriter.writerow(str(weight1)+','+str(weight2)+','+str(b-2))
						# Plot the +1 labeled points
						plt.plot(separate(data_list, 1)[0], separate(data_list, 1)[1], 'bo')
						# # Plot the -1 labeled points
						plt.plot(separate(data_list, -1)[0], separate(data_list, -1)[1], 'ro')
						# # Plot the decision boundary
						b_changed = b-50
						plt.plot(x, -weight1*x/weight2-b_changed/weight2)
						plt.title("Graph visualization")
						plt.xlabel("X coord")
						plt.ylabel("Y coord");

						plt.show()	
						return
				else:
					counter = 0

				weight1_aux = weight1
				weight2_aux = weight2
				b_aux = b

input_file = str(sys.argv[1])
# print(input_file)
output_file = str(sys.argv[2])
# print(output_file)
perceptron(extractData(input_file), output_file)