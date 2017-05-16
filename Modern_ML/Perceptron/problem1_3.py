# **
# Code written entireley by Eric Alcaide
# https://github.com/EricAlcaide
# **

# Import modules
import sys
import csv
import numpy as np
import matplotlib 
# import sklearn

# extract the data from the csv
def extractData(filename):
	features = []
	import csv
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
		
def perceptron(data_list, output_file):

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
				# Adjust weights if predictions are incorrect
				if point[2] * f_x(weight1, weight2, point[0], point[1], b) <= 0:
					weight1 = weight1 + point[0]*point[2]
					weight2 = weight2 + point[1]*point[2]
					b = -weight1*point[0]-weight2*point[1]
					# Write CSV file
					docuwriter = csv.writer(csvfile, delimiter=' ')
					without_comas = str(weight1)+','+str(weight2)+','+str(b-2)
					docuwriter.writerow([without_comas])
				# Check convergence
				if weight1_aux == weight1 and weight2_aux == weight2 and b == b_aux:
					counter = counter+1
					if counter >= convergence_consideration:
						# Write CSV file
						docuwriter = csv.writer(csvfile, delimiter=' ')
						without_comas = str(weight1)+','+str(weight2)+','+str(b-2)
						docuwriter.writerow([without_comas])
						return
				else:
					counter = 0

				weight1_aux = weight1
				weight2_aux = weight2
				b_aux = b

input_file = str(sys.argv[1])
output_file = str(sys.argv[2])
perceptron(extractData(input_file), output_file)