# Code written entireley by Eric Alcaide
# https://github.com/EricAlcaide

# Import modules
import numpy as np 
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

class Regression():
	input_file = "input2.csv"
	# Extract data from the csv file (excel-like file) and rescale it
	def extractData(self):
		data = pd.read_csv(self.input_file, names=['Age','Weight','Height'])
		# Separate the data by features
		age = data['Age'].values
		weight = data['Weight'].values
		height = data['Height'].values
		# Rescale the data (subtract its mean and divide by its standard deviation)
		age_rescaled = (age - age.mean()) / age.std()
		weight_rescaled = (weight - weight.mean()) / weight.std()
		
		return age_rescaled, weight_rescaled, height

	# Perform Linear Regression
	def run(self,features, it, alpha = 0.005):
		# Initialize weights of the hyperplane (aka Betas) to 0
		w = [0, 0, 0]
		# Will only restart when all points have been checked
		for i in range(it):	
			# Gradient sum for each w - updated later & risk & number of points
			gradient_sum = [0, 0, 0] 	
			risk = 0
			w_num = len(features[0])								
			# Check all points 
			for num in range(w_num):
				point = [features[0][num], features[1][num], features[2][num]]
				# Make the prediction w/ the regression function
				f_x = w[0] + w[1]*point[0] + w[2]*point[1] 		
				# Build the gradients for each feature/dimension
				gradient_sum[0] += (f_x - point[2])
				gradient_sum[1] += (f_x - point[2])*point[0]
				gradient_sum[2] += (f_x - point[2])*point[1]
				# Update risk w/ squared error
				risk += (f_x - point[2])**2

			# Update betas/weights with the (gradients * learning rate) & risk
			risk *= (1/(2*w_num)) 
			w[0] -= alpha*(1/w_num)*gradient_sum[0]
			w[1] -= alpha*(1/w_num)*gradient_sum[1]
			w[2] -= alpha*(1/w_num)*gradient_sum[2]
			
		# Output f(x) function and the regression hyperplane
		print("After", it, ": f(x;w) =", w[0], "+", w[1], "* x1 +", w[2], "* X2 || Risk:", risk)
		self.plotGraph(features, [w[0], w[1], w[2]])

	# Plot a graph representation of the data in 3D
	def plotGraph(self,features,plane):
		# Declaring some parameters
		mpl.rcParams['legend.fontsize'] = 10
		fig = plt.figure()
		ax = fig.gca(projection='3d')
		ax.plot(features[0], features[1], features[2], 'r.', label='Population Data') # Plot points
		ax.legend()

		if plane != False:
			# create x,y points to be part of the hyperplane
			xx, yy = np.meshgrid(np.linspace(features[0].min()-1, features[0].max()+1), np.linspace(features[1].min()-1, features[1].max()+1))
			# calculate corresponding z for each x,y pair
			z = plane[0] + plane[1]*xx + plane[2]*yy
			# plot the surface
			ax.plot_surface(xx, yy, z, cmap=cm.coolwarm)

		plt.show()

	# Run the algorithm and plot the graphs
	def justDoIt(self):
		features = self.extractData()
		regression.plotGraph(features, plane = False) 	# First plot the points to visualize data
		its = [200, 250, 500, 1000, 1500]						# Try linear regression with each number of iterations here - see how does the hyperplane change
		for i in its:
			regression.run(features, i)

if __name__ == "__main__":
	regression = Regression()
	regression.justDoIt()