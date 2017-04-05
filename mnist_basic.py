# Code extracted from the first neural network of this tutorial:
# https://www.youtube.com/watch?v=vq2nnJ4g6N0&list=WL&index=4&t=464s
# First single layer NN from scratch - 92% accuracy

# Import Tensorflow library and mnist data
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# Define the batch size
batch_size = 100
# Define variables for the equation Y = WX +b
X = tf.placeholder(tf.float32, [None, 28 * 28]) 	# One layer because 28*28 gray-scaled images, the None will become the batch size
W = tf.Variable(tf.zeros([784, 10])) 				# Initiaize the weights to 0
b = tf.Variable(tf.zeros([10]))						# Initialize biases to 0
# Defining the model
Y = tf.nn.softmax(tf.matmul(tf.reshape(X, [-1, 784]), W) + b) # Flattering images
# Defining the placeholder for correct answers
Y_ = tf.placeholder(tf.float32, [None, 10]) 		# "One-hot" encoded vector (00001000000)
# SUCCESS metrics
# Loss function to determine how bad is the model
cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))		# If Y is 1, log(Y) = 0, if Y is 0, log(Y) = -infinite
# % of correct answers found in batch
is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))	# "One-hot" decoding here
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))	# Measure success rate (between 0 and 1)
# Training step
alpha = 0.003 # Defining the learning rate here
optimizer = tf.train.GradientDescentOptimizer(alpha)
train_step = optimizer.minimize(cross_entropy)
# RUN code here
init = tf.global_variables_initializer()
sess = tf.Session() 								# Code in tf is not computed until RUN
sess.run(init)
# Training loop
for i in range(2100):
	# Load batch of images and correct answers
	batch_X, batch_Y = mnist.train.next_batch(batch_size)	# Train on mini_batches of 100 images
	train_data = {X: batch_X, Y_: batch_Y}
	# Train
	sess.run(train_step, feed_dict = train_data)
	# {X: batch_X, Y_: batch_Y}
	if i != 0 and i%100 == 0:
		# Success?
		a, c = sess.run([accuracy, cross_entropy], feed_dict = train_data)
		print("On training:")
		print("     A:", a)
		# Success on test data?
		test_data = {X: mnist.test.images, Y_: mnist.test.labels}
		a, c = sess.run([accuracy, cross_entropy], feed_dict = test_data)
		print("On testing:")
		print("     A:", a)