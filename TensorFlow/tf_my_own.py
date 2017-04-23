# Code extracted from the first neural network of this tutorial:
# https://www.youtube.com/watch?v=vq2nnJ4g6N0&list=WL&index=4&t=464s
# First NN from scratch

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
import numpy as np

def my_function(x):
	x = float(x)
	y = (np.sqrt(x))+(x/4) if x > 0 else 0

	return y

batch_size = 100
K = 200
L = 100
M = 60
N = 30

W1 = tf.Variable(tf.truncated_normal([28*28, K], stddev = 0.1))
B1 = tf.Variable(tf.zeros([K]))
W2 = tf.Variable(tf.truncated_normal([K, L], stddev = 0.1))
B2 = tf.Variable(tf.zeros([L]))
W3 = tf.Variable(tf.truncated_normal([L, M], stddev = 0.1))
B3 = tf.Variable(tf.zeros([M]))
W4 = tf.Variable(tf.truncated_normal([M, N], stddev = 0.1))
B4 = tf.Variable(tf.zeros([N]))
W5 = tf.Variable(tf.truncated_normal([N, 10], stddev = 0.1))
B5 = tf.Variable(tf.zeros([10]))

X = tf.placeholder(tf.float32, [None, 28 * 28]) 	# One layer because 28*28 gray-scaled images, the None will become the batch size
X = tf.reshape(X, [-1, 28*28])

# Defining the model - changing relu by my_function
Y1 = tf.nn.relu(tf.matmul(X, W1) + B1)
Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3)
Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4)
Y5 = tf.nn.softmax(tf.matmul(Y4, W5) + B5)

init = tf.global_variables_initializer()

# Defining the placeholder for correct answers
Y_ = tf.placeholder(tf.float32, [None, 10]) 		# "One-hot" encoded vector (00001000000)

# SUCCESS metrics
# Loss function to determine how bad is the model
cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y5))		# If Y is 1, log(Y) = 0, if Y is 0, log(Y) = -infinite
# % of correct answers found in batch
is_correct = tf.equal(tf.argmax(Y5, 1), tf.argmax(Y_, 1))	# "One-hot" decoding here
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# Training step
alpha = 0.003 # Defining the learning rate here
optimizer = tf.train.GradientDescentOptimizer(alpha)
train_step = optimizer.minimize(cross_entropy)

# RUN the fuckin' code here
sess = tf.Session() 								# Code in tf is not computed until RUN
sess.run(init)
# Training loop
for i in range(1100):
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
		print("     C:", c)
		# Success on test data?
		test_data = {X: mnist.test.images, Y_: mnist.test.labels}
		a, c = sess.run([accuracy, cross_entropy], feed_dict = test_data)
		print("On testing:")
		print("     A:", a)
		print("     C:", c)
		print(Y5)