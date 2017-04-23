# import tensorflow as tf 

# x = 8.0

# Y = tf.nn.sigmoid(x)

# init = tf.global_variables_initializer()

# sess = tf.Session()
# sess.run(init)
# print(sess.run(Y))
# print("hi")
import numpy as np

def my_function(x):
	x = float(x)
	y = (np.sqrt(x))+(x/4) if x>0 else 0

	return y

print(my_function(4.5))