import numpy as np 

x = np.array([[1,1,1,0,0, 0,1,1,1,0],[0,0,1,1,1,0,0,1,1,0],
				[1,1,1,0,0,0,1,1,1,0],[0,0,1,1,1,0,0,1,1,0],
				[1,1,1,0,0,0,1,1,1,0],[0,0,1,1,1,0,0,1,1,0],
				[1,1,1,0,0,0,1,1,1,0],[0,0,1,1,1,0,0,1,1,0],
				[1,1,1,0,0,0,1,1,1,0],[0,0,1,1,1,0,0,1,1,0]])

w = np.array([[1,0,1],[0,1,0],[1,0,1]])
b = 1

def relu(x, deriv = False):
	if deriv:
		return np.sign(self.relu(x))
	return np.maximum(x,0)

def pooling(x, patch, max = True):
	dims = np.ceil(np.array(x.shape)/patch).astype(int)
	pool = np.zeros((dims))
	print(x,pool)

	for i in range(pool.shape[0]):
		j = i*patch
		j_ = j+patch
		# Prevent out of range errors
		if j == x.shape[0]: break
		if j_ > x.shape[0]: j_ = x.shape[0]

		for k in range(pool.shape[1]):
			l = k*patch
			l_ = l+patch
			# Prevent out of range errors
			if l == x.shape[1]: break
			if l_ > x.shape[1]: l_ = x.shape[1]
			print("i",i, "k",k)
			print("j, j_",j, j_, "l, l_",l, l_)

			print(x[j:j_, l:l_])
			pool[i,k] = np.amax(x[j:j_, l:l_])


	return pool 

def conv(w,x,b):
	convolved = np.zeros(np.array(x.shape)-2)
	# i will be the left corner, j the right one
	for i in range(len(x)+1-len(w)):
		j = i+len(w)
		for k in range(len(x[0,:])+1-len(w[0,:])):
			l = k+len(w[0,:])
			# print("len(w[0,:])", len(w[0,:]))
			# print("i",i, "j",j)
			# print("k",k, "l",l)
			# print(x[i:j, k:l])

			convolved[i,k] = np.sum(w * x[i:j, k:l])
			#print("")
	return convolved+b

# ConvoBlock

print(x, x.shape)
print("")
print(w, w.shape)
print("")
conv2d = conv(w,x,b)
activ = relu(conv2d)
pooled = pooling(activ, 2)
print(conv2d)
print("")
print(activ)
print("")
print(pooled)