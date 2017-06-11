import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer 

class Conv_net:
	"""Class conv_net contains all the layers required for building a 
	    Convolutional Network (convolutional, max pooling, and fully connected
	    layers). 
	"""

	@staticmethod
	def add_conv(X, shape, strides, padding = 'SAME'):
		'''Create convolution layer with relu activation. Kernel weights and biases are 
		initalized with Xavier initialization and zero respectivly.'''
		kernel = tf.get_variable('kernel',shape = shape, initializer = xavier_initializer())
		conv = tf.nn.conv2d(input = X, filter = kernel, strides = strides, padding = padding)
		bias = tf.get_variable('bias',shape = [shape[-1]], 
								initializer = tf.constant_initializer(value = 0.0))
		relu = tf.nn.relu(features = conv + bias)
		return relu

	@staticmethod
	def add_max_pool(X, window_size, strides, padding = 'VALID'):
		'''Create max pool layer'''
		return tf.nn.max_pool(value = X, ksize = window_size, strides = strides, padding =  padding)

	@staticmethod
	def add_fc(X, n_in, n_out, act_func = True):
		'''Create feed forward layer. Weights and biases are initalized with
		Xavier initialization and zero respectivly. If act_func = True then
		relu is used and non-linear layer is returned. If act_func = False then logits are 
		returned for softmax classification.'''
		weights = tf.get_variable('weights', shape = [n_in, n_out], initializer = xavier_initializer())
		biases = tf.get_variable('bias', shape = [n_out], 
								  initializer = tf.constant_initializer(value = 0.0))
		z = tf.matmul(X, weights) + biases
		if act_func:
			return tf.nn.relu(features = z)
		else:
			return z
