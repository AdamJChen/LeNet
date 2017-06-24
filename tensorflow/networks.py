import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as xavier_init

class Conv_net:
	"""Class conv_net contains layers required for building a 
	    Convolutional Network (convolutional layer and max pooling layer).
	"""

	@staticmethod
	def add_conv(X, shape, strides, padding = 'SAME'):
		'''Create convolution layer with relu activation. Kernel weights and biases are 
		initalized with Xavier initialization and zero respectivly.'''
		kernel = tf.get_variable('kernel',shape = shape, initializer = xavier_init())
		conv = tf.nn.conv2d(input = X, filter = kernel, strides = strides, padding = padding)
		bias = tf.get_variable('bias',shape = [shape[-1]], 
								initializer = tf.constant_initializer(value = 0.0))
		relu = tf.nn.relu(features = conv + bias)
		return relu

	@staticmethod
	def add_max_pool(X, window_size, strides, padding = 'VALID'):
		'''Create max pool layer'''
		return tf.nn.max_pool(value = X, ksize = window_size, strides = strides, padding =  padding)


class Feed_forward:
	"""Class Feed_forward can be used for building feed forward networks or the fully 
	   layer in a convolutional net."""

	@staticmethod
	def add_fc(X, n_in, n_out, act_func = True):
		'''Create feed forward layer. Weights and biases are initalized with
		Xavier initialization and zero respectivly. If act_func = True then
		relu is used and non-linear layer is returned. If act_func = False then logits are 
		returned for softmax classification.'''
		weights = tf.get_variable('weights', shape = [n_in, n_out], initializer = xavier_init())
		biases = tf.get_variable('bias', shape = [n_out], 
								  initializer = tf.constant_initializer(value = 0.0))
		z = tf.matmul(X, weights) + biases
		if act_func:
			return tf.nn.relu(features = z)
		else:
			return z


class Utes:
	"""Class Utes contains various utilities for training models"""

	@staticmethod
	def predictions(logits):
		"""Returns the predictions after softmax classification.
			Output is of type int32.
			Takes the argument Logits which is the outputs of the model."""
		y_hat = tf.nn.softmax(logits = logits)
		predictions = tf.argmax(input = y_hat, axis =  1)
		return tf.to_int32(predictions)
	
	@staticmethod
	def accuracy(labels, predictions, n_classes):
		"""Returns the accuracy of a model on a train or test set.
		   Labels and Predictions are of type int32.
		   Labels and predictions are assumed to not be one-hot encoded. """
		equality = tf.equal(x = predictions, y = labels) # match the type of labels
		return  tf.reduce_mean(tf.cast(equality, tf.float32))