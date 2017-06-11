import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from networks import Conv_net


# get CIFAR10 data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# format data
def std_img(data): return (data - np.mean(data)/255.0)
X_train = std_img(X_train)
X_test = std_img(X_test)
y_train = y_train[:,0] #remove a dimension for one hot encoding
y_test = y_test[:,0]

# parameters
image_width = X_train.shape[1]
image_depth = X_train.shape[3] 
shapes = {'c1' : [5, 5, 3, 20], 'c2' : [5, 5, 20, 50]}
strides = {'c' : [1, 1 ,1 ,1], 'p' : [1, 2, 2, 1]}
pool_win_size = [1, 2, 2, 1]
n_cells = 500
n_classes = 10
learning_rate = 0.01
batch_size = 100
n_epochs = 20


# LeNet Model
def leNet(X):
	"""Defines the LeNet architecture for image recognition. Returns logits for prediction or training"""
	with tf.variable_scope('layer1') as scope:
		conv1 = Conv_net.add_conv(X = X, shape = shapes['c1'], strides = strides['c'])
		pool1 = Conv_net.add_max_pool(X = conv1, window_size = pool_win_size, strides = strides['p'])

	with tf.variable_scope('layer2') as scope:
		conv2 = Conv_net.add_conv(X = pool1, shape = shapes['c2'], strides = strides['c'])
		pool2 = Conv_net.add_max_pool(X = conv2, window_size = pool_win_size , strides = strides['p'])

	#FIXed to BATCH SIZE
	flat = tf.reshape(pool2, (batch_size, -1)) # flattens volume into plane for feed forward layers, X.shape[0] is none

	with tf.variable_scope('layer3') as scope:
		fc = Conv_net.add_fc( X = flat, n_in = 3200, n_out = n_cells, act_func = True) # n_in needs to be int

	with tf.variable_scope('layer4') as scope:
		return Conv_net.add_fc( X = fc, n_in = n_cells, n_out = n_classes, act_func = False)


def accuracy(labels, logits, n_classes):
	"""Returns the accuracy of given examples"""
	predictions = tf.nn.softmax(logits)
	one_hot_labels = tf.one_hot(indices = labels, depth = n_classes)
	return tf.metrics.accuracy(labels = one_hot_labels, predictions = predictions)


# graph construction
graph = tf.Graph()

with graph.as_default():
	#data operations
	X = tf.placeholder(dtype = tf.float32, shape = (None,image_width, image_width, image_depth))
	y = tf.placeholder(tf.int32, shape=(None,))

	# train model
	model = leNet(X)
	element_wise_cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = model, labels = y)
	cost = tf.reduce_mean(element_wise_cost)	
	optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
	train = optimizer.minimize(cost)
	
	# evaluate accuracy
	train_accuracy = accuracy(y, model, n_classes =n_classes)
	test_accuracy = accuracy(y, model, n_classes =n_classes)


# graph  execution
with tf.Session(graph = graph) as sess:
	
	# initialize all variables
	sess.run(tf.global_variables_initializer())

	for epoch in range(n_epochs):
		for batch in range(X_train.shape[0]/ batch_size):
			# define offset and end of batch
			offset = (batch * batch_size)
			end_batch = offset + batch_size
			# get batch
			batch_data = X_train[offset: end_batch,:, :, :]
			batch_labels = y_train[offset: end_batch]
		# batch_data = X_train[0:5]
		# batch_labels = y_train[0:5]
			feed_dict = {X: batch_data, y: batch_labels}
			_, cost_value = sess.run([train, cost], feed_dict = feed_dict)
		print 'batch number: %d cost: %f' % (batch, cost_value)
